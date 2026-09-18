#!/usr/bin/env python3
"""
TinyTransformer-Zynq PC 上位机客户端
与 FPGA 端 Zynq Server (transformclient) 通讯
协议定义参考: network.h / network.c
"""

import socket
import struct
import numpy as np
from PIL import Image
import argparse
import sys
import os
import time
from pathlib import Path

# --------------------------------------------------------------------------
# 特征提取: 复用训练时的 Backbone (MobileNetV2 features + proj) + tf_val + scale
# 保证 PC 端提特征与训练/量化完全一致, 否则 FPGA 分类全错。
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

_BACKBONE = None       # (bb, scale, device)
_FEATURE_BACKEND = None


def _load_backbone(device="cpu", ckpt=None):
    """加载训练好的 Backbone (mobilenet_v2 features + proj) 与特征缩放 scale。"""
    global _BACKBONE
    if _BACKBONE is not None:
        return _BACKBONE

    import torch
    from train_v2 import Backbone

    if ckpt is None:
        ckpt = ROOT / "weights" / "best_final.pth"
    ckpt = Path(ckpt)
    if not ckpt.exists():
        raise FileNotFoundError(
            f"未找到特征提取权重 {ckpt}，请先用 train_final.py 训练。"
        )

    ck = torch.load(str(ckpt), map_location=device)
    bb = Backbone().to(device)
    bb.proj.load_state_dict(ck["proj"])
    bb.eval()
    scale = float(ck["scale"])
    _BACKBONE = (bb, scale, device)
    return _BACKBONE


def extract_features(image_path, device="cpu", ckpt=None):
    """
    图片 -> 16 维 Q12 int16 特征 (4x4)。

    完整链路 (与 train_final.py / eval_fixed.py 一致):
        tf_val(图片) -> Backbone(224x224) -> (4,4) float
        -> * scale -> * 4096 (Q12) -> clip -> int16
    """
    import torch
    from train_v2 import tf_val

    bb, scale, dev = _load_backbone(device, ckpt)
    img = Image.open(image_path).convert("RGB")
    x = tf_val(img).unsqueeze(0).to(dev)
    with torch.no_grad():
        f = bb(x) * scale                       # (1, 4, 4) float
    fq = np.clip(f.cpu().numpy() * 4096.0, -32768, 32767).astype(np.int16)
    return fq.reshape(MAX_SEQ_LEN, MAX_EMBED_DIM)

# 兼容旧调用名
EMBED_MODELS_AVAILABLE = False
EmbedModel = None


def create_embed_model(*args, **kwargs):
    raise RuntimeError("已改用 train_v2.Backbone 提特征，请用 extract_features()")

# ========== 协议常量 (与 network.h 严格保持一致) ==========
PROTOCOL_MAGIC = 0x54524E53  # "TRNS"
DEFAULT_PORT = 5000
MAX_SEQ_LEN = 4
MAX_EMBED_DIM = 4
MAX_PACKET_SIZE = 4096

PKT_TYPE_HELLO       = 0x01
PKT_TYPE_HELLO_ACK   = 0x02
PKT_TYPE_CONFIG      = 0x10
PKT_TYPE_CONFIG_ACK  = 0x11
PKT_TYPE_WEIGHTS     = 0x20
PKT_TYPE_WEIGHTS_ACK = 0x21
PKT_TYPE_INFERENCE   = 0x30
PKT_TYPE_RESULT      = 0x31
PKT_TYPE_ERROR       = 0xFF

# CRC32 表 (从 network.c 完整复制)
def _gen_crc32_table():
    tbl=[]
    for i in range(256):
        c=i
        for _ in range(8):
            c=(c>>1)^0xEDB88320 if (c&1) else (c>>1)
        tbl.append(c & 0xFFFFFFFF)
    return tbl

CRC32_TABLE = _gen_crc32_table()

def crc32_calc(data: bytes) -> int:
    crc = 0xFFFFFFFF
    for b in data:
        crc = CRC32_TABLE[(crc ^ b) & 0xFF] ^ (crc >> 8)
    return ~crc & 0xFFFFFFFF


class ProtocolError(Exception):
    pass


class TinyTransformerClient:
    """PC 端上位机客户端"""
    
    def __init__(self, host: str, port: int = DEFAULT_PORT, timeout: float = 5.0):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.sock: socket.socket | None = None
        self.seq = 0

    def connect(self) -> bool:
        """建立 TCP 连接"""
        try:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect((self.host, self.port))
            # 关闭 Nagle 算法，降低延迟
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            print(f"[+] Connected to {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"[-] Connection failed: {e}")
            return False

    def close(self):
        if self.sock:
            self.sock.close()
            self.sock = None

    def _send_all(self, data: bytes):
        """发送所有数据，处理部分发送"""
        sent = 0
        while sent < len(data):
            n = self.sock.send(data[sent:])
            if n <= 0:
                raise ProtocolError("Socket send returned 0")
            sent += n

    def _recv_exact(self, n: int) -> bytes:
        """精确接收 n 字节"""
        data = b''
        while len(data) < n:
            chunk = self.sock.recv(n - len(data))
            if not chunk:
                raise ProtocolError("Connection closed by peer")
            data += chunk
        return data

    def _send_packet(self, pkt_type: int, payload: bytes = b'') -> bool:
        """发送完整数据包 (含头部 CRC)"""
        length = len(payload)
        # 先打包头部 (crc32 占位为 0)
        hdr = struct.pack('<IHHII', PROTOCOL_MAGIC, pkt_type, self.seq, length, 0)
        # 计算 payload 的 CRC32
        crc = crc32_calc(payload)
        # 重新打包头部
        hdr = struct.pack('<IHHII', PROTOCOL_MAGIC, pkt_type, self.seq, length, crc)
        try:
            self._send_all(hdr + payload)
            return True
        except Exception as e:
            print(f"[-] Send failed: {e}")
            return False

    def _recv_packet(self):
        """接收完整数据包，验证 CRC"""
        # 读取头部 (16 字节)
        hdr_data = self._recv_exact(16)
        magic, pkt_type, seq, length, crc = struct.unpack('<IHHII', hdr_data)
        
        if magic != PROTOCOL_MAGIC:
            raise ProtocolError(f"Invalid magic: 0x{magic:08x}")
        
        # 读取 payload
        payload = self._recv_exact(length) if length > 0 else b''
        
        # 验证 CRC
        if crc32_calc(payload) != crc:
            raise ProtocolError(f"CRC mismatch: expected 0x{crc:08x}, got 0x{crc32_calc(payload):08x}")
        
        return pkt_type, seq, payload

    def hello(self) -> bool:
        """握手"""
        print("[*] Sending HELLO...")
        if not self._send_packet(PKT_TYPE_HELLO):
            return False
        pkt_type, seq, payload = self._recv_packet()
        if pkt_type == PKT_TYPE_HELLO_ACK:
            print("[+] HELLO_ACK received")
            self.seq += 1
            return True
        print(f"[-] Unexpected response: type=0x{pkt_type:02x}")
        return False

    def send_config(self, num_layers=2, seq_len=4, embed_dim=4, ffn_dim=16, num_heads=1) -> bool:
        """发送模型配置"""
        print(f"[*] Sending CONFIG: layers={num_layers}, seq_len={seq_len}, embed={embed_dim}, ffn={ffn_dim}, heads={num_heads}")
        payload = struct.pack('<IIIII', num_layers, seq_len, embed_dim, ffn_dim, num_heads)
        if not self._send_packet(PKT_TYPE_CONFIG, payload):
            return False
        pkt_type, seq, payload = self._recv_packet()
        if pkt_type == PKT_TYPE_CONFIG_ACK:
            print("[+] CONFIG_ACK received")
            self.seq += 1
            return True
        print(f"[-] Config failed: type=0x{pkt_type:02x}")
        return False

    def send_weights(self, weights_path: str) -> bool:
        """发送权重文件"""
        if not os.path.exists(weights_path):
            print(f"[-] Weights file not found: {weights_path}")
            return False
        
        with open(weights_path, 'rb') as f:
            weights_data = f.read()
        
        # 当前 HLS 核 (xform_core.cpp) 期望 sizeof(transformer_weights_t)
        #   = NUM_WEIGHTS(522) * 2 = 1044 bytes
        expected = 1044
        print(f"[*] Sending WEIGHTS: {len(weights_data)} bytes (expected {expected})")
        
        if not self._send_packet(PKT_TYPE_WEIGHTS, weights_data):
            return False
        
        pkt_type, seq, payload = self._recv_packet()
        if pkt_type == PKT_TYPE_WEIGHTS_ACK:
            print("[+] WEIGHTS_ACK received")
            self.seq += 1
            return True
        elif pkt_type == PKT_TYPE_ERROR:
            err_code, msg = struct.unpack('<I64s', payload)
            print(f"[-] Server error: code={err_code}, msg={msg.decode().strip()}")
        else:
            print(f"[-] Weights failed: type=0x{pkt_type:02x}")
        return False

    def inference(self, input_tensor: np.ndarray) -> np.ndarray | None:
        """执行推理
        input_tensor: (4, 4) int16 numpy array
        返回: (4, 4) int16 numpy array
        """
        if input_tensor.shape != (MAX_SEQ_LEN, MAX_EMBED_DIM):
            raise ValueError(f"Input must be {MAX_SEQ_LEN}x{MAX_EMBED_DIM}, got {input_tensor.shape}")
        if input_tensor.dtype != np.int16:
            input_tensor = input_tensor.astype(np.int16)
        
        payload = input_tensor.tobytes()  # 32 bytes
        print(f"[*] Sending INFERENCE (seq={self.seq})...")
        
        if not self._send_packet(PKT_TYPE_INFERENCE, payload):
            return None
        
        pkt_type, seq, payload = self._recv_packet()
        
        if pkt_type == PKT_TYPE_RESULT:
            # pkt_result_t: header + int16 data[4][4] + uint32 latency_us = 36 bytes payload
            output_data = payload[:32]
            latency_us = struct.unpack('<I', payload[32:36])[0] if len(payload) >= 36 else 0
            output = np.frombuffer(output_data, dtype=np.int16).reshape(MAX_SEQ_LEN, MAX_EMBED_DIM)
            print(f"[+] RESULT received, latency={latency_us} us")
            self.seq += 1
            return output
        elif pkt_type == PKT_TYPE_ERROR:
            err_code, msg = struct.unpack('<I64s', payload)
            print(f"[-] Server error: code={err_code}, msg={msg.decode().strip()}")
        else:
            print(f"[-] Unexpected response: type=0x{pkt_type:02x}")
        return None


# ========== 图片预处理 ==========
def preprocess_image(image_path: str, embed_model=None,
                     normalize_mode: str = '0_1', device: str = 'cpu',
                     ckpt=None) -> np.ndarray:
    """
    图片预处理: 直接复用训练特征链 (Backbone + tf_val + scale + Q12),
    返回 (4,4) int16, 即发给 FPGA 的 payload。

    embed_model / normalize_mode 仅为兼容旧调用签名保留, 不再使用。
    """
    features_int16 = extract_features(image_path, device=device, ckpt=ckpt)
    print(f"[+] Preprocessed: -> features(4x4) -> int16\n{features_int16}")
    return features_int16


def classify_output(output: np.ndarray) -> tuple:
    """
    解析 FPGA 输出为猫/狗分类概率
    假设输出 output[0,0] = cat logit, output[0,1] = dog logit
    """
    cat_logit = float(output[0, 0])
    dog_logit = float(output[0, 1])

    # 数值稳定的 Softmax (减去最大值防止溢出)
    m = max(cat_logit, dog_logit)
    exp_cat = np.exp(cat_logit - m)
    exp_dog = np.exp(dog_logit - m)
    total = exp_cat + exp_dog
    cat_prob = exp_cat / total
    dog_prob = exp_dog / total
    
    label = "猫" if cat_prob > dog_prob else "狗"
    confidence = max(cat_prob, dog_prob)
    
    return label, cat_prob, dog_prob, confidence


# ========== 主程序 ==========
def main():
    parser = argparse.ArgumentParser(
        description="TinyTransformer-Zynq PC 上位机客户端 - 猫狗分类演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python tiny_transformer_client.py cat.jpg --host 192.168.0.102
  python tiny_transformer_client.py dog.png --host 192.168.0.102 --skip-weights
  python tiny_transformer_client.py test.jpg --host 192.168.0.102 --weights ./weights.bin --no-config
        """
    )
    parser.add_argument('image', help='输入图片路径 (JPG/PNG)')
    parser.add_argument('--host', default='192.168.0.102', help='Zynq 服务器 IP (默认: 192.168.0.102)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT, help=f'端口 (默认: {DEFAULT_PORT})')
    parser.add_argument('--weights', default='xform_weights.bin', help='权重文件路径 (默认: xform_weights.bin)')
    parser.add_argument('--skip-weights', action='store_true', help='跳过发送权重 (FPGA 已预加载)')
    parser.add_argument('--no-config', action='store_true', help='跳过发送配置 (使用 FPGA 默认配置)')
    parser.add_argument('--timeout', type=float, default=5.0, help='Socket 超时秒数 (默认: 5.0)')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'],
                        help='特征提取设备 (默认: cpu)')
    parser.add_argument('--ckpt', default=None, help='特征权重 best_final.pth (默认: weights/best_final.pth)')

    args = parser.parse_args()

    # 检查图片
    if not os.path.exists(args.image):
        print(f"[-] Image not found: {args.image}")
        return 1

    # 解析权重文件路径 (支持相对路径)
    weights_path = args.weights
    if not os.path.isabs(weights_path):
        # 尝试在脚本目录查找
        script_dir = os.path.dirname(os.path.abspath(__file__))
        alt_path = os.path.join(script_dir, weights_path)
        if os.path.exists(alt_path):
            weights_path = alt_path
        else:
            # 尝试在 weights 子目录查找
            weights_dir_path = os.path.join(script_dir, "weights", weights_path)
            if os.path.exists(weights_dir_path):
                weights_path = weights_dir_path
                print(f"[*] Using weights from weights directory: {weights_path}")

    # 1. 图片预处理
    print(f"\n=== 步骤 1: 图片预处理 ===")
    print(f"Input: {args.image}")

    try:
        input_tensor = preprocess_image(args.image, device=args.device, ckpt=args.ckpt)
    except Exception as e:
        print(f"[-] Preprocessing failed: {e}")
        return 1

    # 2. 连接并通讯
    print(f"\n=== 步骤 2: 连接 Zynq Server ===")
    client = TinyTransformerClient(args.host, args.port, args.timeout)
    
    if not client.connect():
        return 1

    try:
        # 2.1 握手
        if not client.hello():
            return 1

        # 2.2 发送配置 (可选)
        if not args.no_config:
            if not client.send_config():
                return 1

        # 2.3 发送权重 (可选)
        if not args.skip_weights:
            if not client.send_weights(weights_path):
                print("[!] Weight send failed, continuing anyway...")

        # 2.4 推理
        print(f"\n=== 步骤 3: 推理 ===")
        output = client.inference(input_tensor)
        if output is None:
            print("[-] Inference failed")
            return 1

        print(f"\n[+] FPGA Output (int16):\n{output}")

        # 3. 解析分类结果
        print(f"\n=== 步骤 4: 结果解析 ===")
        label, cat_prob, dog_prob, confidence = classify_output(output)
        
        print(f"\n{'='*40}")
        print(f"  分类结果: {label}")
        print(f"  猫 概率: {cat_prob:.4f} ({cat_prob*100:.1f}%)")
        print(f"  狗 概率: {dog_prob:.4f} ({dog_prob*100:.1f}%)")
        print(f"  置信度:  {confidence:.4f} ({confidence*100:.1f}%)")
        print(f"{'='*40}")

    except ProtocolError as e:
        print(f"[-] Protocol error: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n[!] Interrupted by user")
        return 130
    except Exception as e:
        print(f"[-] Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        client.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())