#!/usr/bin/env python3
"""
主启动程序：检查 FPGA、数据集、权重 -> 训练/加载 -> 启动 GUI
"""

import sys
import os
import subprocess
import time
import argparse
from pathlib import Path

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from tiny_transformer_client import TinyTransformerClient, DEFAULT_PORT


def check_fpga_connection(host: str, port: int, timeout: float = 3.0) -> bool:
    """Check FPGA server connection"""
    print(f"[CHECK] FPGA connection: {host}:{port} ...", end=" ", flush=True)
    client = TinyTransformerClient(host, port, timeout)
    if client.connect():
        if client.hello():
            print("[OK] Connected")
            client.close()
            return True
        else:
            print("[FAIL] Handshake failed")
    else:
        print("[FAIL] Connection failed")
    client.close()
    return False


def check_dataset() -> bool:
    """Check if dataset exists"""
    data_dir = PROJECT_ROOT / "data" / "processed"
    train_dir = data_dir / "train"
    val_dir = data_dir / "val"
    
    if train_dir.exists() and val_dir.exists():
        cat_count = len(list((train_dir / "cats").glob("*.jpg"))) if (train_dir / "cats").exists() else 0
        dog_count = len(list((train_dir / "dogs").glob("*.jpg"))) if (train_dir / "dogs").exists() else 0
        if cat_count > 0 and dog_count > 0:
            print(f"[CHECK] Dataset: OK (cats:{cat_count}, dogs:{dog_count})")
            return True
    
    print("[CHECK] Dataset: MISSING")
    return False


def download_dataset(auto: bool = False) -> bool:
    """下载并准备数据集"""
    print("[操作] 下载数据集...")
    
    script = PROJECT_ROOT / "download_dataset.py"
    if not script.exists():
        print("  ✗ download_dataset.py 不存在")
        return False
    
    cmd = [sys.executable, str(script)]
    if auto:
        cmd.append("--manual")  # 手动模式，假设用户已放好 zip 文件
    
    try:
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=300)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ✗ 下载超时")
        return False
    except Exception as e:
        print(f"  ✗ 下载失败: {e}")
        return False


def check_weights() -> bool:
    """Check weight files"""
    weights_dir = PROJECT_ROOT / "weights"
    weights_file = weights_dir / "xform_weights.bin"
    best_model = weights_dir / "best_final.pth"

    if weights_file.exists() and weights_file.stat().st_size == 1044:
        print(f"[CHECK] FPGA weights: OK ({weights_file.stat().st_size} bytes)")
        return True

    if best_model.exists():
        print(f"[CHECK] PyTorch model: OK, need export FPGA weights")
        return "need_export"

    print("[CHECK] Weights: MISSING")
    return False


def train_model(embed_model: str = "random", epochs: int = 20, device: str = "auto") -> bool:
    """训练模型并导出权重 (使用 train_final.py: Backbone + proj + TinyTransformer)"""
    print(f"[操作] 训练模型 (epochs={epochs})...")

    script = PROJECT_ROOT / "train_final.py"
    if not script.exists():
        print("  ✗ train_final.py 不存在")
        return False

    if device == "auto":
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cmd = [
        sys.executable, str(script),
        "--epochs", str(epochs),
        "--out", "xform_weights.bin",
    ]

    try:
        print(f"  运行: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, timeout=3600)
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("  ✗ 训练超时")
        return False
    except Exception as e:
        print(f"  ✗ 训练失败: {e}")
        return False


def export_weights() -> bool:
    """从已训练的 best_final.pth 导出 522 int16 Q12 FPGA 权重"""
    print("[操作] 导出 FPGA 权重...")
    try:
        from pathlib import Path as _P
        import sys as _sys
        _sys.path.insert(0, str(PROJECT_ROOT))
        import torch
        from train import TinyTransformer, MODEL_CONFIG
        from train_v2 import export522

        ck = torch.load(PROJECT_ROOT / "weights" / "best_final.pth", map_location="cpu")
        tt = TinyTransformer(MODEL_CONFIG)
        tt.load_state_dict(ck["model"])
        w = export522(tt, _P(PROJECT_ROOT / "weights" / "xform_weights.bin"))
        print(f"[+] 导出 {w.size} int16 -> weights/xform_weights.bin")
        return True
    except Exception as e:
        print(f"  ✗ 导出失败: {e}")
        return False


def launch_gui(host: str, port: int):
    """启动 GUI"""
    print("[启动] 启动 GUI...")
    
    script = PROJECT_ROOT / "gui_client.py"
    if not script.exists():
        print("  ✗ gui_client.py 不存在")
        return False
    
    # 设置环境变量传递给 GUI
    env = os.environ.copy()
    env["FPGA_HOST"] = host
    env["FPGA_PORT"] = str(port)
    
    try:
        subprocess.run([sys.executable, str(script)], cwd=PROJECT_ROOT, env=env)
        return True
    except KeyboardInterrupt:
        print("\n[INFO] User interrupted")
        return True
    except Exception as e:
        print(f"  [FAIL] GUI launch failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="TinyTransformer-Zynq Cat Dog Classification - One-click Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Full auto-check flow
  python main.py --host 192.168.1.100 --skip-fpga-check  # Skip FPGA check
  python main.py --train --epochs 30 --embed mobilenetv2  # Force retrain
  python main.py --gui-only                   # Launch GUI directly
        """
    )
    parser.add_argument("--host", default="192.168.1.100", help="FPGA IP address")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="FPGA port")
    parser.add_argument("--skip-fpga-check", action="store_true", help="Skip FPGA connection check")
    parser.add_argument("--skip-dataset-check", action="store_true", help="Skip dataset check")
    parser.add_argument("--skip-weight-check", action="store_true", help="Skip weight check")
    parser.add_argument("--train", action="store_true", help="Force retrain")
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--embed", choices=["random", "mobilenetv2", "vit"], 
                       default="random", help="Embedding model type")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device")
    parser.add_argument("--gui-only", action="store_true", help="Launch GUI directly, skip all checks")
    parser.add_argument("--auto-download", action="store_true", help="Auto download dataset (requires kaggle)")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TinyTransformer-Zynq Cat Dog Classification - Launcher")
    print("=" * 60)
    
    # 1. Launch GUI only
    if args.gui_only:
        launch_gui(args.host, args.port)
        return 0
    
    # 2. Check FPGA connection
    if not args.skip_fpga_check:
        if not check_fpga_connection(args.host, args.port):
            print("\n[WARN] FPGA not connected, please check:")
            print("  1. FPGA board powered on")
            print("  2. Network cable connected")
            print("  3. IP address correct (current: {})".format(args.host))
            print("  4. FPGA server program running")
            
            choice = input("\nContinue to launch GUI? (y/N): ").strip().lower()
            if choice != 'y':
                print("Cancelled")
                return 1
    else:
        print("[SKIP] FPGA connection check")
    
    # 3. Check dataset
    if not args.skip_dataset_check:
        if not check_dataset():
            print("\n[ACTION] Dataset missing, downloading...")
            if not download_dataset(auto=not args.auto_download):
                print("  [FAIL] Dataset preparation failed")
                return 1
            
            # Re-check
            if not check_dataset():
                print("  [FAIL] Dataset still unavailable")
                return 1
    else:
        print("[SKIP] Dataset check")
    
    # 4. Check/train weights
    if not args.skip_weight_check:
        weight_status = check_weights()
        
        if args.train or weight_status is False:
            # Need training
            print("\n[ACTION] Starting training...")
            if not train_model(args.embed, args.epochs, args.device):
                print("  [FAIL] Training failed")
                return 1
            
            # Verify export
            if not check_weights():
                print("  [FAIL] Weight export verification failed")
                return 1
                
        elif weight_status == "need_export":
            # Has PyTorch model, need export
            print("\n[ACTION] Exporting FPGA weights...")
            if not export_weights():
                print("  [FAIL] Weight export failed")
                return 1
        else:
            print("[CHECK] Weights: READY")
    else:
        print("[SKIP] Weight check")
    
    # 5. Launch GUI
    print("\n" + "=" * 60)
    print("All checks passed, launching GUI...")
    print("=" * 60)
    
    launch_gui(args.host, args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())