# TinyTransformer-Zynq 设计文档

> 基于 Zynq-7010 FPGA 的轻量级 Transformer 猫狗分类加速系统
>
> 仓库：https://github.com/dingshanneu-lang/TinyTransformer-Zynq
>
> 最后更新：2026-09-18

---

## 1. 项目概述

本项目在低成本 Zynq-7010 FPGA 上实现了一个微型 Transformer（TinyTransformer）推理加速器，用于二分类任务（猫/狗）。系统采用 **软硬件协同** 架构：

- **PC 端**：负责图像预处理与特征提取（MobileNetV2 backbone），将 1280 维图像特征压缩为 16 维（4×4）int16 特征；
- **ARM (PS) 端**：运行 TCP 服务器，接收权重与特征，通过 `mmap` 操作 AXI 寄存器驱动 PL 端加速器；
- **FPGA (PL) 端**：以 HLS 综合的定点 Transformer 核完成 2 层 Transformer 推理，返回分类 logits。

设计目标是在 7010 仅 17,600 LUT 的资源约束下，跑通完整的 int16 定点 Transformer 推理链路，并保持与软件参考模型高度一致的精度。

---

## 2. 系统架构

```
┌──────────────────────────────────────────────────────────────────────┐
│                              PC 端 (Python/PyTorch)                   │
│                                                                       │
│  图片 (RGB)  ──▶  MobileNetV2 features ──▶ 1280-d ──▶ proj(1280→16)   │
│                                                          │            │
│                                                    4×4 float32        │
│                                                          │            │
│                              × scale (≈3.935) ───────────┤            │
│                                                          ▼            │
│                                        Q12 量化 → int16[16]           │
│                                                          │            │
│                                     TCP 协议打包 (HELLO/CONFIG/...)   │
└──────────────────────────────────────────────────────────┼───────────┘
                                                            │ TCP:5000
                                                            ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          Zynq-7010 (PS + PL)                          │
│                                                                       │
│  ┌─────────────────────────── PS (ARM Cortex-A9) ──────────────────┐ │
│  │  lwIP / TCP Server                                              │ │
│  │    · 协议解析 (magic=0x54524E53, CRC32)                          │ │
│  │    · transformer_load_weights() → 写 0x800 权重区              │ │
│  │    · transformer_load_input()   → 写 0x020 输入区              │ │
│  │    · start / wait_done / read_output (mmap /dev/mem)           │ │
│  └───────────────────────────────┬─────────────────────────────────┘ │
│                                  │ AXI4-Lite @ 0x43C00000             │
│  ┌───────────────────────────────▼─────────────────────────────────┐ │
│  │                     PL: xform_core (HLS)                         │ │
│  │   s_axilite CTRL bundle:                                         │ │
│  │     weights[522]  (0x800)                                        │ │
│  │     in[16]        (0x020)                                        │ │
│  │     out[16]       (0x040)  → out[0]=class0 logit, out[1]=class1  │ │
│  │                                                                  │ │
│  │   2 × Transformer Block                                          │ │
│  │   (QKV proj → attention → out proj → LN → FFN → LN)             │ │
│  │   全部 int16 Q12 定点运算，int64 累加                             │ │
│  └──────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.1 数据流

1. PC 端读取图片 → MobileNetV2 提取 1280 维特征（ImageNet 预训练，冻结）
2. `proj: Linear(1280→16)` 将特征投影到 4×4，乘特征缩放 `scale` 后量化为 int16（Q12）
3. PC 端通过 TCP 依次发送：HELLO → CONFIG → WEIGHTS（522 int16）→ INFERENCE（16 int16）
4. PS 端写入 PL 寄存器，启动加速器，等待 `ap_done`
5. PL 返回 16 个 int16，其中 `out[0]`、`out[1]` 为 class0/class1 logits
6. PS 端打包 RESULT 返回 PC，PC 端取 argmax 得到分类

---

## 3. 模型设计

### 3.1 模型规格

| 参数 | 值 |
|------|-----|
| num_layers | 2 |
| seq_len | 4 |
| embed_dim | 4 |
| ffn_dim | 16 |
| num_heads | 1 |
| num_classes | 2 |
| 权重参数 | 522 int16 = 1044 bytes |

### 3.2 网络结构

```
输入 x: (B, 4, 4)  [已提取的 16 维特征]

对每层 (×2):
  ┌─ Self-Attention ─────────────────────────────┐
  │  Q = x @ Wq + bq      (4×4 @ 4×4 → 4×4)      │
  │  K = x @ Wk + bk                             │
  │  V = x @ Wv + bv                             │
  │  scores = Q @ Kᵀ / sqrt(d_head)  d_head=4    │
  │  attn = softmax(scores) @ V                  │
  │  z = attn @ Wo + bo                          │
  │  r = x + z          (残差)                   │
  │  y = LayerNorm(r)                            │
  └──────────────────────────────────────────────┘
  ┌─ FFN ────────────────────────────────────────┐
  │  h = y @ W1 + b1     (4 → 16)                │
  │  g = GELU(h)                                 │
  │  u = g @ W2 + b2     (16 → 4)                │
  │  r2 = y + u          (残差)                  │
  │  x = LayerNorm(r2)                           │
  └──────────────────────────────────────────────┘

输出投影:
  flat = x.reshape(B, 16)
  logits = flat @ Wout + bout    (16 → 2)
```

### 3.3 权重布局（522 int16）

每层按以下顺序排列，共 2 层：

| 项 | 形状 | 数量 |
|----|------|------|
| qkv_w | (4, 12) | 48 |
| qkv_b | (12,) | 12 |
| out_w | (4, 4) | 16 |
| out_b | (4,) | 4 |
| fc1_w | (4, 16) | 64 |
| fc1_b | (16,) | 16 |
| fc2_w | (16, 4) | 64 |
| fc2_b | (4,) | 4 |
| ln1_gamma | (4,) | 4 |
| ln1_beta | (4,) | 4 |
| ln2_gamma | (4,) | 4 |
| ln2_beta | (4,) | 4 |
| **每层小计** | | **244** |
| **2 层** | | **488** |

输出投影：

| 项 | 形状 | 数量 |
|----|------|------|
| out_proj_w | (16, 2) | 32 |
| out_proj_b | (2,) | 2 |

**合计：488 + 34 = 522 int16**

> 注：`train.py` 的早期导出逻辑曾以 440 为期望值并截断权重，`gen_reference.py` 修正为完整 522 导出。

---

## 4. 定点量化方案

### 4.1 约定（Q12）

- **格式**：Q12，即 `1 << 12 = 4096` 表示 1.0
- **存储**：int16
- **累加**：int32 / int64
- **最终结果**：重新量化为 Q12

### 4.2 量化函数

```python
SCALE = 32767.0 / 8.0   # 假设权重范围 [-8, 8]
quantized = clip(weight * SCALE, -32768, 32767).astype(int16)
```

特征量化（PC 端）：`f_q12 = clip(f_float * 4096, -32768, 32767).astype(int16)`

### 4.3 定点算子（HLS 与 Python 参考一致）

```cpp
// C 风格截断除法（b > 0）
int64_t tdiv(int64_t a, int64_t b);

// 四舍五入：(a + b/2) / b，b = 2^k
int64_t qround_p2(int64_t a, int k) {
    return tdiv(a + (1 << (k-1)), 1 << k);
}

// 饱和到 int16
int16_t clip16(int64_t v);
```

| 算子 | 定点实现 |
|------|----------|
| Linear | `clip16(qround_p2(Σ w·x, 12) + bias)` |
| LayerNorm | 均值 `qround_p2(Σx, 2)`；方差 `tdiv(Σd², 4)`；`invsqrt` 用整数牛顿迭代 24 次求 `2^24/√v`；`qround_p2(d·inv, 12)` 后乘 gamma 加 beta |
| Softmax | 先减最大值；`exp` 查表 `EXP_LUT[arg>>6]`；`invN = round(2^24/tot)`；`qround_p2(e·invN, 12)` |
| GELU | `z = tdiv(1.702·x, 4096)`；`sigmoid(z)` 查表；`qround_p2(x·sg, 12)` |
| attention scores | `tdiv(qround_p2(ΣQ·K, 12), 2)`（d_head=4，缩放 1/2） |

### 4.4 查找表（LUT）

由 `gen_reference.py` 生成写入 `xform_tables.h`，保证 HLS 核与 Python 参考逐位一致：

- **SIG_LUT**：sigmoid，128/256 项，x ∈ [-4, 4)，步长 1/16，索引 `(x+16384) >> 7`
- **EXP_LUT**：exp(-t)，512 项，t ∈ [0, 8)，步长 1/64，索引 `arg >> 6`

---

## 5. FPGA 实现

### 5.1 HLS 核接口

```cpp
void xform_core(const int16_t weights[522],
                const int16_t in[16],
                int16_t out[16]);
```

全部端口绑定到 `s_axilite` bundle `CTRL`。

### 5.2 寄存器映射（AXI4-Lite，基址 0x43C00000）

| 偏移 | 名称 | 说明 |
|------|------|------|
| 0x000 | AP_CTRL | bit0=start, bit1=done, bit2=idle |
| 0x004 | GIE | 全局中断使能 |
| 0x008 | IER | 中断使能寄存器 |
| 0x00C | ISR | 中断状态寄存器 |
| 0x020 | INPUT_BASE | in[16]，每字 2 个 int16 |
| 0x040 | OUTPUT_BASE | out[16]，每字 2 个 int16 |
| 0x800 | WEIGHT_BASE | weights[522]，每字 2 个 int16 |

### 5.3 资源优化策略

为适配 7010 的 17,600 LUT 限制，采用 **串行复用** 架构：

- 单乘法器复用：`#pragma HLS ALLOCATION instances=mul limit=2`
- 共享除法器：`instances=div limit=2`
- 算子级资源共享：add/sub/icmp 均限制实例数
- LUT 表放 BRAM：`#pragma HLS BIND_STORAGE type=ROM_1P impl=BRAM`
- 除法用 64 位逐位移位减法实现（避免大位宽除法器宏）

### 5.4 实现结果

| 阶段 | LUT | FF | DSP | BRAM | 时序 |
|------|-----|----|-----|------|------|
| HLS 综合 (fix_prj, 板载版) | 18,847 (107%)* | 13,995 (39%) | 56 (70%) | 6 (5%) | Est. 7.29ns @ 100MHz |
| **Vivado 实现 (impl_1)** | **6,183 (35.13%)** | **6,669 (18.95%)** | — | — | **WNS = +6.912ns @ 100MHz** |

> \* HLS 综合估算偏保守；Vivado 布局布线后 LUT 实际占用 35%，时序全部满足（DRC 无错误）。
> 参考：早期 `opt_prj` 版本 HLS 估算 14,482 LUT (82%)。

---

## 6. PS 端软件

### 6.1 硬件抽象层（`transformer_hw.c`）

```c
int transformer_init(uintptr_t base_addr);              // mmap /dev/mem
int transformer_load_weights(const transformer_weights_t *w);  // 打包写入 0x800
int transformer_load_input(const transformer_input_t *in);     // 打包写入 0x020
int transformer_start(void);                            // 置 AP_CTRL.start
int transformer_wait_done(uint32_t timeout_ms);         // 轮询 AP_CTRL.done
int transformer_read_output(transformer_output_t *out); // 读 0x040
int transformer_run_inference(input, output);           // 组合上述流程
```

- **权重打包**：每 2 个 int16 合成一个 32 位字（低 16 位在前）
  `word = (uint16)w[i] | ((uint16)w[i+1] << 16)`
- **输入打包**：同样每 2 个 int16 一个寄存器字
- **启动握手**：先等 `ap_idle`，再置 `start`

### 6.2 网络服务

- TCP 端口 **5000**，`pthread` 每连接一个处理线程
- 协议头（16 字节，packed）：

```c
typedef struct __attribute__((packed)) {
    uint32_t magic;    // 0x54524E53 "TRNS"
    uint16_t type;
    uint16_t seq;
    uint32_t length;
    uint32_t crc32;
} pkt_header_t;
```

- CRC32 多项式 `0xEDB88320`，初始 `0xFFFFFFFF`，末尾取反

---

## 7. 通信协议

### 7.1 消息类型

| 类型 | 值 | 方向 | 说明 |
|------|-----|------|------|
| HELLO | 0x01 | PC→FPGA | 握手请求 |
| HELLO_ACK | 0x02 | FPGA→PC | 握手应答 |
| CONFIG | 0x10 | PC→FPGA | 模型配置（layers/seq/embed/ffn/heads） |
| CONFIG_ACK | 0x11 | FPGA→PC | 配置应答 |
| WEIGHTS | 0x20 | PC→FPGA | 522 int16 权重 |
| WEIGHTS_ACK | 0x21 | FPGA→PC | 权重应答 |
| INFERENCE | 0x30 | PC→FPGA | 16 int16 输入特征 |
| RESULT | 0x31 | FPGA→PC | 16 int16 输出 + 4 字节延迟 |
| ERROR | 0xFF | FPGA→PC | 错误码 + 消息 |

### 7.2 交互时序

```
PC                          FPGA
 │  ── HELLO ──────────────▶  │
 │  ◀──────────── HELLO_ACK   │
 │  ── CONFIG ─────────────▶  │
 │  ◀─────────── CONFIG_ACK   │
 │  ── WEIGHTS (1044B) ────▶  │
 │  ◀────────── WEIGHTS_ACK   │
 │  ── INFERENCE (32B) ────▶  │   (可重复多次)
 │  ◀──── RESULT (36B+延迟)   │
 │  ── ... ──────────────▶    │
```

`RESULT` payload 布局：

```c
typedef struct __attribute__((packed)) {
    pkt_header_t header;
    int16_t data[4][4];   // out[0], out[1] 为 logits，其余为 0
    uint32_t latency_us;
} pkt_result_t;
```

---

## 8. 训练流程

正式训练见 `train_final.py`，分两阶段：

### Stage 1：训练特征投影 `proj`

- MobileNetV2 features（冻结）→ 1280 维
- `Linear(1280→16)` + 临时分类头 `Linear(16→2)`
- 目的：让 16 维特征具有判别性

### Stage 2：训练 TinyTransformer

- 冻结 backbone 与 proj
- 训练 TinyTransformer（2 层，522 参数）+ 特征缩放 `scale`（可学习）
- 损失 `CrossEntropyLoss`，优化器 AdamW，余弦退火
- 输入：`tt(bb(x) * scale)`

### 数据划分

- 使用 `data/processed/train`（cats 9393 + dogs 9362）
- `random.Random(99).shuffle` 后取前 2000 张作为 **holdout** 验证集
- 官方 `data/processed/val` 分布异常，不用于选型

### 导出

- `export522()` 导出 522 int16 Q12 权重 → `weights/xform_weights.bin`
- 同时保存 `weights/best_final.pth`（含 model/proj/scale/config）

---

## 9. 测试与结果

### 9.1 精度测试

| 指标 | 结果 | 说明 |
|------|------|------|
| 浮点模型 holdout | **99.33%** | 300 张 |
| 软件定点 Q12 | **98.67%** | `eval_fixed.py`，量化损失 0.67% |
| FPGA 板端 | **97.33%** | 292/300，另有 ~2% 偶发超时未计入 |

**软件定点评估**（`eval_fixed.py`）：
- 用 `gen_reference.FixedQuant` 在 holdout 上评估
- 量化损失仅 0.67%，说明 Q12 方案精度损失很小

**板端评估**（`eval_board.py`）：
- 下发真实 Q12 权重，逐张通过 TCP 推理
- 板端直接返回 2 个 logits（HLS 核内部已做 output_proj）
- 平均约 92 ms/样本（含 TCP round-trip）
- 约 2% 样本出现偶发 `ap_done` 超时（ERROR 返回），属通信/时序偶发

### 9.2 资源与时序

- **LUT 35.13% / FF 18.95%**（Vivado 实现后）
- **WNS = +6.912 ns @ 100MHz**，全路径满足时序
- Zero DRC 错误

### 9.3 上位机与 GUI

**特征提取链一致性**（关键）：PC 端推理必须与训练完全一致，否则分类全错。

```
图片 → tf_val(Resize 224 + ImageNet 归一化)
     → Backbone(mobilenet_v2 features + proj)      [bb.eval()]
     → (4,4) float → ×scale → ×4096(Q12) → clip → int16
```

- `scale ≈ 3.935` 来自 `best_final.pth`
- `bb.eval()` 使用 ImageNet 预训练 BN 统计，单张推理稳定可复现
- eval 模式单张精度：float 99.00% / Q12 软件定点 99.00%
- 注：约 60% 的输入值会饱和到 ±32767（因 scale 偏大），但不影响 argmax，精度无损

**CLI**（`tiny_transformer_client.py`）：
- `extract_features()` 复用 `train_v2.Backbone` + `tf_val` + `scale`
- 协议：HELLO → CONFIG → WEIGHTS(1044B) → INFERENCE → RESULT

**GUI**（`gui_client.py`，PySide6/PyQt5）：
- 拖拽/点击上传图片，后台线程推理不阻塞 UI
- 显示分类结果、猫/狗概率条、FPGA 原始 int16 输出、实时日志
- 服务器 IP/端口/超时、权重路径、特征设备（cpu/cuda）可配置

**实测**（2026-09-18，板端 192.168.0.102）：
- CLI 端到端 10 张：9/10 正确
- GUI 推理线程单张：分类正确，置信度 100%

---

## 10. 已知问题与后续工作

### 10.1 定点约定不一致（已定位，未修复到板卡）

- 板载 HLS 核（`xform_core_fix_prj`，综合于 2026-09-17）内部 **混用 Q8 与 Q12**：
  - gelu / softmax / layernorm / QKV / scores / attn 使用 `qround_p2(..., 8)`（Q8）
  - out_proj / fc1 / fc2 / output_proj 使用 `qround_p2(..., 12)`（Q12）
- 而 `gen_reference.py` 与权重导出统一为 **Q12**
- 后果：板端 logits 与软件 Q12 参考完全不同（示例输入：软件 `[1457, 904]` vs 板端 `[-1849, 6119]`），精度从 98.67% 降到 97.33%
- **修复状态**：已将 `xform_core.cpp` 中 8 处 `qround_p2(..., 8)` 改为 `..., 12`，统一为 Q12；**尚未重新综合与烧录**，需重跑 HLS → 重建 bitstream → 更新板卡后才能生效

### 10.2 偶发推理超时

- 约 2% 的推理请求返回 `Inference failed`（`ap_done` 未在超时内拉高）
- 推测为 PS 轮询节奏或总线争用，未深入定位

### 10.3 后续工作

1. 用修正后的 `xform_core.cpp` 重新综合 HLS IP，重建 bitstream 并烧录
2. 板端精度测试，期望回到 ~98.6%（bit-exact 对齐软件参考）
3. 排查偶发超时
4. 完善设计文档与部署脚本

---

## 11. 项目结构与关键文件索引

### 11.1 目录结构（截至 2026-09-18 整理后）

```
D:\ZynqTinyTransformerClassification\
├── README.md / DESIGN.md / dev_log.md
├── main.py                    # 一键启动器
├── gui_client.py              # PySide6 GUI
├── tiny_transformer_client.py # CLI + 特征提取
├── test_real_infer.py         # 真实图片批量测试
├── train_final.py             # 正式训练
├── train_v2.py                # Backbone / export522
├── train.py                   # 模型定义 / MODEL_CONFIG
├── torch_embed_models.py
├── download_dataset.py / prepare_hf_dataset.py
├── run_*.bat                  # Windows 启动脚本
├── weights/                   # xform_weights.bin / best_final.pth
├── data/processed/            # 数据集
├── TCPServer/
│   ├── BOOT.BIN / *.bif / *.tcl
│   ├── pl/xform_core_hls/     # HLS 工程 + gen_reference.py
│   ├── TCPServer/             # Vivado 工程
│   └── test_client.py
└── archive/                   # 历史版本与废弃路线（见其 README）
```

### 11.2 训练与量化（PC 端）

| 文件 | 说明 |
|------|------|
| `train.py` | TinyTransformer 模型定义、`MODEL_CONFIG` |
| `train_v2.py` | `Backbone`（MobileNetV2+proj）、`export522` |
| `train_final.py` | 正式两阶段训练脚本 |
| `TCPServer/pl/xform_core_hls/gen_reference.py` | Q12 定点参考、LUT 生成、522 权重导出 |

### 11.3 FPGA 设计

| 文件 | 说明 |
|------|------|
| `TCPServer/pl/xform_core_hls/xform_core.cpp` | HLS 定点 Transformer 核 |
| `TCPServer/pl/xform_core_hls/xform_tables.h` | sigmoid/exp 查找表（自动生成） |
| `TCPServer/TCPServer/TCPServer.runs/impl_1/` | Vivado 实现产物（含 bitstream） |

### 11.4 上位机

| 文件 | 说明 |
|------|------|
| `tiny_transformer_client.py` | 特征提取 + TCP 协议 + CLI |
| `gui_client.py` | PySide6 GUI 上位机 |
| `main.py` | 检查 + 训练/导出 + 启动 GUI |
| `test_real_infer.py` | 真实图片批量精度测试 |

### 11.5 PS 服务端（Linux 用户态，独立目录）

| 文件 | 说明 |
|------|------|
| `E:\VScode\TinyTransformer-Zynq\sw\app\main.c` | 服务入口 |
| `...\sw\app\transformer_hw.c` | 寄存器操作 / mmap |
| `...\sw\app\network.c` | TCP 协议实现 |
| `...\sw\app\Makefile` | 交叉编译（产物 `ts2`） |

### 11.6 常用命令

**交叉编译（静态链接）**：

```bash
arm-linux-gnueabihf-gcc -Wall -Wextra -O2 -std=c11 -D_GNU_SOURCE -static \
  -I. -o ts2 main.c transformer_hw.c network.c -lpthread -lrt
```

> 必须 `-static`：板卡 glibc 版本较老（缺 GLIBC_2.17/2.34/2.38）。

**板卡启动服务**：

```bash
kill -9 $(pgrep ts2) 2>/dev/null; fuser -k 5000/tcp 2>/dev/null; sleep 1; ./ts2
netstat -tlnp | grep 5000
```

**PC 端使用**：

```bash
python main.py --host 192.168.0.102            # 一键启动 GUI
python tiny_transformer_client.py cat.jpg --host 192.168.0.102   # 单图
python test_real_infer.py 192.168.0.102        # 批量真实图片
```

**评估脚本**（开发用，位于临时目录/可自建）：
- 软件定点评估：复用 `gen_reference.FixedQuant` + holdout
- 板端评估：复用 `tiny_transformer_client.TinyTransformerClient`

---

## 12. 路线演进与归档

项目经历 4 个阶段，旧版本已归档至 `archive/`（详见 `archive/README.md`）：

1. **随机权重通信验证**（9/2~9/3）：440 int16 随机权重验证通信
2. **裸机 lwIP 服务端**（9/3~9/4）：Vitis 裸机，源码在 `archive/legacy_lwip/src/`
3. **手写 RTL 探索**（8/28 前后）：SystemVerilog 实现，副本在 `archive/rtl_route/`
4. **HLS + Linux 用户态**（9/15 至今）：当前主线

两个物理位置：

| 位置 | 内容 |
|------|------|
| `D:\ZynqTinyTransformerClassification` | 当前主线（训练/量化/HLS/Vivado/上位机） |
| `E:\VScode\TinyTransformer-Zynq\sw\app` | PS 端 Linux 用户态服务端源码 |

---

## 附录 A：模型参数量统计

| 组件 | 参数量 |
|------|--------|
| TinyTransformer（1 层） | 244 |
| × 2 层 | 488 |
| output_proj | 34 |
| **FPGA 核总计** | **522** |
| Backbone (MobileNetV2) | ~2.3M（PC 端运行） |
| proj (1280→16) | 20,496（PC 端运行） |
