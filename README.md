# TinyTransformer-Zynq 猫狗分类系统

基于 Zynq-7010 FPGA 的轻量级 Transformer 猫狗分类演示系统。PC 端负责图片预处理与特征提取（MobileNetV2 + proj），FPGA 端以 int16 定点执行 Transformer 推理。

> 完整设计说明见 [DESIGN.md](DESIGN.md)；历史版本与废弃路线见 [archive/README.md](archive/README.md)。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        PC 端 (Python/PyTorch)                   │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 图片输入  │──▶ │ MobileNetV2  │──▶ │ proj(1280→16) ×scale │  │
│  │ (RGB)    │    │ features     │    │ → Q12 int16 (4×4)    │  │
│  └──────────┘    └──────────────┘    └──────────┬───────────┘  │
└─────────────────────────────────────────────────┼──────────────┘
                                                  │ TCP:5000
                                                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      FPGA 端 (Zynq-7010)                        │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │ 接收特征  │──▶ │ xform_core   │──▶ │ out[0],out[1]        │  │
│  │ (32B)    │    │ (HLS, int16) │    │ = 2 个 logits        │  │
│  └──────────┘    └──────────────┘    └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 核心特性

- **FPGA 推理**：TinyTransformer 纯硬件实现（HLS 综合，int16 Q12 定点运算）
- **精度链路**：PC 端特征提取与训练完全一致，软件定点 98.67% / 板端 ~97%
- **一键启动**：自动检查 FPGA、数据集、权重 → 训练/导出 → 启动 GUI
- **可视化 GUI**：拖拽图片、实时日志、概率条显示
- **CLI 客户端**：单图推理与批量真实图片测试

## 目录结构

```
D:\ZynqTinyTransformerClassification\
├── README.md / DESIGN.md      # 文档
├── main.py                    # 一键启动器 (推荐入口)
├── gui_client.py              # PySide6 GUI 客户端
├── tiny_transformer_client.py # CLI 推理客户端 (特征提取 + TCP 协议)
├── test_real_infer.py         # 真实图片批量测试
├── train_final.py             # 正式两阶段训练 (推荐)
├── train_v2.py                # Backbone / 数据增强 / export522
├── train.py                   # 模型定义 (TinyTransformer, MODEL_CONFIG)
├── torch_embed_models.py      # 训练端嵌入模型 (train.py 使用)
├── prepare_hf_dataset.py      # 数据集准备
├── download_dataset.py        # Kaggle 数据集下载
├── requirements.txt
├── weights/
│   ├── xform_weights.bin      # FPGA 权重 (1044 bytes, 522 int16 Q12)
│   └── best_final.pth         # 训练检查点 (model + proj + scale)
├── data/processed/            # 数据集 (train/val, cats/dogs)
├── TCPServer/
│   ├── BOOT.BIN               # 板卡启动镜像
│   ├── pl/xform_core_hls/     # HLS 工程
│   │   ├── xform_core.cpp     # HLS 定点 Transformer 核
│   │   ├── gen_reference.py   # Q12 参考 + 权重导出 + LUT 生成
│   │   └── xform_tables.h     # sigmoid/exp 查找表
│   ├── TCPServer/             # Vivado 工程
│   └── test_client.py         # 协议测试客户端
└── archive/                   # 历史版本与废弃路线 (见其 README)
```

> PS 端 Linux 用户态服务端源码位于 `E:\VScode\TinyTransformer-Zynq\sw\app\`
> (`main.c` / `transformer_hw.c` / `network.c`，编译产物 `ts2`)。

## 快速开始

### 1. 环境准备

```bash
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\pip install PySide6
```

### 2. 准备数据集

```bash
# 方式 A：自动下载 (需配置 kaggle.json)
python download_dataset.py

# 方式 B：手动放入 data/raw/ 后
python download_dataset.py --manual
```

### 3. 训练并导出权重

```bash
# 正式训练 (MobileNetV2 backbone + proj + TinyTransformer, 522 int16)
python train_final.py --epochs 10

# 产物:
#   weights/xform_weights.bin  (522 int16 Q12, 1044 bytes)
#   weights/best_final.pth     (model + proj + scale)
```

### 4. 启动程序

| 场景 | 命令 |
|------|------|
| 一键启动 (检查 + GUI) | `python main.py --host 192.168.0.102` |
| 直接启动 GUI | `python main.py --gui-only --host 192.168.0.102` |
| CLI 单图推理 | `python tiny_transformer_client.py cat.jpg --host 192.168.0.102` |
| 真实图片批量测试 | `python test_real_infer.py 192.168.0.102` |

## FPGA 端规格

- **输入**：4×4 int16 (16 维特征, 32 bytes)，Q12
- **模型**：2 层 Transformer (embed_dim=4, seq_len=4, ffn_dim=16, num_heads=1)
- **输出**：`out[0]` = class0 (猫) logit，`out[1]` = class1 (狗) logit，其余为 0
- **权重**：522 int16 = 1044 bytes (Q12)
- **资源**：Vivado 实现后 LUT 35.13% / FF 18.95%，WNS +6.912ns @ 100MHz

### 协议包格式

```
HELLO → CONFIG → WEIGHTS(1044B) → INFERENCE(32B, 可多次) → RESULT
```

- Magic: `0x54524E53` ("TRNS")
- CRC32 校验 (多项式 `0xEDB88320`)
- 端口：5000

## 常见问题

### Q: 分类结果不对？
- 确认 `weights/xform_weights.bin` 是 1044 bytes (522 int16)
- 确认 `weights/best_final.pth` 存在（PC 端特征提取需要其中的 `proj` 和 `scale`）
- 特征提取链必须与训练一致：`tf_val → Backbone → ×scale → ×4096(Q12) → int16`

### Q: FPGA 连接失败？
- 确认板卡上电、网线连接、IP 正确（默认 `192.168.0.102`）
- 确认板卡上 `ts2` 服务已运行：`netstat -tlnp | grep 5000`
- 若端口被占：`kill -9 $(pgrep ts2); fuser -k 5000/tcp; sleep 1; ./ts2`

### Q: 权重大小不匹配？
- 当前 HLS 核期望 **522 int16 = 1044 bytes**
- 旧的 `weights.bin` (880 bytes) 已废弃，见 `archive/`

## 训练流程详解

`train_final.py` 分两阶段：

1. **Stage 1**：冻结 MobileNetV2 features，训练 `proj: Linear(1280→16)`（配临时分类头），使 16 维特征可分
2. **Stage 2**：冻结 backbone + proj，训练 TinyTransformer（2 层 522 参数）+ 特征缩放 `scale`

数据划分：`data/processed/train` 内 `random.Random(99).shuffle` 后取前 2000 张作为 holdout。

## 精度指标

| 指标 | 结果 |
|------|------|
| 浮点模型 holdout | 99.33% |
| 软件定点 Q12 | 98.67% |
| FPGA 板端 | ~97.33% |

> 注：板端与软件参考存在定点约定差异（Q8/Q12 混用），详见 DESIGN.md 第 10 节。

## 依赖版本

| 包 | 版本 |
|----|------|
| Python | 3.10+ |
| torch | 2.6.0 |
| torchvision | 0.21+ |
| numpy | 2.x |
| Pillow | 10.x |
| PySide6 | 6.x |

## 许可证

MIT License - 仅供学习研究使用。

## 致谢

- [Kaggle Dogs vs Cats](https://www.kaggle.com/c/dogs-vs-cats/data) 数据集
- PyTorch / torchvision
- PySide6 (Qt for Python)
