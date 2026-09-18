# TinyTransformer FPGA 硬件加速 — 设计文档

AX7010（Zynq-7020）上 2 层 TinyTransformer 分类推理的 PL 端硬件加速实现。
模型：embed_dim=4、seq_len=4、FFN 中间维=16、2 层、单头；输出 2 类 logits。

## 1. 系统架构

```
PC (Ubuntu/Cybersecurity 192.168.0.100.30)
   │  TCP 协议 (192.168.0.102:5000)
   ▼
Zynq PS (dual Cortex-A9)
   ├─ lwIP 服务器 (应用固件)
   ├─ transformer_hw 驱动（AXI-Lite 寄存器读写）
   ▼
AXI Interconnect (M_AXI_GP0)
   ▼
PL: xform_core (Vitis HLS 定点加速核, 基址 0x43C00000)
```

- 推理核由 **Vitis HLS 2024.2** 从 C++ 综合生成，替换原空模板 IP（`xilinx.com:hls:xform_core:1.0`）。
- PS 侧固件保留 lwIP 服务器 + GIC，仅驱动层针对新寄存器映射重写。
- 网络未变：PS 静态 IP 192.168.0.102/24，PC 192.168.0.100，端口 5000，GIC 初始化必须先于 lwip_init()。

## 2. 定点数值方案

全部数据以 **Q12（scale=4096）** int16 存储，中间累加 int64。

| 运算 | 实现 |
|---|---|
| 乘法后量化 | `acc=Σ x·w`(int64) → `qround(acc,b)=(acc+b/2)/b`，b=4096 |
| 除法 | `tdiv(a,b)`：正数截断、负数绝对值后截断再取负（C 除 / 向下取整，为保证与 Python 参考一致自实现） |
| 裁剪 | `clip16(x)` 到 [-32768,32767] |
| LayerNorm | μ=Σ/4；x-μ 以 Q12 计算；`var=Σ(x-μ)² /4`；逆标准差用 **24 次整数牛顿迭代**（`inv=round(2^24/√var)`）；`y=γ·x·inv/2^12 + β` |
| GELU | 近似 `x·sigmoid(1.702x)`，系数 6971/4096≈1.702，sigmoid 查表 |
| Sigmoid | 查 SIG_LUT（128 项，0.01 步长），线性插值 |
| Softmax | exp 查 EXP_LUT（512 项），`mx=max(scores)`；`arg=(mx-s_i)` 归一化；`invN=round(2^24/Σexp)` 后乘算 |
| Attention score | `scores[q][k]=Σ (Q[q]·K[k])/4096/2`（标度 /2，d_head=4） |

**关键保留（历史修复）**：
- `scores` 必须用 **int32**（Q/K 乘积累加后经 /4096/2 仍可能超出 int16，如 -37748 会回绕成 +27788，导致 softmax 选出错误位置）。这是 C 仿真 4/8 → 8/8 的根因修复。
- fc1/fc2/out/output_proj 的权重索引须按 torch `Linear: y=x·Wᵀ` 转置导出（见 §4），曾因索引反了导致 FC 层错误。

## 3. HLS 核

- 工程目录：`TCPServer\pl\xform_core_hls\`，源 `xform_core.cpp`、`xform_tables.h`（LUT），`tb.cpp`（自检）。
- 接口：`weights[522]`、`in[16]`、`out[16]`，全部 s_axilite（协议为 CSR，非深存储器流）；输出只有 out[0]、out[1] 有效。
- 局部数组/中间量全展开（ARRAY_PARTITION），DSP 乘加流水。
- 综合结果（xc7z020-clg400-1，100MHz=10ns）：**LUT 69% / FF 42% / DSP 63% / BRAM 23%，最坏估计时钟 7.25ns**（达标）。
- C 仿真：8 组测试向量与 Python 定点参考**逐位完全一致（8/8 PASS）**。

### 寄存器映射（HLS 生成，s_axi_CTRL）

| 地址 | 大小(字*32b) | 内容 |
|---|---|---|
| 0x000 | - | AP_CTRL（bit0 start，bit1 done，bit2 idle） |
| 0x004 | - | GIE |
| 0x008 | - | IER |
| 0x00C | - | ISR |
| 0x020 | 16 | in[16]（int16 存入各自 32b 寄存器） |
| 0x040 | 16 | out[16]（前 2 个为 logits） |
| 0x800 | 522 | weights[522]（int16） |

基地址 0x43C00000（4K 窗口）。旧模板 IP 的寄存器（REG_CTRL 0x00 等）全部废弃。

## 4. 权重布局与导出

权重文件 `weights\xform_weights.bin` **1044B（522×int16）**，由 `gen_reference.py` 从 `best_tinytransformer_fpga.pth`（随机初始化模型）导出。布局：

每层 244 个（×2 层）+ output_proj 34 = 522：

```
qkv_w[48]  (e*12 + k*4 + j,  k 0/1/2=Q/K/V)   [w: qkv_w[e*12+0*4+j]]
qkv_b[12]  (e, e+4, e+8) = bias Q/K/V
out_w[16]  out_b[4]
fc1_w[64]  fc1_b[16]
fc2_w[64]  fc2_b[4]
ln1_g[4]   ln1_b[4]   ln2_g[4]   ln2_b[4]     (2 层 × 244)
op_w[32]   op_b[2]                             (output_proj: logits[c]=Σ op_w[i*2+c]·x[i])
```

导出时对 torch 权重做 `.T`（因 `Linear` 是 `x·Wᵀ`），索引方向必须与 .T 后的 ravel 对应（见 §2）。

## 5. PS 侧软件（lwIP 应用）

- `transformer_hw.h/.c`：新寄存器宏（REG_AP_CTRL/IN 0x20/OUT 0x40/W 0x800），`transformer_weights_t` 改为 `int16_t w[522]`，输出 2 个 logits。
- `lwip_server.c`：移除 BUGBYPASS（之前直接回显输入作为结果），恢复真实 WEIGHTS 写入 + INFERENCE 调用；修正 ERROR 包 `header.length`（原为 4 但实际发送 84B，改为 68 = payload 长度）；用 XTime 计算 latency_us。
- `protocol.h`：`pkt_result_t` 改为 `logits[2] + latency_us`；新增 NUM_WEIGHTS=522、NUM_LOGITS=2。
- `test_client.py`：发送真实 1044B 权重；`--verify` 模式对 8 组参考向量逐组比对（期望 q12 输出）。

## 6. 构建与验证流程

### A. HLS 核（PC 上完成）
```
cd TCPServer\pl\xform_core_hls
python gen_reference.py          # 生成 xform_tables.h、weights_int16.bin、test_vectors.json
<把 tee.exe（Git usr/bin）加入 PATH>
E:\Xilinx\Vitis_HLS\2024.2\bin\vitis_hls.bat -f csim_only.tcl   # 8/8 PASS
E:\Xilinx\Vitis_HLS\2024.2\bin\vitis_hls.bat -f run_hls.tcl     # csynth + export IP
```
产物：`xform_core_prj\solution1\impl\ip\`（含 xilinx_com_hls_xform_core_1_0.zip）。

### B. Vivado 换 IP（需 FPGA 上电/有 Vivado）
```
vivado -mode batch -source TCPServer\pl\xform_core_hls\replace_xform_core.tcl
   [-tclargs <TCPServer.xpr 路径>]
```
脚本：打开 TCPServer.xpr → 移除旧 `xform_core_0` → 加入 HLS IP → 连 s_axi_CTRL/ACLK/ARESETN → 分配 0x43C00000/4K → validate → 综合 → 实现 → write_bitstream。

### C. Vitis 更新平台 + 固件
- 用新 bitstream 重建/更新硬件平台（Vitis 中 Update Hardware），确保 `xparameters.h` 含 `XPAR_XFORM_CORE_0_BASEADDR`（若接口名导入为 `_S_AXI_` 变体需调整 `transformer_hw.h`）。
- 重新编译 lwIP 应用，生成 BOOT.BIN（FSBL+bitstream+u-boot/app）。

### D. 端到端验证（FPGA 上电）
```
python test_client.py --verify 192.168.0.102
# 期望：8/8 exact match（参考=q12，硬件 logits 应逐位一致），并输出 latency
```

## 7. 已验证项 / 待办

已通过：
- HLS C 仿真 vs Python 定点参考 8/8 逐位匹配；
- HLS 综合时序/资源达标，IP 导出成功；
- PS 驱动/protocol/test 代码更新完成（静态一致性检查）。

待 FPGA 上电后验证：
- Vivado 换 IP 后 bitstream 重建；
- Vitis xparameters 宏名/基址确认与 `transform_hw.h` 中的 `XPAR_XFORM_CORE_0_BASEADDR` 一致；
- 端到端 8/8 验证 + latency 实测；
- 后续可训练真实模型生成正式权重（现有为随机初始化）。

## 8. 目录速览

```
TCPServer\pl\xform_core_hls\
   xform_core.cpp          HLS 核源（固定点）
   xform_tables.h          sigmoid/exp LUT（由 gen_reference.py 生成）
   tb.cpp                  C 仿真自检（绝对路径读 ref/）
   gen_reference.py        LUT+权重+测试向量生成器
   csim_only.tcl / run_hls.tcl
   ref\test_vectors.json   8 组向量（in/float/q12）
   ref\weights_int16.bin   522 int16
   xform_core_prj\...      HLS 工程 + 导出 IP
   replace_xform_core.tcl  Vivado 替换脚本
TCPServer\TCPServer\lwIP_app\src\
   transformer_hw.c/.h     AXI-Lite 驱动（新寄存器）
   lwip_server.c           TCP 服务器（真实 HW 调用）
   protocol.h              协议（result=logits[2]+latency）
TCPServer\test_client.py   端到端测试客户端
weights\xform_weights.bin  1044B 权重
```