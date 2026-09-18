2026-09-13 12:33:32 - [分析] Zynq 7010 LUT 超用根因确认
- 板卡: xc7z010clg400-1 (17,600 LUT)
- 现有 HLS variants 全部超限:
  * xform_core_prj: 14.5k LUT (HLS 82%) -> Vivado 24k+ (FAIL)
  * xform_core_a4_prj: 38k LUT (216%)
  * xform_core_pb_prj: 36.7k LUT (208%)
  * xform_core_variant_prj: 18.9k LUT (107%) <- 当前 Vivado 用的 IP
- 瓶颈: 64位 tdiv 除法器、invsqrt 24次迭代、softmax/layernorm 非线性逻辑大量 LUT

2026-09-13 12:33:32 - [决策] 重写极简顺序版 xform_core_minimal.cpp
- 策略: 仅保留线性矩阵乘 (MAC 串行复用)
- 所有非线性移到 PS (ARM Cortex-A9) 跑:
  * layernorm, softmax, gelu/sigmoid -> 软件实现
  * Q12 缩放用移位代替除法
- 单 MAC 复用: #pragma HLS ALLOCATION instances=mul limit=1
- 目标资源: <8k LUT, <20 DSP, 留足余量给 lwIP + PS

2026-09-13 12:33:32 - [下一步] 
1. 创建新 HLS 工程 xform_core_min_prj
2. 编写 xform_core_minimal.cpp
3. C 仿真对齐 gen_reference.py
4. 综合 -> 导出 IP -> 替换 Vivado BD 中 xform_core_0
5. 重新综合实现 -> 生成比特流
6. 配套修改 lwip_server.c: 推理流程拆分 HW/PS 两阶段
2026-09-13 15:00:49 - [HLS] xform_core_minimal 综合成功
- LUT: 11,326 (64%) <- 从 18.9k 降到 11.3k，终于放进 7010
- FF: 10,731 (30%)
- DSP: 11 (13%)
- BRAM: 2 (1%)
- Fmax: 120 MHz (满足 100MHz 要求)
- IP 已导出: ip_minimal.zip

2026-09-13 15:00:49 - [下一步] 
1. 将新 IP 复制到 Vivado IP repo
2. 更新 BD 中的 xform_core_0 实例
3. 重新综合实现 -> 生成比特流
2026-09-13 15:53:26 - [里程碑] 方案 A 启动：回退完整 IP + 优化到 <17.6k LUT
============================================================
现状回顾：
- minimal IP (仅 MAC): 7.5k LUT, 通过布局布线，但精度不匹配 (0/8)
- 原版完整 IP (variant_prj): 18.9k LUT (107%), 精度 bit-exact (8/8) 但超 LUT

方案 A 目标：在原版 xform_core.cpp 基础上省 1.3k LUT
优化手段（按效果排序）：
1. 除法/开根号强制共享: #pragma HLS ALLOCATION instances=div limit=1; sqrt limit=1
2. Softmax/Exp LUT 放 BRAM: #pragma HLS BIND_STORAGE type=RAM_1P_BRAM
3. 量化降级: Q12 -> Q10 (再省 ~15% LUT)

当前最新比特流: D:\pl_new.bit (minimal 版)
已备份: xform_core_min_prj_backup/

下一步：
- 修改 xform_core.cpp 加 ALLOCATION pragma
- HLS 综合 variant_prj (或新建 prj) 看 LUT
- 如达标 -> 导出 IP -> 替换 BD -> 重新实现 -> 生成 bitstream
- 重新导出 XSA -> Vitis 重建 lwIP_app -> 板卡验证
============================================================

2026-09-14 12:54:00 - [里程碑] 方案 A 完成：原版 xform_core.cpp 优化达标并成功烧录板卡
============================================================
HLS 优化成果：
- 在原版 xform_core.cpp 基础上添加 8 行 ALLOCATION pragma + 2 个 BIND_STORAGE
- 算子级资源共享: div/sqrt/mul/add/icmp 全部强制共享
- LUT 表放 BRAM: SIG_LUT/EXP_LUT 通过 #pragma HLS BIND_STORAGE impl=BRAM

HLS 综合结果 (xform_core_opt_prj)：
- LUT: 14,482 (82%)  <- 从 18,840 (107%) 降 4.3k，达标 < 17,600
- DSP: 39 (48%)
- BRAM: 4 (3%)
- FF: 10,213 (29%)
- Fmax: 139 MHz (满足 100MHz)

Vivado 实现结果 (impl_1)：
- LUT: 5,503 (31.27%)  实际布局后远低于 HLS 估算
- Slice: 2,215 (50.34%)
- DSP: 39 (48.75%)
- BRAM: 2.5 (4.17%)
- Timing: WNS = 7.729 ns @ 100MHz ✅ 全路径满足时序
- Zero DRC / Zero Critical Warnings

关键文件产出：
- Bitstream: D:\ZynqTinyTransformerClassification\TCPServer\TCPServer\TCPServer.runs\impl_1\TCPServer_wrapper.bit (2.0 MB)
- XSA: D:\ZynqTinyTransformerClassification\TCPServer\TCPServer\TCPServer_wrapper.xsa
- 板卡烧录: 12:54:34 完成，End of startup status: HIGH

当前状态：
✅ PL 侧完全就绪 (bitstream 已烧录)
🔄 PS 侧进行中：需用新 XSA 重建 Vitis 平台编译 lwIP_app.elf
  - 现有兼容 ELF: lwIP_app/Debug/lwIP_app.elf (2026/09/11) 可先验证 bitstream
  - 新平台 lwIP_Server_new 已生成，含完整 lwIP 源码
  - 下一步：下载 ELF → 板卡跑 8 条测试向量 → 确认 8/8 bit-exact

============================================================
