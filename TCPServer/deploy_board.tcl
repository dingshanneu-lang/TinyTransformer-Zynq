# deploy_board.tcl
# One-shot deploy: program new HLS bitstream + load new lwIP_app.elf, then run.
# Usage:
#   E:\XilinxTools\Vitis\2024.2\bin\xsct.bat deploy_board.tcl
#
# Prereqs:
#   - Board powered on + JTAG (Xilinx hw_server) reachable at tcp:127.0.0.1:3121
#   - New bitstream built by replace_xform_core.tcl
#   - lwIP_app.elf rebuilt from lwIP_app/src (new HLS register map)

set bitstream "D:/ZynqTinyTransformerClassification/TCPServer/TCPServer/TCPServer.runs/impl_1/TCPServer_wrapper.bit"
set elf       "D:/ZynqTinyTransformerClassification/TCPServer/TCPServer/lwIP_app/Debug/lwIP_app.elf"
set xsa       "D:/ZynqTinyTransformerClassification/TCPServer/TCPServer/lwIP_Server/export/lwIP_Server/hw/TCPServer_wrapper.xsa"
set psinit    "D:/ZynqTinyTransformerClassification/TCPServer/TCPServer/lwIP_app/_ide/psinit/ps7_init.tcl"

puts "=== [clock format [clock seconds] -format %T] Deploy start ==="

connect -url tcp:127.0.0.1:3121

# 1) Load hardware platform + new bitstream (program PL)
targets -set -nocase -filter {name =~"APU*"}
rst -system
after 3000
targets -set -nocase -filter {name =~"APU*"}
fpga -file $bitstream
after 1000

# 2) Init PS clocks / mux (from the live Vivado project ps7_init)
targets -set -nocase -filter {name =~"APU*"}
source $psinit
ps7_init
ps7_post_config
after 2000

# 3) Reset processor to clear MMU, then download ELF
targets -set -nocase -filter {name =~ "*A9*#0"}
rst -processor
after 1000

configparams force-mem-access 1
dow $elf
configparams force-mem-access 0
con
after 500

targets -set -nocase -filter {name =~"APU*"}
puts "=== [clock format [clock seconds] -format %T] Deploy complete ==="
puts "=== Server should be listening on 192.168.1.10:5000 ==="