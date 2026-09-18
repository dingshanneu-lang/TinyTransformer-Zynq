# program_bitstream_only.tcl
# Load ONLY the new HLS bitstream into PL over JTAG. Does NOT touch the APU,
# does NOT reset the system, so the running Linux (PS) stays alive.
#
# Usage:
#   E:\XilinxTools\Vitis\2024.2\bin\xsct.bat program_bitstream_only.tcl
#
# Or from a Vitis XSCT console:
#   source D:/ZynqTinyTransformerClassification/TCPServer/program_bitstream_only.tcl
#
# NOTE: Do NOT use "Run as Hardware" - that resets the APU and kills Linux.

set bitstream "D:/ZynqTinyTransformerClassification/TCPServer/TCPServer/TCPServer.runs/impl_1/TCPServer_wrapper.bit"

connect -url tcp:127.0.0.1:3121
puts "connected targets:"
targets

puts "=== Programming PL with new HLS bitstream ==="
fpga -file $bitstream
after 2000

puts "=== Done. PL now has the new xform_core (0x43C00000). Linux untouched. ==="