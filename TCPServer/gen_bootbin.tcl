# gen_bootbin.tcl - Generate BOOT.BIN from FSBL + bitstream + ELF
set fsbl "D:/ZynqTinyTransformerClassification/TCPServer/bootgen_work/fsbl.elf"
set bit  "D:/ZynqTinyTransformerClassification/TCPServer/bootgen_work/TCPServer_wrapper.bit"
set elf  "D:/ZynqTinyTransformerClassification/TCPServer/bootgen_work/lwIP_app.elf"
set out  "D:/ZynqTinyTransformerClassification/TCPServer/bootgen_work/BOOT.BIN"

# Write BIF file
set bif [open "D:/ZynqTinyTransformerClassification/TCPServer/bootgen_work/gen.bif" w]
puts $bif "the_ROM_image:"
puts $bif "\{"
puts $bif "\t\[bootloader\]$fsbl"
puts $bif "\t\[destination_device=pl\]$bit"
puts $bif "\t\[destination_device=cpu,exception_level=el-3,target=baremetal\]$elf"
puts $bif "\}"
close $bif

puts "BIF file generated"
puts "Use bootgen to create BOOT.BIN:"
puts "  bootgen -w -o $out gen.bif"
