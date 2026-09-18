# run_hls.tcl - Vitis HLS flow for xform_core
# Usage: vitis_hls -f run_hls.tcl

open_project xform_core_prj -reset
set_top xform_core
add_files xform_core.cpp
add_files -tb tb.cpp
open_solution solution1 -reset
set_part xc7z010clg400-1
create_clock -period 10 -name default
csim_design
csynth_design
export_design -format ip_catalog -output ./xform_core_ip
exit