# csim_run.tcl - rebuild project and run only C simulation (fast bit-exact check)
# Usage: vitis_hls -f csim_run.tcl
open_project xform_core_prj -reset
set_top xform_core
add_files xform_core.cpp
add_files -tb tb.cpp
open_solution solution1 -reset
set_part xc7z010clg400-1
create_clock -period 10 -name default
csim_design
exit