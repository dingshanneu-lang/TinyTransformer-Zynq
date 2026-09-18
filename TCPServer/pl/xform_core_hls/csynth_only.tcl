# csynth_only.tcl - rebuild project and run ONLY synthesis (debug: no csim)
open_project xform_core_prj -reset
set_top xform_core
add_files xform_core.cpp
open_solution solution1 -reset
set_part xc7z010clg400-1
create_clock -period 10 -name default
csynth_design
exit