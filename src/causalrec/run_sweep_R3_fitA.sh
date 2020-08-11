#!/bin/bash
#


python -u wg_fitA_py2.py -ddir "../../dat/proc/R3_wg" -odir "../../out/R3_wg_Afit"

python -u sg_fitA_py2.py -ddir "../../dat/proc/R3_sg" -odir "../../out/R3_sg_Afit"
