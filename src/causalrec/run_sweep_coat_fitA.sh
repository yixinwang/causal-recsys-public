#!/bin/bash
#

python -u wg_fitA_py2.py -ddir "../../dat/proc/coat_wg" -odir "../../out/coat_wg_Afit"

python -u sg_fitA_py2.py -ddir "../../dat/proc/coat_sg" -odir "../../out/coat_sg_Afit"
