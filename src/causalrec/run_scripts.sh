#!/bin/bash

#SBATCH --account=sml
#SBATCH -c 4
#SBATCH --time=11:00:00
#SBATCH --mem-per-cpu=32gb

source /proj/sml_netapp/opt/anaconda2-4.2.0/etc/profile.d/conda.sh
conda activate py2 

python -u ${MODELCODEPY} -ddir ${DATADIR} -cdir ${LOCALFITDIR} -odir ${OUTDIR} -odim ${OUTDIM} -cdim ${CAUDIM} -th ${THOLD} -M ${BATCHSIZE} -nitr ${NITER} -pU ${PRIORU} -pV ${PRIORV} -alpha ${ALPHA} -binary ${BINARY}