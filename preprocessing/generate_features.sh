#!/bin/bash
#SBATCH --partition=dcs-gpu
#SBATCH --account=dcs-res
#SBATCH --gpus-per-node=1
#SBATCH --mem=64G
#SBATCH --output=slurm.out
#SBATCH -t 0-90:00:00

module load Anaconda3/5.3.0
# module load cuDNN/7.6.4.38-gcccuda-
module load SoX/14.4.2-GCC-8.3.0

source activate picknnmix

stems=(bass drums other vocals mixture)

stem=${stems[$SLURM_ARRAY_TASK_ID]}

python dump_mfcc.py --tsv_dir ../manifests/wav/train/ \
    --stem $stem \
    --feat_dir ../manifests/feats/default \
    --nshard 1 \
    --rank 0

python dump_mfcc.py --tsv_dir ../manifests/wav/test/ \
    --stem $stem \
    --feat_dir ../manifests/feats/default \
    --nshard 1 \
    --rank 0
