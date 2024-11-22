#!/bin/bash
#SBATCH --job-name=/bce_gs
#SBATCH --partition=single
#SBATCH --time=30:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/bce_gs.out

conda activate base


python logit_loss_gs.py

