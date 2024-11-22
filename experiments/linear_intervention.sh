#!/bin/bash
#SBATCH --job-name=linear_intervention
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/linear_intervention

conda activate base

python linear_intervention.py