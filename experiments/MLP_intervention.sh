#!/bin/bash
#SBATCH --job-name=MLP_intervention
#SBATCH --partition=single
#SBATCH --time=30:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/MLP_interventions

conda activate base

python MLP_intervention.py