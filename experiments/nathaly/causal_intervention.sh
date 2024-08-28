#!/bin/bash
#SBATCH --job-name=causal_intervention
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/causal_intervention_NAT

conda activate base

python causal_intervention.py