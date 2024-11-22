#!/bin/bash
#SBATCH --job-name=wildjailbreak
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/collect_wildjailbreak_8000

conda activate base

python get_activations_wildJailbreak_len8000.py