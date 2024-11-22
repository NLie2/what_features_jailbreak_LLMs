#!/bin/bash
#SBATCH --job-name=wildjailbreak
#SBATCH --partition=single
#SBATCH --time=5:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/get_response_activations_wildjailbreak_8000

conda activate base

python get_response_activations_wildJailbreak_len8000.py