#!/bin/bash
#SBATCH --job-name=wei_activations
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/latest.log

conda activate base

python get_prompt_activations_wei_dataset.py --model_name "meta-llama/Meta-Llama-3-8B"