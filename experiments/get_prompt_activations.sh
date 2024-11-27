#!/bin/bash
#SBATCH --job-name=get_prompt_activations
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/latest.log

conda activate base

python get_prompt_activations.py --model_name "meta-llama/Llama-3.1-8B-Instruct"