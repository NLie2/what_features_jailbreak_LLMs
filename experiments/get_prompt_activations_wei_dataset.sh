#!/bin/bash
#SBATCH --job-name=wei_activations
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=sevdeawesome@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/get_prompt_activations_three_new_jailbreaks.log

conda activate base

python get_prompt_activations_wei_dataset.py