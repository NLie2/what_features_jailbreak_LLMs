#!/bin/bash
#SBATCH --job-name=rate_wei
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/rate_wei_data

conda activate base

python rate_dataset.py /data/nathalie_maria_kirch/ERA_Fellowship/datasets/wei_dataset_meta-llamaMeta-Llama-3-8B-Instruct_prompt_res_activations_0409202422:04:21.pkl /data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/wildJailBreak_Llama-3-8B-Instruct_rated.pkl 
