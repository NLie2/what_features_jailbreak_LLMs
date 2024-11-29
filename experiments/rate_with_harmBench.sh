#!/bin/bash
#SBATCH --job-name=rate_with_harmBench
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/rate.log

conda activate base

python rate_dataset.py --input_path /data/nathalie_maria_kirch/ERA_Fellowship/datasets/_meta-llamaLlama-3.1-8B-Instruct_prompt_res_activations_2911202410:07:48.pkl --column_to_grade "output_text"