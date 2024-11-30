#!/bin/bash
#SBATCH --job-name=rate_with_harmBench
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/rate.log

conda activate base

python rate_dataset.py --input_path datasets/_meta-llamaMeta-Llama-3-8B_prompt_res_activations_2311202405:12:17.pkl --column_to_grade "output_text"