#!/bin/bash
#SBATCH --job-name=/offensivemlpc10
#SBATCH --partition=single
#SBATCH --time=30:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/offensivemlpc10.out

conda activate base


python defensive_mlp_perturbation.py
