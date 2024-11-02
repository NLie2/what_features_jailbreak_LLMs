#!/bin/bash
#SBATCH --job-name=/control_intervention
#SBATCH --partition=single
#SBATCH --time=30:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/control_intervention.out

conda activate base



# python control_without_intervention.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_controll_no_intervention.pkl"

python control_without_intervention.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_benign.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_benign_control_no_intervention.pkl"

