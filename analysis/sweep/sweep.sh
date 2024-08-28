#!/bin/bash
#SBATCH --job-name=sweep_litmus
#SBATCH --partition=single
#SBATCH --time=3:00:00
#SBATCH --gpus-per-node=0
#SBATCH --mail-user=sevdeawesome@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/sweep_basic

source activate base

python sweep.py