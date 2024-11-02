#!/bin/bash
#SBATCH --job-name=/linear_ben_-025
#SBATCH --partition=single
#SBATCH --time=30:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/linear_ben_-025.out

conda activate base


    # # Pass arguments from the command line to main()
    # dataset_path = sys.argv[1]  
    # probe_path = sys.argv[2]
    # save_path = sys.argv[3]
    # intervention_strength = int(sys.argv[4])
    # perturb_prompt_or_all = sys.argv[5]


python linear_intervention_with_existing_probe.py "/datasets/latest/validation_data_benign.pkl" "/datasets/Probes/linear_probe.pkl" "/datasets/latest/validation_data_benign_with_linear_intervention_strength_-025.pkl" -0.25 "all"
