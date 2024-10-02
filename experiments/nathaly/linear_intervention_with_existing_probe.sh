#!/bin/bash
#SBATCH --job-name=/linear_intervention_all_all_-1_benign
#SBATCH --partition=single
#SBATCH --time=30:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/linear_intervention_all_all_-1_benign.out

conda activate base


    # # Pass arguments from the command line to main()
    # dataset_path = sys.argv[1]  
    # probe_path = sys.argv[2]
    # save_path = sys.argv[3]
    # intervention_strength = int(sys.argv[4])
    # perturb_prompt_or_all = sys.argv[5]


python linear_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_benign.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/linear_probe.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_linear_intervention_all_all_-1_benign_requests.pkl" -1 "all"
