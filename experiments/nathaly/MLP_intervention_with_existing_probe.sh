#!/bin/bash
#SBATCH --job-name=/old_probe_logit_1_60_01
#SBATCH --partition=single
#SBATCH --time=30:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=nathalie.kirch.nk@gmail.com
#SBATCH --mail-type=END
#SBATCH --output=logs/old_probe_logit_1_60_01.out

conda activate base


#    dataset_path = sys.argv[1]  
#    probe_path = sys.argv[2]
#    save_path = sys.argv[3]
#    target = sys.argv[4]
#    perturb_prompt_or_all = sys.argv[5]
#    nr perturbations = sys.argv[6]


# python MLP_intervention_with_existing_probe.py /data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_only_requests.pkl /data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth /data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_validation_target_0_perturb_prompt.pkl 0 prompt

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_benign.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_0_MSE_benign_requests.pkl" 0 "all" "MSE"

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_0_logitloss.pkl" 0 "all" "logit"

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss.pkl" 1 "all" "logit"

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_benign.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_benign.pkl" 1 "all" "logit"

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_150_01.pkl" 1 "all" "logit" 150 0.1

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_300_01.pkl" 1 "all" "logit" 300 0.1


# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_500_01.pkl" 1 "all" "logit" 500 0.1


python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_60_01.pkl" 1 "all" "logit" 60 0.1


# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_bce_500_015.pkl" 1 "all" "BCE" 500 0.15

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_all_1_bce_500_015.pkl" 1 "all" "BCE" 500 0.1

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_old_probe_all_1_bce_1000_01.pkl" 1 "all" "BCE" 1000 0.1

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_old_probe_all_1_bce_new_30_01.pkl" 1 "all" "BCE_new" 30 0.1

# python MLP_intervention_with_existing_probe.py "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_MLP_intervention_old_probe_all_0_bce_1000_01.pkl" 0 "all" "BCE" 1000 -0.01
