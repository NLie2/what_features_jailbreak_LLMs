#!/bin/bash
#SBATCH --job-name=rate_logit
#SBATCH --partition=single
#SBATCH --time=20:00:00
#SBATCH --gpus-per-node=1
#SBATCH --mail-type=END
#SBATCH --output=logs/rate_logit.out

conda activate base

# python rate_dataset.py /datasets/wei_dataset_meta-llamaMeta-Llama-3-8B-Instruct_prompt_res_activations_0409202422:04:21.pkl /datasets/latest/wildJailBreak_Llama-3-8B-Instruct_rated.pkl 


# python rate_dataset.py /datasets/latest/three_new_fixed.pkl /datasets/latest/three_new_rated.pkl


# python rate_dataset.py //datasets/latest/validation_data_with_MLP_intervention_only_requests.pkl /datasets/latest/validation_data_with_MLP_intervention_only_requests_rated.pkl


# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_validation_target_0_perturb_prompt_rated.pkl /datasets/latest/validation_data_with_MLP_intervention_validation_target_0_perturb_prompt_rated.pkl

# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_validation_target_1_perturb_prompt_rated.pkl //datasets/latest/validation_data_with_MLP_intervention_validation_target_1_perturb_prompt_rated.pkl


# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_validation_target_1_rated.pkl /datasets/latest/validation_data_with_MLP_intervention_validation_target_1_rated.pkl


# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_all_1.pkl /datasets/latest/validation_data_with_MLP_intervention_all_1_rated.pkl


# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_all_0_BCE.pkl /datasets/latest/validation_data_with_MLP_intervention_all_0_BCE_rated.pkl

# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_all_1_BCE.pkl /datasets/latest/validation_data_with_MLP_intervention_all_1_BCE_rated.pkl

# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_random.pkl /datasets/latest/validation_data_with_MLP_intervention_random_rated.pkl



# python rate_dataset.py /datasets/latest/validation_data_with_linear_intervention_all_all_1.pkl /datasets/latest/validation_data_with_linear_intervention_all_all_1_rated.pkl

# python rate_dataset.py data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_linear_intervention_all_all_-1.pkl data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_with_linear_intervention_all_all_-1_rated.pkl


# python rate_dataset.py /datasets/latest/validation_data_with_linear_intervention_all_all_-1_benign_requests.pkl /datasets/latest/validation_data_with_linear_intervention_all_all_-1_benign_requests_rated.pkl

# python rate_dataset.py /datasets/latest/validation_data_with_linear_intervention_all_all_1_benign_requests.pkl /datasets/latest/validation_data_with_linear_intervention_all_all_1_benign_requests_rated.pkl

# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss.pkl /datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_rated.pkl



# python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_300_025.pkl /datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_300_025_rated.pkl

python rate_dataset.py /datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_150_025.pkl /datasets/latest/validation_data_with_MLP_intervention_all_1_logitloss_150_025_rated.pkl