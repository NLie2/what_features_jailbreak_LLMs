import sys 
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 

sys.path.append("/data/nathalie_maria_kirch/ERA_Fellowship")


model_name = "google/gemma-7b-it"
num_samples = 8000
dataset_name = "allenai/wildjailbreak"


from gen_data.get_wildJailbreak import load_dataset_with_activations, get_activations_from_df
import pandas as pd

path = load_dataset_with_activations(model_name = model_name, n_rows = num_samples, generate_output = True, token_position = "last", dataset_name = "allenai/wildjailbreak", only_harmful = False , rate = False)

print(path)



