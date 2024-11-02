import sys 
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from gen_data.get_wildJailbreak import get_activations_from_df



from gen_data.get_wildJailbreak import get_activations_from_df
import pandas as pd



df = pd.read_pickle("/datasets/allenaiwildjailbreak_googlegemma-2b-it_len_8000_rated_0608202415:19:55.pkl")

model_name = "google/gemma-2b-it"
num_samples = len(df)
dataset_name = "allenai/wildjailbreak"

save_path = get_activations_from_df(model_name = model_name, prompt_column = "forbidden_prompt", df = df, n_rows = num_samples)


print(save_path)