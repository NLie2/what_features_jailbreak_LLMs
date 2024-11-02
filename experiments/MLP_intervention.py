import pandas as pd
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

sys.path.append('/data/nathalie_maria_kirch/ERA_Fellowship')

from analysis.supervised import *
from analysis.causal_interventions import *
from analysis.Probes import SimpleMLP


df = pd.read_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/wei_dataset_rated.pkl")

print("dataset loaded")

## CONFIG
model_name = "google/gemma-7b-it"
layer=17
activation_column="prompt_res_activations"
num_samples = len(df)
dataset_name = "wei"

model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


# Add column that has 0 if "rating" is "No" and 1 if "rating" is "Yes"
df['rating_binary'] = df['rating'].apply(lambda x: 0 if x == "No" else 1)
df.head(1)

def extract_layer(row, layer):
    return row['prompt_res_activations'][layer][0].cpu().numpy().flatten()

# Create new columns for each layer
df[f'layer_{layer}'] = df.apply(lambda row: extract_layer(row, layer), axis=1)

# drop all rows that have auto_payload_splitting, auto_obfuscation, dev_mode_with_rant in jailbreak_prompt_name
df = df[~df["jailbreak_prompt_name"].isin(["auto_payload_splitting", "auto_obfuscation", "dev_mode_with_rant"])]
df["jailbreak_prompt_name"].value_counts()

labels = df["rating_binary"].to_list()
data = df[f'layer_{layer}'].to_list()
print("LEN LABELS", len(labels))
print("LEN DATA", len(data))

split_idx = int(0.8* len(df))


#####

mlp = SimpleMLP()

mlp.train_MLP(data = data, labels = labels, split_idx = split_idx, num_epochs = 500)
mlp.get_accuracy(data = data, labels = labels, split_idx = split_idx)

mlp.train()
# run on every element of the sample
for index, row in df.iterrows():
    # save in sample_df
    response = get_response_MLP(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=200, layers_to_intervene=[16], intervention_strength=1)
    df.at[index, "response"] = response
    print(response)
    
    # Save every 10 samples
    if index % 20 == 0:
        df.to_pickle(f"/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/wei_dataset_with_MLP_intervention_full.pkl")
    