import sys 
import os
import torch
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer 

from analysis.supervised import *
from analysis.causal_interventions import *

model_name = "google/gemma-7b-it"
layer=17


def extract_layer(row, layer):
    return row['prompt_res_activations'][layer][0].cpu().numpy().flatten()


df = pd.read_pickle("/datasets/latest/wei_dataset_rated.pkl")

# Add column that has 0 if "rating" is "No" and 1 if "rating" is "Yes"
df['rating_binary'] = df['rating'].apply(lambda x: 0 if x == "No" else 1)

# Create new columns for each layer
df[f'layer_{layer}'] = df.apply(lambda row: extract_layer(row, layer), axis=1)


## Delete bad data entries

# Get all rows that contain the word "error" in the jailbreak prompt text column 
df_new = df[df["jailbreak_prompt_text"].str.contains("ERROR", case=False)]
df_new["jailbreak_prompt_name"].value_counts()

# drop all rows that have auto_payload_splitting, auto_obfuscation, dev_mode_with_rant in jailbreak_prompt_name
df = df[~df["jailbreak_prompt_name"].isin(["auto_payload_splitting", "auto_obfuscation", "dev_mode_with_rant"])]
df["jailbreak_prompt_name"].value_counts()

## Shuffle and balance dataset
df = balance_dataset(df, "rating_binary")
df["rating_binary"].value_counts()

df = df.sample(frac=1).reset_index(drop=True)


## Linear probe
split_idx = int(0.8* len(df))
labels = df["rating_binary"].to_list()
data = df[f'layer_{layer}'].to_list()


probe, accuracy = create_logistic_regression_model(data, labels, split_idx)
print("PROBE ACURACY", accuracy)


## Linear intervention
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


exp_config = [{"layers_to_intervene": [10], "strength":3}, 
              {"layers_to_intervene": [17], "strength":3}, 
              {"layers_to_intervene": [10, 12, 15, 17], "strength": 1}
              ]

for exp in exp_config: 
  print("EXPERIMENT", exp)
  df["response_after_causal_intervention"] = None

  for idx, row in df.iterrows():
    
    print(row["jailbreak_prompt_text"])
    response = get_response(model, tokenizer, row["jailbreak_prompt_text"], probe = probe, max_new_tokens=100, layers_to_intervene=exp["layers_to_intervene"], intervention_strength=exp["strength"])

    print("response")
    print(response)

  df.at[idx, "response_after_causal_intervention"] 
  
  path = f"/datasets/latest/wei_dataset_with_linear_intervention.pkl"
  pd.to_pickle(path)
  print(f"saved to {path}")
  print()

  







