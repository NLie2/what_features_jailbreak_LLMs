import sys 
import os
import torch
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer 
from analysis.unsupervised import *
from analysis.supervised import *

def check_all_parameters(model):
  return all(p.device.type == 'cuda' for p in model.parameters())


def check_hflm_model_devices(hflm_model):
    print("HF Model device check:", check_all_parameters(hflm_model.model))
    for name, param in hflm_model.model.named_parameters():
        # print(f"{name} on {param.device}")
        if param.device.type != 'cuda':
            # print(f"Parameter {name} is not on GPU")
            return False
    return True
  
def fix_wei_dataset(df, layer=17, activation_column="prompt_res_activations", shuffle=True):
    def extract_layer(row, layer):
        return row[activation_column][layer][0].cpu().numpy().flatten()

# Create new columns for each layer
    df[f'layer_{layer}'] = df.apply(lambda row: extract_layer(row, layer), axis=1)

    df['rating_binary'] = df['rating'].apply(lambda x: 0 if x == "No" else 1)
    df = df[~df["jailbreak_prompt_text"].str.contains("ERROR")]
    df =  balance_dataset(df, "rating_binary")
    # shuffle
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    
    return df

def fix_wjb_dataset(df, layer=17, activation_column="prompt_res_activations", shuffle=True):
    def extract_layer(row, layer):
        return row[activation_column][layer][0].cpu().numpy().flatten()
    
    # Create new columns for each layer
    df[f'layer_{layer}'] = df.apply(lambda row: extract_layer(row, layer), axis=1)

    df['rating_binary'] = df['rating'].apply(lambda x: 0 if x == "No" else 1)
    df = balance_dataset(df, "rating_binary")

    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)

    return df