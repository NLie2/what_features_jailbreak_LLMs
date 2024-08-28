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