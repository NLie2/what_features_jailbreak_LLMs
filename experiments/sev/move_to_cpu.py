
import sys 
import os
sys.path.append("/data/nathalie_maria_kirch/ERA_Fellowship")


import numpy as np
import pandas as pd
import torch

# NEW activations from nathalie
# df = pd.read_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/allenaiwildjailbreak_googlegemma-2b-it_last_token_of_prompt_activations_2407202414:33:56.pkl")

# path = "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/allenaiwildjailbreak_googlegemma-2b-it_2107202412:21:34.pkl"
# df1 = pd.read_pickle(path)

# path2 = "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/allenaiwildjailbreak_googlegemma-2b-it_2107202414:21:25.pkl"
# df2 = pd.read_pickle(path2)

# df = pd.concat([df1, df2])

# path = "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/allenaiwildjailbreak_googlegemma-2b-it_response_2407202414:03:08.pkl"
path = "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/allenaiwildjailbreak_googlegemma-2b-it_last_token_of_prompt_activations_2407202414:33:56.pkl"
df =  pd.read_pickle(path)
def move_tensor_to_cpu(item):
    if isinstance(item, torch.Tensor):
        return item.cpu()
    elif isinstance(item, dict):
        return {k: move_tensor_to_cpu(v) for k, v in item.items()}
    elif isinstance(item, list):
        return [move_tensor_to_cpu(v) for v in item]
    else:
        return item

def move_dataframe_to_cpu(df):
    for column in df.columns:
        df[column] = df[column].apply(move_tensor_to_cpu)
    return df

df = move_dataframe_to_cpu(df)

# print(df.head(1)['res_activations'])
# print(df.head(1)['last_token_of_prompt_activations'])

df.to_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/800_examples_nathalie_fixed.pkl")
# df.to_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/activations_new.pkl")