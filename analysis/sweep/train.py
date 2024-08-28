import torch as t
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
import numpy as np
import wandb
import shutil
import gc
import pandas as pd

import sys
sys.path.append('/data/nathalie_maria_kirch/ERA_Fellowship')
from analysis.unsupervised import *
from analysis.supervised import *
from analysis.utils import * 

def set_seed(seed):
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)

set_seed(42)


def train():
    print("TRAINING")
    try:
        wandb.init(project="project-alpha")
        GPU = "cuda" if t.cuda.is_available() else "cpu"
        config = wandb.config

        # log the config
        print("CONFIG")
        print(config)
        print(type(config))
        print(config.dataset)
        print(config.lr)
        print(config.model)


        #! STEP 1: Load the data into a clear format
        if config.dataset == "wei":
            print("Loading Wei's dataset")
            df = pd.read_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/wei_dataset_googlegemma-7b_with_ratings2.pkl")
            df = fix_wei_dataset(df)
        elif config.dataset == "wildjailbreak":
            print("BREAKPOINT 1")
        else:
            print("BREAKPOINT 2")

        # load data into train and test
        data = df[f'layer_{config.layer}'].to_list()
        split_idx = int(0.8* len(df))
        labels = df["rating_binary"].to_list()

        if config.model == "linear_probe":
            probe, accuracy = create_logistic_regression_model(data, labels, split_idx)
            print("Accuracy: ", accuracy)
            wandb.log({"highest_val_accuracy": accuracy})
        elif "mlp" in config.model:
            print("Creating MLP model with hidden units: ", config.model)
            # the config.model is gonna look like "mlp_hidden_8" or "mlp_hidden_16", extract the number of hidden units
            hidden_units = int(config.model.split("_")[-1]) 
            print("Hidden units: ", hidden_units)
             
            probe, accuracy = create_mlp_model(data, labels, split_idx, (hidden_units,))
            print("Accuracy: ", accuracy)
            wandb.log({"highest_val_accuracy": accuracy})
        else:
            print("BREAKPOINT 3")
            

    except t.cuda.OutOfMemoryError:
        print("OUT OF MEMORY!!!")
        del train_data
        del test_data
        del meta_model
        gc.collect()
        t.cuda.empty_cache()

if __name__ == "__main__":
    class Config:
        def __init__(self, config):
            for k, v in config.items():
                setattr(self, k, v)

    def eat(*args, **kwargs):
        print("wandb call", args, kwargs)

    config = {
            "layer": 17,
            "lr": random.choice([0.01, 0.1]),
            # "model": random.choice(["linear_probe", "ml"]),
            # "model": "linear_probe",
            "model": "mlp_hidden_10",
            # "dataset": random.choice(["wei", "wildjailbreak", "both"])
            "dataset": "wei"
        }
    wandb.config = config
    # wandb.init = wandb.log = eat
    
    with wandb.init(project="project-alpha", config=config):
        train()