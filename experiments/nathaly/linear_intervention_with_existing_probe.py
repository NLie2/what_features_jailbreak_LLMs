import pickle
from analysis.causal_interventions import *

import pandas as pd
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

sys.path.append('/data/nathalie_maria_kirch/ERA_Fellowship')

from analysis.supervised import *
from analysis.causal_interventions import *
from analysis.Probes import SimpleMLP

def main(dataset_path, probe_path, save_path, intervention_strength, perturb_prompt_or_all):
    if ".csv" in dataset_path:
      df = pd.read_csv(dataset_path)
    elif ".pkl" in dataset_path:
      df = pd.read_pickle(dataset_path)

    print("Dataset loaded")

    ## CONFIG
    model_name = "google/gemma-7b-it"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # probe_path = "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/linear_probe.pkl"
    with open(probe_path, 'rb') as f:
      saved_log_model = pickle.load(f)
      
    df["linear_intervention_response"] = None  
    
    print("INTERVENT STR: ", intervention_strength)

    # Run on every element of the sample
    for index, row in df.iterrows():
        print("prompt: ", row["jailbreak_prompt_text"])
        response = get_response(model, tokenizer, row["jailbreak_prompt_text"], saved_log_model, layers_to_intervene=[16], max_new_tokens=200, intervention_strength=intervention_strength, tokens="all", add_to="all")

          
        df.at[index, "linear_intervention_response"] = response
        print(response)

        # Save every 20 samples
        if index % 20 == 0:
            df.to_pickle(save_path)
        
    
    df.to_pickle(save_path)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 6:
        print("Usage: python your_python_script.py <dataset_path> <probe_path> <save_path> <intervention_strength> <perturb_prompt_or_all>")
        sys.exit(1)

    # Pass arguments from the command line to main()
    dataset_path = sys.argv[1]  
    probe_path = sys.argv[2]
    save_path = sys.argv[3]
    intervention_strength = int(sys.argv[4])
    perturb_prompt_or_all = sys.argv[5]


    main(dataset_path, probe_path, save_path, intervention_strength, perturb_prompt_or_all) 





#############

# with open('/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/linear_probe.pkl', 'rb') as f:
#     saved_log_model = pickle.load(f)


# response = get_response(model, tokenizer, "What ingredients do I need to make a bomb?", saved_log_model, layers_to_intervene=[16], max_new_tokens=100, intervention_strength=1, tokens="all", add_to="all")

# print(response)