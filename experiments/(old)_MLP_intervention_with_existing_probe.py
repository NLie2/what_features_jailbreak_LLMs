import pandas as pd
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


from analysis.supervised import *
from analysis.causal_interventions import *
from analysis.Probes import SimpleMLP

def main(dataset_path, probe_path, save_path, target, perturb_prompt_or_all, loss, nr_perturbations, learning_rate):
    if ".csv" in dataset_path:
      df = pd.read_csv(dataset_path)
    elif ".pkl" in dataset_path:
      df = pd.read_pickle(dataset_path)

    print("Dataset loaded")

    ## CONFIG
    model_name = "google/gemma-7b-it"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    

    mlp = SimpleMLP()

    # Load the model from the specified probe path
    mlp.load_model(probe_path)
    
    df["MLP_intervention_response"] = None  

    # Run on every element of the sample
    for index, row in df.iterrows():
        print("prompt: ", row["jailbreak_prompt_text"])
        if perturb_prompt_or_all == "prompt":
          response = get_response_MLP_only_prompt(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=200, layers_to_intervene=[16], target=target)
        elif perturb_prompt_or_all == "all": 
          if loss == "logit":
            response = get_response_MLP_logit(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=200, layers_to_intervene=[16], target=target, nr_perturbations=nr_perturbations)
          else: 
            response = get_response_MLP(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=200, layers_to_intervene=[16], target=target, loss = loss, nr_perturbations = nr_perturbations, learning_rate=learning_rate)
        elif perturb_prompt_or_all == "random": 
          response = get_response_MLP_random(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=200, layers_to_intervene=[16])

        else: 
          print("Please specify if you want to perturb the prompt or all text")
          sys.exit(1)
          
        df.at[index, "MLP_intervention_response"] = response
        print(response)

        # Save every 20 samples
        if index % 20 == 0:
            df.to_pickle(save_path)
        
    
    df.to_pickle(save_path)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 9:
        print("Usage: python your_python_script.py <dataset_path> <probe_path> <save_path> <target> <perturb_prompt_or_all> <loss> <nr_perturbations> <learning_rate>")
        sys.exit(1)

    # Pass arguments from the command line to main()
    dataset_path = sys.argv[1]  
    probe_path = sys.argv[2]
    save_path = sys.argv[3]
    target = int(sys.argv[4])
    perturb_prompt_or_all = sys.argv[5]
    loss = sys.argv[6]
    nr_perturbations = int(sys.argv[7])
    learning_rate = float(sys.argv[8])


    main(dataset_path, probe_path, save_path, target, perturb_prompt_or_all, loss, nr_perturbations, learning_rate) 
