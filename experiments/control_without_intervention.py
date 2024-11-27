import pandas as pd
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


from pathlib import Path

# Get the current working directory (where the notebook is)
notebook_dir = Path.cwd()

# Move up to the root directory (parent of 'notebooks')
project_root = notebook_dir.parent


sys.path.append(notebook_dir.parent)

from analysis.supervised import *
from analysis.causal_interventions import *
from analysis.Probes import SimpleMLP

def main(dataset_path, save_path):
    if ".csv" in dataset_path:
      df = pd.read_csv(dataset_path)
    elif ".pkl" in dataset_path:
      df = pd.read_pickle(dataset_path)

    print("Dataset loaded")

    ## CONFIG
    model_name = "google/gemma-7b-it"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    
    df["control_no_intervention_response"] = None  

    # Run on every element of the sample
    for index, row in df.iterrows():
      print("prompt: ", row["jailbreak_prompt_text"])
      
      input_ids = tokenizer(row["jailbreak_prompt_text"], return_tensors="pt")

      outputs = model.generate(**input_ids, max_new_tokens=200)

      response = tokenizer.decode(outputs[0])

      df.at[index, "control_no_intervention_response"] = response
      print(response)

      # Save every 20 samples
      if index % 20 == 0:
        df.to_pickle(save_path)
        
    
    df.to_pickle(save_path)
    print(f"Data saved to {save_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_python_script.py <dataset_path> <save_path>")
        sys.exit(1)

    # Pass arguments from the command line to main()
    dataset_path = sys.argv[1]  
    save_path = sys.argv[2]

    main(dataset_path, save_path) 
