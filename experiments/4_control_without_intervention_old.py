# NOT USING CHAT TEMPLATES 

from experiments.paths import PROJECT_ROOT

import pandas as pd
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os 


from pathlib import Path

from analysis.supervised import *
from analysis.causal_interventions import *
from analysis.Probes import SimpleMLP
from gen_data.harmBench_autograder import *
from analysis.models import *


def main(dataset_path, model_name):
    if ".csv" in dataset_path:
      df = pd.read_csv(str(dataset_path))
    elif ".pkl" in dataset_path:
      df = pd.read_pickle(str(dataset_path))
      

    model_name, num_layers, rating_column, hf_ident, input_size  =  get_model_details(model_name)
    model = AutoModelForCausalLM.from_pretrained(hf_ident)
    tokenizer = AutoTokenizer.from_pretrained(hf_ident)
    
    print("Model name: ", model_name)


    print("Dataset loaded")
    
    harmful_harmless = "harmful"
    if "benign" in str(dataset_path): 
      harmful_harmless = "harmless"
    
    filename = PROJECT_ROOT / f"/datasets/{model_name}/final/{harmful_harmless}/no_intervention_control.pkl"
    save_path = filename
    # save_path = str(project_root / filename)

    # rated_save_path  = str(project_root / filename.replace(".pkl", "_rated.pkl"))
    rated_save_path = filename.replace(".pkl", "_rated.pkl")
    
    print("save path: ", str(save_path))
    
    
    # tokenizer = AutoTokenizer.from_pretrained(hf_ident)
    # model = AutoModel.from_pretrained(hf_ident, torch_dtype=torch.float16, device_map="auto")

    print("model loaded")
    
    model.to("cuda")   
    
    df["control_no_intervention_response"] = None  

    # Run on every element of the sample
    for index, row in df.iterrows():
      
      prompt = row["jailbreak_prompt_text"]
      print("prompt: ", prompt)
      
      input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

      outputs = model.generate(**input_ids, max_new_tokens=200, do_sample=False)

      response = tokenizer.decode(outputs[0])
      
      
      if response.startswith(prompt): response = response[len(prompt):].strip()
      if response.startswith("<|begin_of_text|>"+prompt): response = response[len("<|begin_of_text|>"+prompt):].strip()
      if response.startswith("<bos>"+prompt): response = response[len("<bos>"+prompt):].strip()

      df.at[index, "control_no_intervention_response"] = response
      print(response)

      # Save every 20 samples
      if index % 20 == 0:
        df.to_pickle(save_path)
        
    
    df.to_pickle(save_path)
    print(f"Data saved to {save_path}")
    
    
    rate_dataset(df, request_column = "jailbreak_prompt_text", response_column="control_no_intervention_response", output_path = rated_save_path)
    
    print("rated data saved to", rated_save_path)
        
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_python_script.py <dataset_path> <model_name>")
        sys.exit(1)

    # Pass arguments from the command line to main()
    dataset_path = sys.argv[1]  
    model_name = sys.argv[2]

    main(dataset_path, model_name) 
