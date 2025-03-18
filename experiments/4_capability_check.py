from analysis.Probes import *
import json
import os
import argparse


from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
import torch
import sys
import pickle

from pathlib import Path
import os
import pandas as pd
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

from gen_data.harmBench_autograder import grade_with_HF
from analysis.causal_interventions import *
from analysis.models import *
from analysis.eval import *

# Import paths module
from experiments.paths import PROJECT_ROOT

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
      
def main(model_name, probe_path=None, c=None, offensive=None, learning_rate= None): 
  # Using PROJECT_ROOT instead of computing it from cwd
                     
  device = torch.device("cuda")

  model_name, num_layers, rating_column, hf_ident, input_size  =  get_model_details(model_name)

  model = AutoModelForCausalLM.from_pretrained(hf_ident)
  tokenizer = AutoTokenizer.from_pretrained(hf_ident)
  
  middle_layer = num_layers // 2

  model.to(device)

  hflm_model = HFLM(model)
  hflm_model.model.to(device)

  assert  check_all_parameters(hflm_model.model)
  
  if probe_path: 
    print("OFFENSIVE", offensive, type(offensive))                        
  
    offens_defens = "offensive" if offensive else "defensive"
  
    probe_path = PROJECT_ROOT / probe_path  
    if "linear" in str(probe_path):
      probe_type = "linear"

      with open(probe_path, 'rb') as f:
            probe = pickle.load(f)
      
      # c = intervention strength, negative c= defenisve intervention
      offensive = True if c > 0 else False
      
      insert_linear_hook(model=model, tokenizer=tokenizer, probe=probe, layers_to_intervene=[middle_layer], intervention_strength=c)
          
    elif "mlp" in str(probe_path):
      probe_type = "mlp"
      probe = SimpleMLP(input_size = input_size )
      probe.load_probe(probe_path)
      probe.to(device)
      
      assert check_all_parameters(probe)
      
      insert_mlp_hook(model=model, mlp = probe, layers_to_intervene=[middle_layer], c = c, learning_rate = learning_rate, offensive= offensive)
    
    elif "transformer" in str(probe_path):
      probe_type = "transformer" 
      probe = TransformerProbe(input_size = input_size )

      probe.load_probe(probe_path)
      probe.to(device)
      
      assert check_all_parameters(probe)
      
      
      insert_transformer_hook(model=model, probe = probe, layers_to_intervene=[middle_layer], c = c, learning_rate = learning_rate, offensive= offensive)
      assert check_hflm_model_devices(hflm_model)

  
  eval_output = evaluator.simple_evaluate(model=hflm_model, tasks=['mmlu'], device='cuda', verbosity='ERROR')

  print(eval_output['results']['mmlu'])
  
  save_experiment_params(project_root = PROJECT_ROOT, model_name= model_name, capability = eval_output['results'], probe_path=probe_path, c= c, offensive=offensive, learning_rate=learning_rate)



if __name__ == "__main__":
    # Set up argument parsing
    print("Raw arguments:", sys.argv)  # Add this line to see what arguments are being passed
    
    parser = argparse.ArgumentParser(description="Run MMLU capability check on specified model.")
    parser.add_argument("--model_name", type=str, required=True, help="model name")
    parser.add_argument("--probe_path", type=str, help="probe path (linear, mlp, or transformer)")
    parser.add_argument("--c", type=float, help="c value for intervention")
    parser.add_argument("--offensive", action="store_true", help="if set, run in offensive mode")
    parser.add_argument("--learning_rate", type=float, help="learning rate for intervention")
    args = parser.parse_args()
    
    
    # Run the main function with the provided arguments
    main(model_name = args.model_name, probe_path = args.probe_path, c = args.c, offensive = args.offensive, learning_rate = args.learning_rate)
