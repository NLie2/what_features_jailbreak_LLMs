import pandas as pd
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
import argparse
import ast
import sys
import os
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer

from analysis.causal_interventions import *
from analysis.utils import*


from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM


import pandas as pd
import numpy as np
from itertools import product



import pandas as pd
import numpy as np
from itertools import product
import argparse


import pandas as pd
import numpy as np
import os
from itertools import product


############

from pathlib import Path
import os

# Import paths module
from experiments.paths import PROJECT_ROOT, set_project_root_as_cwd

# Set project root as current working directory
set_project_root_as_cwd()

###############

from analysis.unsupervised import *
from analysis.supervised import *
from gen_data.get_activations import *
from analysis.models import *
from gen_data.harmBench_autograder import *


######## FUNCTIONS
PROBE_CONFIGS = {
      "linear": {
          "loader": lambda path: pickle.load(open(path, 'rb')),
          "response_fn": get_response_linear,
          "probe_class": None  # Linear probe doesn't need initialization
      },
      "mlp": {
          "loader": lambda path, input_size: SimpleMLP(input_size=input_size).load_probe(path),
          "response_fn": get_response_MLP,
          "probe_class": SimpleMLP
      },
      "transformer": {
          "loader": lambda path, input_size: TransformerProbe(input_size=input_size).load_probe(path),
          "response_fn": get_response_transformer,
          "probe_class": TransformerProbe
      }
  }


def get_baseline(model, model_name, tokenizer, harmful, autograder, autogr_tokenizer, results_csv_path, single = ""):
    print("get baseline...")
    
    # Setup model
    hflm_model = HFLM(model)
    hflm_model.model.eval()
    
    n_layers = None
    
    base_path = PROJECT_ROOT / f"datasets/{model_name}/final/harmful_{single}/"
    
    # Find data paths
    try:
        # Try to load the preprocessed data if it exists
        preprocessed_df = pd.read_pickle(base_path / f"no_intervention_control_responses.pkl")
    except:
        # If it doesn't exist, perform the processing
        print("Creating preprocessed file...")
        # Get harmful dataset (this is the one we evaluate on)
        data = pd.concat([
            pd.read_pickle(
                PROJECT_ROOT / "datasets/validation/validation_data.pkl",
            ),
            pd.read_pickle(
                PROJECT_ROOT / "datasets/validation/validation_data_benign.pkl",
            ),
            pd.read_pickle(
                PROJECT_ROOT / "datasets/haize/jailbreak_prompts_nathalie_cas_haize_1_short.pkl"
            ),
        ])

    if f"no_intervention_control_rated.pkl" in os.listdir(base_path):
      print(f"Skipping control - already processed")
      
      # Calculate ASR
      results_df = pd.read_pickle(base_path / f"no_intervention_control_rated.pkl")
      rating_col = f"rating_control_no_intervention_response"
      asr = results_df[rating_col].value_counts().get("Yes", 0)
      print(f"ASR: {asr}")
      return asr               
  
    harmful_map = {
    "harmful": 0,
    "harmless": 1,
    "haize": 2
    }
    harmful_v = harmful_map[harmful]
        
    validation_path = [
        project_root / "datasets/validation/validation_data.pkl",
        project_root / "datasets/validation/validation_data_benign.pkl", 
        project_root / "datasets/haize/jailbreak_prompts_nathalie_cas_haize_1_short.pkl"
    ][harmful_v]
    
    df = pd.read_pickle(validation_path)
  
    df["control_no_intervention_response"] = None  

    # Run on every element of the sample
    
    model.cuda()
    for index, row in df.iterrows():
      
      prompt = row["jailbreak_prompt_text"]
      print("prompt: ", prompt)
      
      input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

      outputs = model.generate(**input_ids, max_new_tokens=200)

      response = tokenizer.decode(outputs[0])
      
      
      if response.startswith(prompt): response = response[len(prompt):].strip()
      if response.startswith("<|begin_of_text|>"+prompt): response = response[len("<|begin_of_text|>"+prompt):].strip()
      if response.startswith("<bos>"+prompt): response = response[len("<bos>"+prompt):].strip()

      df.at[index, "control_no_intervention_response"] = response
      print(response)

      # Save periodically
      if index % 20 == 0:
        folder_path = project_root / f"datasets/{model_name}/final/{harmful}/"
        os.makedirs(folder_path, exist_ok=True)
        df.to_pickle(folder_path / f"no_intervention_control.pkl")
    
    df.to_pickle(folder_path / f"no_intervention_control.pkl")
    print(f"Data saved to {str(folder_path / f"no_intervention_control.pkl")}")
    model.cpu()
    
    autograder.cuda()
    
    results_df = grade(
      autograder=autograder,
      autogr_tokenizer=autogr_tokenizer,
      df=df,
      request_column="jailbreak_prompt_text",
      column_to_grade=f"control_no_intervention_response",
      output_path=project_root / f"datasets/{model_name}/final/harmful/no_intervention_control_rated.pkl"
      )
    autograder.cpu()
    
                    
    # Calculate ASR
    rating_col = f"rating_control_no_intervention_response"
    asr = results_df[rating_col].value_counts().get("Yes", 0)
    print(f"ASR: {asr}")
    
    # Create results dictionary
    new_result = {
        "asr": asr,
        "model": model_name,
        "probe_type": "control",
        "offensive_defensive": None,
        "c_strength": None,
        "lr": None
    }

    # Handle CSV operations
    if results_csv_path.exists():
        existing_df = pd.read_csv(results_csv_path)
        # Convert the new result to a single-row DataFrame
        new_result_df = pd.DataFrame([new_result])
        # Concatenate with existing results
        combined_df = pd.concat([existing_df, new_result_df], ignore_index=True)
    else:
        # Create new DataFrame with just the new result
        combined_df = pd.DataFrame([new_result])

    # Save to CSV
    combined_df.to_csv(results_csv_path, index=False)
    print(f"Updated results saved to {results_csv_path}")
    
    return asr



def process_probe(probe_path, model_name, model, tokenizer, input_size, middle_layer, c_strength, lr, offensive_defensive, harmful, project_root):
    # Setup model
    model.cuda()
    device = next(model.parameters()).device
    print(f"Model device: {device}")

    hflm_model = HFLM(model)
    hflm_model.model.to(device)
    assert check_all_parameters(hflm_model.model)
    
    # Determine probe type from path
    probe_type = next(ptype for ptype in PROBE_CONFIGS.keys() if ptype in str(probe_path))
    assert probe_type in str(probe_path)
    
    # Load probe based on type
    config = PROBE_CONFIGS[probe_type]
    if probe_type == "linear":
        probe = config["loader"](probe_path)
    else:
        probe = config["probe_class"](input_size=input_size)
        probe.load_probe(probe_path)
        probe.to(device)
        assert check_all_parameters(probe)
    
    harmful_map = {
    "harmful": 0,
    "harmless": 1,
    "haize": 2
    }
    harmful_v = harmful_map[harmful]
        
    validation_path = [
        project_root / "datasets/validation/validation_data.pkl",
        project_root / "datasets/validation/validation_data_benign.pkl", 
        project_root / "datasets/haize/jailbreak_prompts_nathalie_cas_haize_1_short.pkl"
    ][harmful_v]
    df = pd.read_pickle(validation_path)
    
    # Process each sample
    offensive = True if offensive_defensive == "offensive" else False
    for index, row in df.iterrows():
        # Get response using appropriate function
        if probe_type == "linear":
            response = config["response_fn"](
                model=model,
                tokenizer=tokenizer,
                prompt = row["jailbreak_prompt_text"],
                probe=probe,
                layers_to_intervene=[middle_layer],
                intervention_strength=c_strength
            )
        else:
            response = config["response_fn"](
                model=model,
                tokenizer=tokenizer,
                prompt=row["jailbreak_prompt_text"],
                probe=probe,
                layers_to_intervene=[middle_layer],
                c=c_strength,
                learning_rate=lr,
                offensive=offensive
            )
        
        df.at[index, f"response_{probe_type}_{model_name}_cstrength={c_strength}_lr={lr}"] = response
        print(response, "\n\n")
        
        # Save periodically
        if index % 20 == 0:
            folder_path = project_root / f"datasets/{model_name}/final/{harmful}/"
            os.makedirs(folder_path, exist_ok=True)
            df.to_pickle(folder_path / f"{probe_type}_{offensive_defensive}_c{c_strength}_lr{lr}.pkl")
    
    df.to_pickle(folder_path / f"{probe_type}_{offensive_defensive}_c{c_strength}_lr{lr}.pkl")
    model.cpu()
    return df


def grade(autograder, autogr_tokenizer, df, request_column, column_to_grade, output_path):  
  # Apply the grading function and add the ratings column
  df[f"rating_{column_to_grade}"] = df.apply(lambda row: grade_with_HF(row[request_column], [row[column_to_grade]], autograder, autogr_tokenizer, verbose = True)[0], axis=1)
  print("rated")
  # Save the dataframe to the output path
  df.to_pickle(output_path)
  print(output_path)
  return df


def create_probe_specific_df(probe_type, probe_path, c_strengths, learning_rates=None, linear_strengths=None):
    """Create a DataFrame for a specific probe type with its relevant parameters."""
    if probe_type == 'linear':
        data = {
            'model': [None] * len(linear_strengths),
            'capability': [None] * len(linear_strengths),
            'probe_type': ['linear'] * len(linear_strengths),
            'offensive_defensive': ['offensive' if strength > 0 else 'defensive' for strength in linear_strengths],
            'c_strength': linear_strengths,
            'lr': [None] * len(linear_strengths),
            'probe_path': [probe_path] * len(linear_strengths)
        }
    else:
        combinations = list(product(
            c_strengths,
            learning_rates,
            ['offensive', 'defensive']
        ))
        
        data = {
            'model': [None] * len(combinations),
            'capability': [None] * len(combinations),
            'probe_type': [probe_type] * len(combinations),
            'offensive_defensive': [comb[2] for comb in combinations],
            'c_strength': [comb[0] for comb in combinations],
            'lr': [comb[1] for comb in combinations],
            'probe_path': [probe_path] * len(combinations)
        }
    
    return pd.DataFrame(data)


def get_untested_configurations(probe_type, df, base_path, processed_configs):
    """Find configurations that haven't been tested yet."""
    untested_configs = []
    
    for idx, row in df.iterrows():
        c_strength = row['c_strength']
        lr = row['lr']
        od = row['offensive_defensive']
        
        # Create unique identifier for this configuration
        config_id = f"{probe_type}_{od}_c{c_strength}_lr={lr}"
        
        # Check if configuration has been processed
        if config_id in processed_configs:
            continue
            
        # Check if rated pickle file exists
        if f"{config_id}_rated.pkl" in os.listdir(base_path):
            continue
            
        untested_configs.append((idx, row))
    
    return untested_configs

def search_probe_parameters(
    model_name, model, tokenizer, input_size, middle_layer,
    autograder, autogr_tokenizer, project_root,
    transformer_path, mlp_path, linear_path,
    c_strengths, learning_rates, linear_strengths, results_csv_path, baseline, single = ""
):
    """Perform hyperparameter search for each probe type separately."""
    
    # Track which configurations have been processed
    processed_configs = set()
    if results_csv_path.exists():
        existing_df = pd.read_csv(results_csv_path)
        # Create unique identifier for each configuration
        for _, row in existing_df.iterrows():
            config_id = f"{row['probe_type']}_{row['offensive_defensive']}_c{row['c_strength']}_lr{row['lr']}"
            processed_configs.add(config_id)
    
    probe_configs = [
        ('linear', linear_path, linear_strengths),
        ('mlp', mlp_path, c_strengths),
        ('transformer', transformer_path, c_strengths)
    ]
    
    results = []
    base_path = project_root / f"datasets/{model_name}/final/harmful_{single}"
    base_path.mkdir(parents=True, exist_ok=True)
    
    for probe_type, path, strengths in probe_configs:
        print(f"\nProcessing {probe_type} probe type...")
        
        # Create DataFrame for current probe type
        if probe_type == 'linear':
            df = create_probe_specific_df(probe_type, path, None, 
                                        linear_strengths=strengths)
        else:
            df = create_probe_specific_df(probe_type, path, strengths, 
                                        learning_rates=learning_rates)
        
        # Get untested configurations
        untested_configs = get_untested_configurations(probe_type, df, base_path, processed_configs)
        
        if not untested_configs:
            print(f"All configurations for {probe_type} have been tested. Skipping...")
            continue
            
        print(f"Found {len(untested_configs)} untested configurations for {probe_type}")
        
        # Track if we've found good configurations for both offensive and defensive
        found_good_offensive = False
        found_good_defensive = False
        
        for idx, row in untested_configs:
            od = row['offensive_defensive']
            
            if (od == 'offensive' and found_good_offensive) or \
               (od == 'defensive' and found_good_defensive):
                continue
                
            # Add new check here for existing successful results
            if results_csv_path.exists():
                existing_df = pd.read_csv(results_csv_path)
                if not existing_df.empty:
                    matching_rows = existing_df[(existing_df['probe_type'] == probe_type) & 
                                            (existing_df['offensive_defensive'] == od)]
                    if not matching_rows.empty:
                        if od == 'offensive' and any(matching_rows['asr'] > 90):
                            print(f"Skipping {probe_type} - {od} - already have successful offensive result")
                            continue
                        elif od == 'defensive' and any(matching_rows['asr'] < 5):
                            print(f"Skipping {probe_type} - {od} - already have successful defensive result")
                            continue
            
            c_strength = row['c_strength']
            lr = row['lr']
            
            print(f"\nTesting {probe_type} - {od} with c_strength={c_strength}, lr={lr}")
            
            # Process probe and get results
            result_df = process_probe(
                probe_path=path,
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                input_size=input_size,
                middle_layer=middle_layer,
                c_strength=c_strength,
                lr=lr,
                offensive_defensive=od,
                harmful="harmful",
                project_root=project_root
            )
            
            # Grade results
            autograder.cuda()
            config_id = f"{probe_type}_{od}_c{c_strength}_lr={lr}"
            result_df = grade(
                autograder=autograder,
                autogr_tokenizer=autogr_tokenizer,
                df=result_df,
                request_column="jailbreak_prompt_text",
                column_to_grade=f"response_{probe_type}_{model_name}_cstrength={c_strength}_lr={lr}",
                output_path=base_path / f"{config_id}_rated.pkl"
            )
            autograder.cpu()
            
            # Calculate ASR
            rating_col = f"rating_response_{probe_type}_{model_name}_cstrength={c_strength}_lr={lr}"
            asr = result_df[rating_col].value_counts().get("Yes", 0)
            print(f"ASR: {asr}")
            
            # Store results
            row_dict = row.to_dict()
            row_dict.update({
                'asr': asr,
                'model': model_name
            })
            results.append(row_dict)
            
            # Update processed configurations
            processed_configs.add(config_id)
            
            if results_csv_path.exists():
                existing_df = pd.read_csv(results_csv_path)
                # Add only the new result
                updated_df = pd.concat([existing_df, pd.DataFrame([row_dict])], ignore_index=True)
            else:
                updated_df = pd.DataFrame([row_dict])                
            
            pd.DataFrame(updated_df).to_csv(results_csv_path, index=False)
            print(f"Updated results saved to {results_csv_path}")
            
            # Check if we've found a good configuration
            if od == 'offensive' and asr > 90:
                print(f"Found good offensive configuration for {probe_type}")
                found_good_offensive = True
            elif od == 'defensive' and asr < 5:
                print(f"Found good defensive configuration for {probe_type}")
                found_good_defensive = True


######### SCRIPT

def main(model_name, c_strengths, learning_rates, linear_strengths, linear_path, mlp_path, transformer_path, single= ""):   
    hf_token = get_HF_token()
    torch.cuda.empty_cache()
    
    # Setup language model
    hf_ident, num_layers, rating_column, input_size = get_model_details(model_name)
    middle_layer = num_layers // 2
    
    # Get autograder
    from gen_data.get_autograder import get_ministral_autograder
    autograder, autogr_tokenizer = get_ministral_autograder()
    
    check_baseline = f"datasets/{model_name}/causal_intervention_parameters_{single}.csv"
    
    if check_baseline not in os.listdir("."):
      print("No baseline found, creating one...")
      from transformers import AutoModelForCausalLM, AutoTokenizer
      from lm_eval.models.huggingface import HFLM
      
      model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B")
      tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
      baseline = get_baseline(model, model_name, tokenizer, harmful = "harmful", autograder = autograder, autogr_tokenizer = autogr_tokenizer, results_csv_path = PROJECT_ROOT / f"datasets/{model_name}/causal_intervention_parameters_{single}.csv")
      del model 
      gc.collect()
      torch.cuda.empty_cache()
    else:
      baseline = True
      
    # Load, process and grade probe results
    results = search_probe_parameters(
        model_name=model_name,
        model=None,
        tokenizer=None,
        input_size=input_size,
        middle_layer=middle_layer,
        autograder=autograder,
        autogr_tokenizer=autogr_tokenizer,
        transformer_path=transformer_path,
        mlp_path=mlp_path,
        linear_path=linear_path,
        c_strengths=c_strengths,
        learning_rates=learning_rates,
        linear_strengths=linear_strengths,
        results_csv_path=PROJECT_ROOT / f"datasets/{model_name}/causal_intervention_parameters_{single}.csv",
        baseline=baseline,
        single = single
    )


def parse_array(s):
    try:
        # Safely evaluate string representation of list
        return ast.literal_eval(s)
    except:
        raise argparse.ArgumentTypeError("Array must be in format [x, y, z]")

if __name__ == "__main__":
    print("starting...")
    
    parser = argparse.ArgumentParser(description='Run capability check')
    parser.add_argument('--model_name', type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument('--c_strengths', type=parse_array, default=None)
    parser.add_argument('--learning_rates', type=parse_array, default=None)
    parser.add_argument('--linear_strengths', type=parse_array, default=None)
    parser.add_argument('--linear_path', type=str, default="probes/linear_llama2.pt")
    parser.add_argument('--mlp_path', type=str, default="probes/mlp_llama2.pt")
    parser.add_argument('--transformer_path', type=str, default="probes/transformer_llama2.pt")
    parser.add_argument('--single', type=str, default="")
    args = parser.parse_args()
    
    linear_path = PROJECT_ROOT / args.linear_path
    mlp_path = PROJECT_ROOT / args.mlp_path
    transformer_path = PROJECT_ROOT / args.transformer_path
    
    main(
        model_name=args.model_name,
        c_strengths=args.c_strengths,
        learning_rates=args.learning_rates,
        linear_strengths=args.linear_strengths,
        linear_path=linear_path,
        mlp_path=mlp_path,
        transformer_path=transformer_path,
        single=args.single
    )

