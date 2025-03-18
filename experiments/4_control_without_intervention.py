
# USING CHAT TEMPLATES 

import pandas as pd
import sys
from pathlib import Path
import os
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from tqdm import tqdm
import json
import gc
import re

# Import paths module
from experiments.paths import PROJECT_ROOT

# Add project root to system path
sys.path.append(str(PROJECT_ROOT))

from pathlib import Path

# Get the current working directory (where the notebook is)
notebook_dir = Path.cwd()

# Move up to the root directory (parent of 'notebooks')
project_root = notebook_dir.parent

sys.path.append(notebook_dir.parent)

from analysis.supervised import *
from analysis.causal_interventions import *
from analysis.Probes import SimpleMLP
from gen_data.harmBench_autograder import *
from analysis.models import *

def get_HF_token(): 
    return os.environ.get("HUGGINGFACE_TOKEN")

def get_pad_token(model_id,tokenizer):
    if tokenizer.pad_token is not None:
        return tokenizer.pad_token
    model_to_pad_token = {
        "meta-llama/Llama-3.2-3B-Instruct": "<|finetune_right_pad_id|>", # 128004,
        "google/gemma-7b-it": "<pad>",
        "mistralai/Ministral-8B-Instruct-2410": "<pad>",
        "meta-llama/Llama-3.1-8B-Instruct": "<|finetune_right_pad_id|>"
        
    }
    # pad_token = tokenizer.eos_token
    pad_token = model_to_pad_token[model_id]
    print(model_id, pad_token)
    return pad_token


def load_model_with_lora(
    base_model_id,
    lora_weights_path=None,
):
    token = get_HF_token()
    print(len(token))
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    # need to set the pad token, otherwise, training will fail
    # tokenizer.pad_token = tokenizer.eos_token # if this is set, the trained model will not know when to stop generating, it will however be fine for inference of the base model
    tokenizer.pad_token = get_pad_token(base_model_id, tokenizer)
    print("pad token", tokenizer.pad_token) 
    # tokenizer.pad_token ="<|finetune_right_pad_id|>" # 128004
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # model.resize_token_embeddings(len(tokenizer))
    # model.config.pad_token_id = tokenizer.pad_token_id
    
    if lora_weights_path is not None:
        # Load and apply LoRA weights
        model = PeftModel.from_pretrained(
            model,
            lora_weights_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    
    return model, tokenizer
  
def generate_responses_batched(model, tokenizer, prompts, batch_size=8, max_new_tokens=512):
    # Enable parallel processing if multiple GPUs available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    responses = []
    # Process prompts in batches
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating responses"):
        batch = prompts[i:i + batch_size]
        
        # Format prompts - remove the list wrapping since apply_chat_template already handles formatting
        formatted = [tokenizer.apply_chat_template({"role": "user", "content": p}, tokenize=False) for p in batch]
        
        # Tokenize
        inputs = tokenizer(formatted, return_tensors="pt", padding=True).to(model.device)
        input_length = inputs.input_ids.shape[1]
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                # temperature=0.7,
                # top_p=0.9,
                # repetition_penalty=1.1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode responses
        for output in outputs:
            response = tokenizer.decode(output[input_length:], skip_special_tokens=True).strip()
            if response.startswith("assistant"):
                response = response[len("assistant"):].strip()
            responses.append(response)
    
    return responses
  
def load_and_generate_responses_batched(base_model_id, lora_weights_path, prompts, batch_size=8, max_new_tokens=512):
    print("generate responses batched")
    model, tokenizer = load_model_with_lora(base_model_id, lora_weights_path)
    print("tokenizer pad token returned", tokenizer.pad_token)
    return generate_responses_batched(model, tokenizer, prompts, batch_size, max_new_tokens)


def main(dataset_path, model_name):
    print("contorl_without_intervention_new.py")
    if ".csv" in dataset_path:
      df = pd.read_csv(dataset_path)
    elif ".pkl" in dataset_path:
      df = pd.read_pickle(dataset_path)
    
    harmful_harmless = "harmful"
    if "benign" in str(dataset_path): 
      harmful_harmless = "harmless"
    
    filename = f"datasets/{model_name}/final/{harmful_harmless}/no_intervention_control_new.pkl"
    save_path = project_root / filename

    rated_save_path  = project_root / filename.replace(".pkl", "_rated.pkl")
    
    model_name, num_layers, rating_column, hf_ident, input_size = get_model_details(model_name)
    
    model = AutoModelForCausalLM.from_pretrained(hf_ident)
    tokenizer = AutoTokenizer.from_pretrained(hf_ident)
    
    model.to("cuda")   
    
    df["control_no_intervention_response"] = None  
    lora_weights_path = None

    print("\n\nStage 1: Generating responses for prompts.\n")
    df[f'control_no_intervention_response'] = load_and_generate_responses_batched(hf_ident, lora_weights_path, df['jailbreak_prompt_text'], batch_size=16, max_new_tokens=1024)
    print(f"\n\nStage 1: Saving results to {save_path}...\n")
    df.to_pickle(save_path)
    
    model.cpu()

      
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
