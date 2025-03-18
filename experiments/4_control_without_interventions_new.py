# USING CHAT TEMPLATES AND BATCHING 

import pandas as pd
import sys
from pathlib import Path
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from peft import PeftModel

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


def load_model_with_lora(base_model_id, lora_weights_path=None):
    # Load the base model with minimal configuration
    model = AutoModelForCausalLM.from_pretrained(base_model_id)
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
  
    
    if lora_weights_path is not None:
        # Load and apply LoRA weights
        model = PeftModel.from_pretrained(
            model,
            lora_weights_path,
            device_map="auto"
        )
    
    
    return model, tokenizer

def generate_responses_batched(model, tokenizer, prompts, batch_size=8, max_new_tokens=200):
    responses = []
    
    for prompt in tqdm(prompts, desc="Generating responses"):
        message = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(message, tokenize=False)
        
        inputs = tokenizer(formatted, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=1,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
            

        full_output = tokenizer.decode(output[0], skip_special_tokens=True)
        
        try:
            if "assistant" in full_output:
                response = full_output.split("assistant")[-1].strip()
            else:
                response = full_output.replace(prompt, "").strip()
            
            responses.append(response)
            
        except Exception as e:
            print(f"Error processing response: {e}")
            print(f"Full output was: {full_output}")
            responses.append("")  # Append empty string on error
            
    return responses


def load_and_generate_responses_batched(base_model_id, lora_weights_path, prompts, batch_size=8, max_new_tokens=200):
    model, tokenizer = load_model_with_lora(base_model_id, lora_weights_path)
    model.cuda()  # Explicitly move to CUDA like in second script
    responses = generate_responses_batched(model, tokenizer, prompts, batch_size, max_new_tokens)
    model.cpu()  # Move back to CPU like in second script
    return responses

def main(dataset_path, model_name):
    # Load dataset
    if ".csv" in dataset_path:
        df = pd.read_csv(dataset_path)
    elif ".pkl" in dataset_path:
        df = pd.read_pickle(dataset_path)
    
    harmful_harmless = "harmful" if "benign" not in str(dataset_path) else "harmless"
    
    filename = f"datasets/{model_name}/final/{harmful_harmless}/no_intervention_control_new.pkl"
    save_path = project_root / filename
    rated_save_path = project_root / filename.replace(".pkl", "_rated.pkl")
    
    model_name, num_layers, rating_column, hf_ident, input_size = get_model_details(model_name)
    
    df["control_no_intervention_response"] = None  
    lora_weights_path = None

    print("\n\nStage 1: Generating responses for prompts.\n")
    df['control_no_intervention_response'] = load_and_generate_responses_batched(
        hf_ident, 
        lora_weights_path, 
        df['jailbreak_prompt_text'], 
        batch_size=16, 
        max_new_tokens=200  # Match the max_new_tokens from second script
    )
    
    print(f"\n\nStage 1: Saving results to {save_path}...\n")
    df.to_pickle(save_path)
    print(f"Data saved to {save_path}")
    
    rate_dataset(df, 
                request_column="jailbreak_prompt_text", 
                response_column="control_no_intervention_response", 
                output_path=rated_save_path)
    
    print("rated data saved to", rated_save_path)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python your_python_script.py <dataset_path> <model_name>")
        sys.exit(1)

    dataset_path = sys.argv[1]  
    model_name = sys.argv[2]

    main(dataset_path, model_name)