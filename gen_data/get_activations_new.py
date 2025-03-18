import torch
from transformers import AutoModelForCausalLM, AutoTokenizer 
from datasets import load_dataset
import pandas as pd 
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import pickle
import sys

from .helper import save_with_timestamp

# Ensure environment variables are loaded
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")




def get_pad_token(model_id, tokenizer):
    if tokenizer.pad_token is not None:
        return tokenizer.pad_token
    model_to_pad_token = {
        "meta-llama/Llama-3.2-3B-Instruct": "<|finetune_right_pad_id|>",
        "google/gemma-7b-it": "<pad>",
        "mistralai/Ministral-8B-Instruct-2410": "<pad>",
        "meta-llama/Llama-3.1-8B-Instruct": "<|finetune_right_pad_id|>"
    }
    pad_token = model_to_pad_token[model_id]
    print(f"Using pad token for {model_id}: {pad_token}")
    return pad_token

def get_model(model_name: str):
    # Load the model with float16 precision
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Added float16 precision
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Set the pad token using the same logic as control experiments
    tokenizer.pad_token = get_pad_token(model_name, tokenizer)
    print(f"Pad token set to: {tokenizer.pad_token}")
        
    return model, tokenizer



def get_res_layers_to_enumerate(model):
    model_name = model.config._name_or_path
    if 'gpt' in model_name:
        return model.transformer.h
    elif 'pythia' in model_name:
        return model.gpt_neox.layers
    elif 'bert' in model_name:
        return model.encoder.layer
    elif 'mistral' in model_name:
        return model.model.layers
    elif 'gemma' in model_name:
        return model.model.layers
    elif "llama" in model_name:
        return model.model.layers
    else:
      raise ValueError(f"Unsupported model: {model_name}.")


def get_res_activations(model, 
                    tokenizer,  
                    prompt,
                    type_of_activations = "res_activations", 
                    token_position = "all",
                    generate_output = False,
                    debug_prompt = False):
    """ Returns a torch.Tensor of activations. 

    activations.shape = torch.Size([batch, seq, num_hidden_layers, hidden_size])

    token_position = "last | "all" (default = "all")
    """

    device = next(model.parameters()).device

    # The prompt is already formatted by get_activations_from_df
    formatted_prompt = prompt
    
    if debug_prompt:
        print("\nPrompt Formatting Debug:")
        print(f"Original prompt: {prompt[:100]}...")
        print(f"Formatted prompt: {formatted_prompt[:100]}...")
        print(f"Tokenized length: {len(tokenizer.encode(formatted_prompt))}")
        print("First 10 tokens:", tokenizer.decode(tokenizer.encode(formatted_prompt)[:10]))
    
    # Single tokenization call
    inputs = tokenizer(formatted_prompt, 
                      return_tensors="pt",
                      truncation=True,
                      padding=True).to(device)
    
    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            if token_position == "all": 
                activations[name].append(output[0][:, :, :].detach().cpu())
            elif token_position == "last":
                activations[name].append(output[0][:, -1, :].detach().cpu())
            else: 
                raise ValueError(f"Unsupported token_position: {token_position}")
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks
    for i, layer in enumerate(layers_to_enum):
        hook_handle = layer.register_forward_hook(get_activation(i))
        hooks.append(hook_handle)

    # Process the dataset
    model.eval()
    with torch.no_grad():
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    print("Activations shape:", activations[0][0].shape)

    if generate_output:
        output_sequence = model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            num_return_sequences=1, 
            max_new_tokens=200, 
            temperature=0, 
            do_sample=False, 
            pad_token_id=tokenizer.eos_token_id
        )
        
        generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
        
        # Remove the chat template prefix (if any) and input prompt from the generated text
        if generated_text.startswith(formatted_prompt):
            # If the formatted prompt (including any prefix) is at the start
            generated_text = generated_text[len(formatted_prompt):].strip()
        elif prompt in generated_text:
            # If the raw prompt appears somewhere in the text
            prompt_start = generated_text.find(prompt)
            generated_text = generated_text[prompt_start + len(prompt):].strip()

        return {'output_text': generated_text, type_of_activations: activations}
    
    return {type_of_activations: activations}


def get_activations_from_df(model_name, df, prompt_column, n_rows=None, n_start=0, dataset_name="", 
                          token_position="all", generate_output=False, type_of_activations="res_activations",
                          debug_samples=3, save_interval=1000):
    # get model 
    model, tokenizer = get_model(model_name)

    df[type_of_activations] = None
    
    counter = 0 

    if n_rows == None: 
        n_rows = len(df) 
    
    print(f"Dataset loaded successfully with {n_rows} rows")

    if debug_samples > 0:
        print("\nChecking prompt formatting for first few samples:")
        for i in range(min(debug_samples, len(df))):
            prompt = df.iloc[i][prompt_column]
            # Pass conversation directly without 'messages' keyword
            formatted = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)
            print(f"\nSample {i+1}:")
            print(f"Original: {prompt[:100]}...")
            print(f"Formatted: {formatted[:100]}...")
            print(f"Token length: {len(tokenizer.encode(formatted))}")
            print("-" * 50)

    # Create initial save of empty dataframe to get the filename
    filename = save_with_timestamp(
        model_name=model_name, 
        dataset_name=dataset_name, 
        save=True, 
        df=df, 
        add_info=type_of_activations
    )
    
    for row in df.iloc[n_start:n_start+n_rows].iterrows():
        counter += 1 
        index = row[0]
        print(f"Processing prompt {counter}/{n_rows}")
    
        prompt = row[1][prompt_column]
        # Pass conversation directly without 'messages' keyword
        formatted_prompt = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False)

        output_activations = get_res_activations(
            model=model, 
            tokenizer=tokenizer, 
            token_position=token_position, 
            prompt=formatted_prompt, 
            type_of_activations=type_of_activations, 
            generate_output=generate_output
        )
        
        df.at[index, type_of_activations] = output_activations[type_of_activations]
        
        if generate_output:
            df.at[index, 'output_text'] = output_activations['output_text']

        if counter % save_interval == 0:
            print(f"\nSaving progress at sample {counter}...")
            df.to_pickle(filename)
            print(f"Progress saved to: {filename}")

    print("Processing complete")
    
    df.to_pickle(filename)
    return filename



# Function to average across the sequence dimension
def average_across_sequence(activation):
    return activation.mean(dim=1)


def get_activations_at_token_position(): 
    ##  TODO
    # 
    
    pass

