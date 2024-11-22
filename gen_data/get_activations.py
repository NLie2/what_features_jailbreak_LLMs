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




def get_model(model_name: str):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer



def get_res_layers_to_enumerate(model):
    model_name = model.config._name_or_path
    if 'gpt' in model_name:
        return model.transformer.h
    elif 'pythia' in model_name:
        return model.gpt_neox.layers
    elif 'bert' in model_name:
        return model.encoder.layer
    elif 'Mistral' in model_name:
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
                    generate_output = False):
    """ Returns a torch.Tensor of activations. 

    activations.shape = torch.Size([batch, seq, num_hidden_layers, hidden_size])

    token_position = "last | "all" (default = "all")
    """

    device = next(model.parameters()).device

    #model, tokenizer = models.get_model_from_name(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True, padding=True).to(device)
    len_prompt = len(input_ids)

    activations = {}
    def get_activation(name):
        def hook(model, input, output):
            if name not in activations:
                activations[name] = []
            ## dimensions correspond to 
              # n sequences in the batch
              # n tokens in each sequence
              # hidden size of res stream
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
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        _ = model(**inputs)

    for h in hooks:
        h.remove()

    print("Activations shape:", activations[0][0].shape)

    if generate_output:
      # also add model.generate and save the output 
      
      # set do_sample to True for stochastic output
      output_sequence = model.generate(input_ids, attention_mask=inputs['attention_mask'], num_return_sequences=1, max_new_tokens=200, temperature = 0, do_sample=False, pad_token_id=tokenizer.eos_token_id )
      
      generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
      
      # Check if the generated text starts with the input prompt
      if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()


      return {'output_text': generated_text, type_of_activations : activations}
    
    return {type_of_activations : activations}


def get_activations_from_df(model_name, df, prompt_column, n_rows= None, n_start = 0,  dataset_name =  "", token_position = "all", generate_output = False, type_of_activations = "res_activations"): 
    # get model 
    model, tokenizer = get_model(model_name)

    df[type_of_activations] = None
    
    counter = 0 

    if n_rows == None: 
        n_rows = len(df) 
    
    print(f"Dataset loaded successfully with {n_rows} rows")

    for row in df.iloc[:n_rows].iterrows():
        counter+=1 
        index = row[0]
        print(f"Processing prompt {counter + 1}/{n_rows}")
    
        prompt = row[1][prompt_column]

        output_activations = get_res_activations(model = model, tokenizer = tokenizer, token_position = token_position, prompt = prompt, type_of_activations=type_of_activations, generate_output=generate_output)
        
        df.at[index, type_of_activations] = output_activations[type_of_activations]
        
        if generate_output: df.at[index, 'output_text'] = output_activations['output_text']

    print("Processing complete")
    
    return save_with_timestamp(model_name=model_name, dataset_name=dataset_name, save = True, df=df, add_info = type_of_activations)



# Function to average across the sequence dimension
def average_across_sequence(activation):
    return activation.mean(dim=1)


def get_activations_at_token_position(): 
    ##  TODO
    # 
    
    pass

