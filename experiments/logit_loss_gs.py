import pandas as pd
import sys
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
import torch.optim as optim
from gen_data.get_activations import get_res_layers_to_enumerate

sys.path.append('/data/nathalie_maria_kirch/ERA_Fellowship')

from analysis.supervised import *
from analysis.causal_interventions import *
from analysis.Probes import SimpleMLP
from gen_data.harmBench_autograder import grade_with_HF


def get_response_MLP_logit(model, 
                 tokenizer,  
                 prompt,
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
                 target = 0,
                 towards_target = True,
                 nr_perturbations=32,
                 learning_rate = 0.1):
    """
    Generates a response from a model with a causal intervention applied to specific layers.

    Args:
        model: Pretrained language model.
        tokenizer: Tokenizer compatible with the model.
        prompt (str): The input text prompt.
        mlp: The MLP model used for intervention.
        layers_to_intervene (list of int): Indices of the model layers to apply the intervention.
        max_new_tokens (int): Maximum number of new tokens to generate.
        intervention_strength (float): Strength of the intervention to apply.

    Returns:
        str: The generated text after applying the intervention.
    """
    print(f"logit loss")
    print("nr perturbations", nr_perturbations)
    print("learning rate", learning_rate)

    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = input_ids.size(1)
    
    def linear_intervene(name, target):
        def hook(model, input, output):
          
          with torch.enable_grad():
                      
            # Forward pass
            feature_vector = output[0][:, :, :].requires_grad_()          
            print(feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 768]) True
            
            ## check old predicted class
            pred = mlp(feature_vector)

            # Apply a sigmoid to convert it into a probability (between 0 and 1)
            probability = torch.sigmoid(pred)

            # Check which category it belongs to (class 0 or 1)
            predicted_class = (probability > 0.5).float()
            # print(predicted_class)

            print("OLD Predicted class:", predicted_class)
            
            ### 
            
                    
            # Define perturbation as a parameter or tensor that requires gradient
            perturbation = torch.zeros_like(feature_vector, requires_grad=True)
            
            printed_once = True

            for i in range(nr_perturbations):
              # Apply perturbation to the feature vector
              perturbed_vector = feature_vector + perturbation

              # Get output from the MLP
              mlp_out = mlp(perturbed_vector)
            
              if target == 0:
                loss = mlp_out
                              
              if target == 1:
                loss = -1 * mlp_out

              # print(loss, loss.requires_grad)  # Should print a scalar tensor and True

              # Backward pass to compute gradients w.r.t the perturbation
              loss.backward()

              # Access the gradient of the loss w.r.t. the perturbation
              grad_input = perturbation.grad

              
              if towards_target: 
                perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_()
              else: 
                perturbation = (perturbation + learning_rate * grad_input).clone().detach().requires_grad_()
              
            ## check new predicted class
            # pred = mlp(perturbation)
            pred = mlp(perturbed_vector)

            # Apply a sigmoid to convert it into a probability (between 0 and 1)
            probability = torch.sigmoid(pred)

            # Check which category it belongs to (class 0 or 1)
            predicted_class = (probability > 0.5).float()

            print("NEW Predicted class:", predicted_class.item())
            
            ### 
            
            
            new_out = perturbed_vector, output[1] # concat perturbed vector with the rest of the output

            # returnn perturbation
            return new_out
          
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks for each specified layer
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index, target))
        hooks.append(hook_handle)
    
    try:
        # model.eval()
        with torch.no_grad():
          output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)
            
    finally:
        # Ensure all hooks are removed after use
        for h in hooks:
            h.remove()
    
    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # If outptut sequence repeats prompt, remove it 
    if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

    return generated_text

def main(dataset_path, probe_path, target, nr_perturbations, learning_rate, autograder, autograder_tokenizer):
    save_path = f"/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/temporary_files/gs/bce_target={target}_nrperturbr{nr_perturbations}_lr={learning_rate}_gs.pkl"

    if ".csv" in dataset_path:
      df = pd.read_csv(dataset_path)
    elif ".pkl" in dataset_path:
      df = pd.read_pickle(dataset_path)

    print("Dataset loaded")
    
    df = df[:10]

    ## CONFIG
    model_name = "google/gemma-7b-it"
    
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
  
    mlp = SimpleMLP()

    # Load the model from the specified probe path
    mlp.load_model(probe_path)
    
    df["MLP_intervention_response"] = None  
    df["rating"] = None

    # Run on every element of the sample
    for index, row in df.iterrows():
        # response = get_response_MLP_logit(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=100, layers_to_intervene=[16], target=target, nr_perturbations=nr_perturbations, learning_rate=learning_rate)
        response = get_response_MLP(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=200, layers_to_intervene=[16], target=target, loss = "BCE", nr_perturbations=nr_perturbations, learning_rate=learning_rate)
        
        df.at[index, "MLP_intervention_response"] = response
        print(response)


        # Save every 20 samples
        if index % 2 == 0:
            df.to_pickle(save_path)
      
    try:
      #rate the responses   
      df["rating"] = df.apply(lambda row: grade_with_HF(row['jailbreak_prompt_text'], [row['MLP_intervention_response']], autograder, autograder_tokenizer, verbose = True)[0], axis=1)
    except: 
      print("Error in grading")
      
    df.to_pickle(save_path)
    print(f"Data saved to {save_path}")



learning_rates = [0.1, 0.075, 0.05, 0.025, 0.01]
number_perturbations = [30, 60, 90, 150, 200, 250, 300, 350, 400, 450, 500]


# Load the HarmBench model
autograder_name = "cais/HarmBench-Llama-2-13b-cls"
autograder = AutoModelForCausalLM.from_pretrained(autograder_name)
# autograder.to('cuda')
autograder_tokenizer = AutoTokenizer.from_pretrained(autograder_name, use_fast=False, truncation_side="left", padding_side="left")

for learning_rate in learning_rates:
  for nr_perturbations in number_perturbations:
    for target in [0, 1]:
      if learning_rate in [0.1, 0.075] and nr_perturbations > 150: 
        continue
      dataset_path = "/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl" 
      probe_path = "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth" 
      main(dataset_path, probe_path, target, nr_perturbations, learning_rate, autograder, autograder_tokenizer)
      
    

