import pandas as pd
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append('/data/nathalie_maria_kirch/ERA_Fellowship')

from analysis.unsupervised import *
from analysis.supervised import *
from gen_data.get_activations import *


## CONFIG
model_name = "google/gemma-7b-it"
layer=17


model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)


probe_path = "/data/nathalie_maria_kirch/ERA_Fellowship/experiments/nathaly/mlp_probe_weights.pth"


# Load validation data
df_val_adv = pd.read_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl")
df_val_bng = pd.read_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data_benign.pkl")



# Define an MLP model that accepts input of size [1, 8, 3072]
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Input layer to hidden layer with 8 neurons, process each of the 8 sequences separately
        self.fc1 = nn.Linear(3072, 8)
        self.relu = nn.ReLU()
        # Process the output from the 8 sequences, reduce to a single output
        self.fc2 = nn.Linear(8, 1)
        
        # ! correct loss 
        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x is expected to have shape [1, 8, 3072]
        # Process each of the 8 vectors separately through fc1
        x = self.fc1(x)  # Shape: [1, 8, 8]
        x = self.relu(x)
        x = self.fc2(x)  # Shape: [1, 8, 1]

        # Aggregate the outputs across the sequence dimension (dim=1)
        x = torch.mean(x, dim=1)  # Shape: [1, 1], now a scalar for each batch
        
        # ! correct loss
        # x = self.sigmoid(x)
        return x
      
    def load_model(self, file_path):
      self.load_state_dict(torch.load(file_path))
      self.eval()  # Set to evaluation mode
      print(f"Model weights loaded from {file_path}!")
      
    def save_model(self, file_path):
      torch.save(self.state_dict(), file_path)
      print(f"Model weights saved to {file_path}!")


from analysis.causal_interventions import get_res_layers_to_enumerate

def get_response_MLP(model, 
                 tokenizer,  
                 prompt,
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
                 c = 0,
                 loss = "MSE",
                 towards_target = True,
                 nr_perturbations=32, 
                 learning_rate = 0.1,
                 offensive = True):
    """
    Generates a response from a model with a causal intervention applied to specific layers.

    Args:
        model: Pretrained language model.
        tokenizer: Tokenizer compatible with the model.
        prompt (str): The input text prompt.
        probe: The intervention object (e.g., a linear model with coefficients).
        layers_to_intervene (list of int): Indices of the model layers to apply the intervention.
        max_new_tokens (int): Maximum number of new tokens to generate.
        intervention_strength (float): Strength of the intervention to apply.

    Returns:
        str: The generated text after applying the intervention.
    """
    print(f"Using {loss} loss function")
    print("learning rate", learning_rate)
    print("nr_perturbations", nr_perturbations)
    
    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = input_ids.size(1)
    
    print("loss", loss)
    # Define a loss function
    if loss == "MSE" : criterion = nn.MSELoss()
    
    # elif loss == "BCE" : criterion = nn.BCELoss()
    elif loss == "BCE" : criterion = nn.BCEWithLogitsLoss()
    
    elif loss == "BCE_new" : criterion = nn.BCELoss()
    
    sigmoid = nn.Sigmoid()


    def linear_intervene(name, c, offensive):
        def hook(model, input, output):
          
          with torch.enable_grad():
                      
            # Forward pass
            feature_vector = output[0][:, :, :].requires_grad_().to(output[0].device)          
            # print(feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 768]) True
            
            ## check old predicted class
            pred = mlp(feature_vector)
            print("OLD MLP_out:", pred)
            
            ### 
          
            
                    
            # Define perturbation as a parameter or tensor that requires gradient
            perturbation = torch.zeros_like(feature_vector, requires_grad=True).to(output[0].device)
            
            printed_once = False
            
            for i in range(nr_perturbations):
              # Apply perturbation to the feature vector
              perturbed_vector = feature_vector + perturbation

              # # Get output from the MLP
              # if loss == "BCE_new":
              #   mlp_out = sigmoid(mlp(perturbed_vector))
              # else: 
              mlp_out = mlp(perturbed_vector)
              
              mlp_out_value = mlp_out.item()
              
              # target_out = torch.zeros_like(mlp_out)  # Default to zero, or choose an appropriate value
              if offensive == True:

                  # target = mlp_out_value + c
                  target = max(mlp_out_value+c, c) # if out is very negative, target is c

              else: 
                # defensive
                target = min(mlp_out_value-c, -c) # if out is very positive, target is -c
              
              
              if not printed_once: 
                print("target = ", target)
                printed_once = True

              target_out = torch.full_like(mlp_out, target)               

              # Calculate the loss
              loss = criterion(mlp_out, target_out).to(output[0].device)

              # print(loss, loss.requires_grad)  # Should print a scalar tensor and True

              # Backward pass to compute gradients w.r.t the perturbation
              loss.backward()

              # Access the gradient of the loss w.r.t. the perturbation
              grad_input = perturbation.grad

              # Define a learning rate
              # learning_rate = 0.01

              
              if towards_target: 
                perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_().to(output[0].device)
              else: 
                perturbation = (perturbation + learning_rate * grad_input).clone().detach().requires_grad_().to(output[0].device)
              
            ## check new predicted class
            # pred = mlp(perturbation)
            pred = mlp(perturbed_vector)
            print("NEW MLP_out:", pred)
            
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
        hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index, c, offensive))
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
  
mlp = SimpleMLP()

# Load the model from the specified probe path
mlp.load_model(probe_path)



mlp.train()



df = pd.read_pickle("/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/validation_data.pkl")

c = 10
lr = 0.005
nr_perturbations = 34
for index, row in df.iterrows():
    response = get_response_MLP(model, tokenizer, row["jailbreak_prompt_text"], mlp, max_new_tokens=200, layers_to_intervene=[16], c = c, loss = "MSE", nr_perturbations = nr_perturbations, learning_rate=lr, offensive = True)
    df.at[index, "mlp_intervention_response"] = response
    print(response)


path = f"/data/nathalie_maria_kirch/ERA_Fellowship/datasets/latest/c/offensive_mlp_intervention_c={c}_lr={lr}_nr_p={nr_perturbations}.pkl"
df.to_pickle(path)

print(path)

