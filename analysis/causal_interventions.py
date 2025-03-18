import torch
import torch.nn as nn
import torch.optim as optim
import os
import json
import pandas as pd
from pathlib import Path
from gen_data.get_activations import get_res_layers_to_enumerate


## LINEAR INTERVENTION 
def get_response_linear(model, 
                 tokenizer,  
                 prompt,
                 probe,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=200, 
                 intervention_strength=1,
                 ):
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

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = input_ids.size(1)

    def linear_intervene(name):
        def hook(model, input, output):
          output[0][:, :, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)
          return output
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks for each specified layer
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index))
        hooks.append(hook_handle)
    
    try:
        model.eval()
        with torch.no_grad():
            output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)

    finally:
        # Ensure all hooks are removed after use
        for h in hooks:
            h.remove()
    
    # generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

    # return generated_text
        response = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # If outptut sequence repeats prompt, remove it 
    # if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()
    # Extract response text
    if "user\n" in response:
            parts = response.split("user\n")
            if len(parts) > 1:
                response = parts[-1].split("\n", 1)
                response = response[1].strip() if len(response) > 1 else response[0].strip()
    else:
            response = response[len(prompt):].strip() if response.startswith(prompt) else response
            

    return response   



def insert_linear_hook(model, 
                 tokenizer,  
                 probe,
                 layers_to_intervene=None,  # Allow multiple layers
                 intervention_strength=1,
                 tokens="all", add_to="prompt"):
    """
    """

    def linear_intervene(name):
        def hook(model, input, output):
          device = next(model.parameters()).device
          output[0][:, :, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)            
          return output
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks for each specified layer
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index))
        hooks.append(hook_handle)
        
        
### MLP INTERVENTION 
def get_response_MLP(model, 
                 tokenizer,  
                 prompt,
                 probe,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=200, 
                 c = 0,
                 loss = "MSE",
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
    print("offensive", offensive)

    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    
    print("loss", loss)
    # Define a loss function
    if loss == "MSE" : criterion = nn.MSELoss()
    
    # elif loss == "BCE" : criterion = nn.BCELoss()
    elif loss == "BCE" : criterion = nn.BCEWithLogitsLoss()
    
    elif loss == "BCE_new" : criterion = nn.BCELoss()
    

    def mlp_intervene(name, c, offensive):
        def hook(model, input, output):
          
          with torch.enable_grad():
                      
            # Forward pass
            #feature_vector = output[0][:, :, :].requires_grad_().to(output[0].device)          
            #print("SHAPE", feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 768]) True
            feature_vector = output.requires_grad_().to(output.device) if isinstance(output, torch.Tensor) else output[0].requires_grad_().to(output[0].device)                         
            ## check old predicted class
            pred = probe(feature_vector)
            print("OLD MLP_out:", pred)
            
            ### 
              
            # Define perturbation as a parameter or tensor that requires gradient
            perturbation = torch.zeros_like(feature_vector, requires_grad=True).to(output[0].device)
            
            printed_once = False
            
            for i in range(nr_perturbations):
              # Apply perturbation to the feature vector
              perturbed_vector = feature_vector + perturbation

              mlp_out = probe(perturbed_vector)
              
              mlp_out_value = mlp_out.item()
              
              if offensive == True:
                  target = max(mlp_out_value+c, c) # if out is very negative, target is 0

              else: 
                # defensive
                target = min(mlp_out_value-c, -c) # if out is very positive, target is 0
              
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

              perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_().to(output[0].device)

            ## check new predicted class
            # pred = mlp(perturbation)
            pred = probe(perturbed_vector)
            print("NEW MLP_out:", pred)
            
            ### 
            
            # print("output", output[1].shape)
            # new_out = perturbed_vector, output[1] # concat perturbed vector with the rest of the output

            # returnn perturbation
            # return new_out
            
            if isinstance(output, torch.Tensor):
                return perturbed_vector
            else:
                # Preserve the original structure if output was a tuple
                print("tuple")
                return (perturbed_vector,) + output[1:]
          
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks for each specified layer
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(mlp_intervene(layer_index, c, offensive))
        hooks.append(hook_handle)
    
    try:
        # model.eval()
        with torch.no_grad():
          output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)
            
    finally:
        # Ensure all hooks are removed after use
        for h in hooks:
            h.remove()
    
    response = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # If outptut sequence repeats prompt, remove it 
    # if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()
            # Extract response text
    if "user\n" in response:
            parts = response.split("user\n")
            if len(parts) > 1:
                response = parts[-1].split("\n", 1)
                response = response[1].strip() if len(response) > 1 else response[0].strip()
    else:
            response = response[len(prompt):].strip() if response.startswith(prompt) else response
            

    return response                    
  

def insert_mlp_hook(model, 
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 offensive = True, 
                 nr_perturbations = 34,
                 learning_rate = 0.005,
                 c = 0):
    """
    """
    print("offensive", offensive)

    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    criterion = nn.MSELoss()
    def mlp_intervene(name, c, offensive):
      
        def hook(model, input, output):
                          
          with torch.enable_grad():
                      
            # Forward pass
            # feature_vector = output[0][:, :, :].requires_grad_().to(output[0].device)
            feature_vector = output.requires_grad_().to(output.device) if isinstance(output, torch.Tensor) else output[0].requires_grad_().to(output[0].device)
   
            # Define perturbation as a parameter or tensor that requires gradient
            perturbation = torch.zeros_like(feature_vector, requires_grad=True).to(feature_vector.device)
            
            printed_once = False
            
            for i in range(nr_perturbations):
              # Apply perturbation to the feature vector
              perturbed_vector = feature_vector + perturbation

              mlp_out = mlp(perturbed_vector)
              
              mlp_out_value = mlp_out.item()
              
              # target_out = torch.zeros_like(mlp_out)  # Default to zero, or choose an appropriate value
              if offensive:
            
                  # target = mlp_out_value + c
                  target = max(mlp_out_value+c, c) # if out is very negative, target is c

              else: 
                # defensive
                target = min(mlp_out_value-c, -c) # if out is very positive, target is -c
                    
              
              if not printed_once: 
                # print("target = ", target)
                printed_once = True

              target_out = torch.full_like(mlp_out, target)               

              # Calculate the loss
              loss = criterion(mlp_out, target_out).to(feature_vector.device)

              # Backward pass to compute gradients w.r.t the perturbation
              loss.backward()

              # Access the gradient of the loss w.r.t. the perturbation
              grad_input = perturbation.grad

              
              perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_().to(output[0].device)

            # print("NEW MLP_out:",  pred = mlp(perturbed_vector)) ## check new predicted class
                        
            # Clear unused variables
            del grad_input, mlp_out, loss
            torch.cuda.empty_cache()
                    
            # new_out = perturbed_vector, output[1] # concat perturbed vector with the rest of the output
            # return new_out
            if isinstance(output, torch.Tensor):
                return perturbed_vector
            else:
                # Preserve the original structure if output was a tuple
                return (perturbed_vector,) + output[1:]


          
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks for each specified layer
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(mlp_intervene(layer_index, c, offensive))
        hooks.append(hook_handle)
  
  
### TRANSFORMER INTERVENTION 

def get_response_transformer(model, 
                           tokenizer,  
                           prompt,
                           probe,
                           layers_to_intervene=None,
                           max_new_tokens=200, 
                           c=0,
                           loss="MSE",
                           nr_perturbations=32, 
                           learning_rate=0.1,
                           offensive=True):
    """
    Generates a response with transformer probe-based intervention.
    """
    print(f"Using {loss} loss function")
    print("learning rate", learning_rate)
    print("nr_perturbations", nr_perturbations)
    print("offensive", offensive)
    
    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    
    if loss == "MSE":
        criterion = nn.MSELoss()
    elif loss == "BCE":
        criterion = nn.BCEWithLogitsLoss()
    elif loss == "BCE_new":
        criterion = nn.BCELoss()
    

    def transformer_intervene(name, c, offensive):
        def hook(model, input, output):                                
            with torch.enable_grad():
                # Get feature vector and ensure it's on the right device
                print(output[0].shape)
                output_modified = output[0].clone()

                # feature_vector = output[0][:, -1, :].requires_grad_().to(output[0].device)
                # Forward pass
                # feature_vector = output[0][:, :, :].requires_grad_().to(output[0].device)
                feature_vector = output[:, -1, :].requires_grad_().to(output.device) if isinstance(output, torch.Tensor) else output[0][:, -1, :].requires_grad_().to(output[0].device)
                print("SHAPE", feature_vector.shape, feature_vector.requires_grad)  

                print(feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 1, input_size]), True

                
                ## check old predicted class
                pred = probe(feature_vector)
                print("OLD transformer_out:", pred)
                
                ### 

                # Initialize perturbation
                perturbation = torch.zeros_like(feature_vector, requires_grad=True).to(output[0].device)
                
                printed_once = False
                
                for i in range(nr_perturbations):
                    # Apply perturbation
                    perturbed_vector = feature_vector + perturbation
                    
                    # Forward through probe
                    probe_out = probe(perturbed_vector)
                    probe_out_value = probe_out.item()
                    
                    # Calculate target
                    if offensive:
                        target = max(probe_out_value + c, c)
                    else:
                        target = min(probe_out_value - c, -c)
                    
                    if not printed_once:
                        print("target = ", target)
                        printed_once = True

                    target_out = torch.full_like(probe_out, target)
                    
                    # Calculate loss
                    loss = criterion(probe_out, target_out).to(output[0].device)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update perturbation
                    grad_input = perturbation.grad
                    
                    perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_().to(output[0].device)

                # Final prediction check
                pred = probe(perturbed_vector)
                print("NEW Transformer_out:", pred.item())
                
                
                output_modified[:, -1, :] = perturbed_vector
                # new_out = output_modified , output[1]
                # print(new_out[0].shape)
                
                # return new_out 
                if isinstance(output, torch.Tensor):
                    return output_modified
                else:
                    # Preserve the original structure if output was a tuple
                    return (output_modified,) + output[1:]                       
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(
            transformer_intervene(layer_index, c, offensive)
        )
        hooks.append(hook_handle)
    
    try:
        with torch.no_grad():
            output_sequence = model.generate(
                input_ids, 
                num_return_sequences=1, 
                max_new_tokens=max_new_tokens, 
                do_sample=False,
                output_hidden_states=True
            )
            
    finally:
        # Remove hooks
        for h in hooks:
            h.remove()
    
    # generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # # Remove prompt from output if present
    # if generated_text.startswith(prompt):
    #     generated_text = generated_text[len(prompt):].strip()

    # return generated_text
    response = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # If outptut sequence repeats prompt, remove it 
    # if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()
            # Extract response text
    if "user\n" in response:
            parts = response.split("user\n")
            if len(parts) > 1:
                response = parts[-1].split("\n", 1)
                response = response[1].strip() if len(response) > 1 else response[0].strip()
    else:
            response = response[len(prompt):].strip() if response.startswith(prompt) else response
            

    return response   
 
def insert_transformer_hook(model, 
                 probe,
                 layers_to_intervene=None,  # Allow multiple layers
                 offensive = True, 
                 nr_perturbations = 34,
                 learning_rate = 0.005,
                 c = 0):
    """
    """
    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    criterion = nn.MSELoss()
    def transformer_intervene(name, c, offensive):
        def hook(model, input, output):
            with torch.enable_grad():
                # Get feature vector and ensure it's on the right device
                output_modified = output[0].clone()

                # feature_vector = output[0][:, -1, :].requires_grad_().to(output[0].device)
                feature_vector = output[:, -1, :].requires_grad_().to(output.device) if isinstance(output, torch.Tensor) else output[0][:, -1, :].requires_grad_().to(output[0].device)


                # print(feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 1, input_size]), True

                
                #print("OLD transformer_out:", probe(feature_vector))  ## check old predicted class
                
                # Initialize perturbation
                perturbation = torch.zeros_like(feature_vector, requires_grad=True).to(output[0].device)
                              
                for i in range(nr_perturbations):
                    # Apply perturbation
                    perturbed_vector = feature_vector + perturbation
                    
                    # Forward through probe
                    probe_out = probe(perturbed_vector)
                    probe_out_value = probe_out.item()
                    
                    # Calculate target
                    if offensive:
                        target = max(probe_out_value + c, c)
                    else:
                        target = min(probe_out_value - c, -c)
                

                    target_out = torch.full_like(probe_out, target)
                    
                    # Calculate loss
                    loss = criterion(probe_out, target_out).to(output[0].device)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Update perturbation
                    grad_input = perturbation.grad
                    
                    perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_().to(output[0].device)

                # print("NEW Transformer_out:", probe(perturbed_vector).item()) # Final prediction check
                
                
                # output_modified[:, -1, :] = perturbed_vector
                # new_out = output_modified, output[1]
                
                # return new_out       
              
              
                output_modified[:, -1, :] = perturbed_vector
                # new_out = output_modified , output[1]
                # print(new_out[0].shape)
                
                # return new_out 
                if isinstance(output, torch.Tensor):
                    return output_modified
                else:
                    # Preserve the original structure if output was a tuple
                    return (output_modified,) + output[1:]   
                       
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks for each specified layer
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(transformer_intervene(layer_index, c, offensive))
        hooks.append(hook_handle)
  
 
 
def update_json_file(file_path, key_string, result):
    """
    Updates or creates a JSON file with new key-value pairs, handling permission errors.
    
    Args:
        file_path (str): Path to the JSON file
        key_string (str): Key to be added/updated in the JSON
        result: Value to be stored for the key
    
    Returns:
        tuple: (dict, str) The updated dictionary and the path where it was written
    """
    
    print(f"try filepath {file_path}")
    def try_write_to_path(path):
        # Initialize existing_data
        existing_data = {}
        try:
            # Create directory structure if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Check if the file exists and has content
            if os.path.exists(path) and os.path.getsize(path) > 0:
                try:
                    with open(path, 'r') as json_file:
                        existing_data = json.load(json_file)
                except json.JSONDecodeError:
                    # If the file exists but is not valid JSON, start fresh
                    existing_data = {}
            
            # Add the new key-value pair
            existing_data[key_string] = result
            
            # Write back to the JSON file
            with open(path, 'w') as json_file:
                json.dump(existing_data, json_file, indent=4)
            
            return True, existing_data
        except PermissionError:
            return False, None
    
    # Try original path first
    success, data = try_write_to_path(file_path)
    if success:
        return data, file_path
    
    # If original path fails, try user's home directory
    home_path = os.path.join(os.path.expanduser('~'), 'experiment_results', 
                            os.path.basename(os.path.dirname(file_path)),
                            os.path.basename(file_path))
    success, data = try_write_to_path(home_path)
    if success:
        print(f"Warning: Could not write to {file_path}")
        print(f"Writing to alternative location: {home_path}")
        return data, home_path
    
    # If home directory fails, try temporary directory
    import tempfile
    temp_path = os.path.join(tempfile.gettempdir(), 
                            f"capabilities_{os.path.basename(os.path.dirname(file_path))}_{os.path.basename(file_path)}")
    success, data = try_write_to_path(temp_path)
    if success:
        print(f"Warning: Could not write to {file_path} or {home_path}")
        print(f"Writing to temporary location: {temp_path}")
        return data, temp_path
    
    raise PermissionError(f"Could not write to any location: {file_path}, {home_path}, or {temp_path}")
 
def save_experiment_params(project_root, model_name, capability, probe_path=None, c=None, offensive=None, learning_rate=None):
    """
    Save experiment parameters to a CSV file. Creates a new file if it doesn't exist.
    
    Parameters:
    -----------
    model_name : str
        Name of the model being used
    probe_path : str or Path, optional
        Path to the probe file
    c : float, optional
        Strength parameter
    offensive : bool, optional
        Whether the experiment is offensive (True) or defensive (False)
    learning_rate : float, optional
        Learning rate used in the experiment
    """
    # Create the data directory if it doesn't exist
    data_dir = project_root / 'datasets' / model_name
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Define the CSV file path
    csv_path = data_dir / 'experiment_params.csv'
    
    # Determine probe type from probe_path
    if probe_path:
        if "linear" in str(probe_path):
            probe_type = "linear"
        elif "mlp" in str(probe_path):
            probe_type = "mlp"
        elif "transformer" in str(probe_path):
            probe_type = "transformer"
        else:
            probe_type = "unknown"
    else:
        probe_type = None
    
    # Determine offensive/defensive status
    experiment_type = "offensive" if offensive else "defensive"
    
    # Create new experiment data
    new_data = {
        'model': [model_name],
        'capability': [capability],
        'probe_type': [probe_type],
        'probe_path': [probe_path],
        'offensive_defensive': [experiment_type],
        'c_strength': [c],
        'lr': [learning_rate]
    }
    
    new_df = pd.DataFrame(new_data)
    
    # If file exists, append to it; if not, create new file
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        updated_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        updated_df = new_df
    
    # Save to CSV
    updated_df.to_csv(csv_path, index=False)
    print(f"Saved experiment parameters to {csv_path}")


