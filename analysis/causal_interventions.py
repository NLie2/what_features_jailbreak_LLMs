import torch
import torch.nn as nn
import torch.optim as optim
from gen_data.get_activations import get_res_layers_to_enumerate


## LINEAR INTERVENTION 
def get_response_linear(model, 
                 tokenizer,  
                 prompt,
                 probe,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
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
    
    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

    return generated_text



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
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
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
            feature_vector = output[0][:, :, :].requires_grad_().to(output[0].device)          
            print("SHAPE", feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 768]) True
            
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

              mlp_out = mlp(perturbed_vector)
              
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
    
    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # If outptut sequence repeats prompt, remove it 
    if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

    return generated_text                    
  

def insert_mlp_hook(model, 
                 mlp,
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
    def mlp_intervene(name, c, offensive):
        def hook(model, input, output):
          
          with torch.enable_grad():
                      
            # Forward pass
            feature_vector = output[0][:, :, :].requires_grad_().to(output[0].device)          
            # print(feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 768]) True
            
            # print("OLD MLP_out:", mlp(feature_vector)) ## check old predicted class
                      
                    
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
        hook_handle = model.model.layers[layer_index].register_forward_hook(mlp_intervene(layer_index, c, offensive))
        hooks.append(hook_handle)
  
  
### TRANSFORMER INTERVENTION 

def get_response_transformer(model, 
                           tokenizer,  
                           prompt,
                           probe,
                           layers_to_intervene=None,
                           max_new_tokens=15, 
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

                feature_vector = output[0][:, -1, :].requires_grad_().to(output[0].device)

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
                new_out = output_modified, output[1]
                print(new_out[0].shape)
                
                return new_out            
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
    
    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    # Remove prompt from output if present
    if generated_text.startswith(prompt):
        generated_text = generated_text[len(prompt):].strip()

    return generated_text
 
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

                feature_vector = output[0][:, -1, :].requires_grad_().to(output[0].device)

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
                
                
                output_modified[:, -1, :] = perturbed_vector
                new_out = output_modified, output[1]
                
                return new_out            
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks for each specified layer
    for layer_index in layers_to_intervene:
        if layer_index < 0 or layer_index >= len(layers_to_enum):
            raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
        hook_handle = model.model.layers[layer_index].register_forward_hook(transformer_intervene(layer_index, c, offensive))
        hooks.append(hook_handle)
  
 
 
 #### LEGACY 
  
  # def get_response(model, 
  #                tokenizer,  
  #                prompt,
  #                probe,
  #                layers_to_intervene=None,  # Allow multiple layers
  #                max_new_tokens=15, 
  #                intervention_strength=1,
  #                tokens="all", add_to="prompt"):
  #   """
  #   Generates a response from a model with a causal intervention applied to specific layers.

  #   Args:
  #       model: Pretrained language model.
  #       tokenizer: Tokenizer compatible with the model.
  #       prompt (str): The input text prompt.
  #       probe: The intervention object (e.g., a linear model with coefficients).
  #       layers_to_intervene (list of int): Indices of the model layers to apply the intervention.
  #       max_new_tokens (int): Maximum number of new tokens to generate.
  #       intervention_strength (float): Strength of the intervention to apply.

  #   Returns:
  #       str: The generated text after applying the intervention.
  #   """
  #   print("imported")
  #   if layers_to_intervene is None:
  #       layers_to_intervene = [17]  # Default to layer 17 if no layers specified

  #   device = next(model.parameters()).device
  #   input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
  #   len_prompt = input_ids.size(1)

  #   def linear_intervene(name):
  #       def hook(model, input, output):
  #           if add_to=="prompt":
  #               if output[0].shape[1] != 1:
  #                   if tokens=="last":
  #                       output[0][:, -1, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)
  #                   else:
  #                       output[0][:, :, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)

  #           else: 
  #               if tokens=="last":
  #                   # [1, 8, 3072]
  #                   output[0][:, -1, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)
  #                   # out = output[0][:, -1, :] +intervention_strength * torch.tensor(probe.coef_[0]).to(device)
  #               # elif tokens=="prompt":
  #                   # out = output[0][:, :len_prompt, :] + intervention_strength * torch.tensor(probe.coef_[0]).to(device)
  #               else:
  #                   output[0][:, :, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)
  #                   # out = output[0][:, :, :] + intervention_strength * torch.tensor(probe.coef_[0]).to(device) #third colon is token position
            
  #           return output
  #       return hook

  #   layers_to_enum = get_res_layers_to_enumerate(model)
  #   hooks = []

  #   # Register hooks for each specified layer
  #   for layer_index in layers_to_intervene:
  #       if layer_index < 0 or layer_index >= len(layers_to_enum):
  #           raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
  #       hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index))
  #       hooks.append(hook_handle)
    
  #   try:
  #       model.eval()
  #       with torch.no_grad():
  #           output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)

  #       # for i in range(len(200)):
  #           #autoregressively generate the thing 
  #   finally:
  #       # Ensure all hooks are removed after use
  #       for h in hooks:
  #           h.remove()
    
  #   generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
  #   if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

  #   return generated_text

  
  # def get_response_MLP_only_prompt(model, 
  #                tokenizer,  
  #                prompt,
  #                mlp,
  #                layers_to_intervene=None,  # Allow multiple layers
  #                max_new_tokens=15, 
  #                target = 0,
  #                intervention_strength=1):
  #   """
  #   Generates a response from a model with a causal intervention applied to specific layers.

  #   Args:
  #       model: Pretrained language model.
  #       tokenizer: Tokenizer compatible with the model.
  #       prompt (str): The input text prompt.
  #       probe: The intervention object (e.g., a linear model with coefficients).
  #       layers_to_intervene (list of int): Indices of the model layers to apply the intervention.
  #       max_new_tokens (int): Maximum number of new tokens to generate.
  #       intervention_strength (float): Strength of the intervention to apply.

  #   Returns:
  #       str: The generated text after applying the intervention.
  #   """
  #   print("imported")
  #   if layers_to_intervene is None:
  #       layers_to_intervene = [17]  # Default to layer 17 if no layers specified

  #   device = next(model.parameters()).device
  #   input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
  #   len_prompt = input_ids.size(1)
    
    
  #   # Define a loss function
  #   criterion = nn.MSELoss()
  #   # criterion = nn.BCEWithLogitsLoss()


  #   def linear_intervene(name, target):
  #       def hook(model, input, output):
  #         if output[0].shape[1] != 1:
  #           print("output shape", output[0].shape) # this sohuld not be 1 at second position
          
  #           with torch.enable_grad():
                        
  #             # Forward pass
  #             feature_vector = output[0][:, :, :].requires_grad_()          
  #             print(feature_vector.shape, feature_vector.requires_grad)  # Should print torch.Size([1, 768]) True
              
  #             ## check old predicted class
  #             pred = mlp(feature_vector)

  #             # Apply a sigmoid to convert it into a probability (between 0 and 1)
  #             probability = torch.sigmoid(pred)

  #             # Check which category it belongs to (class 0 or 1)
  #             predicted_class = (probability > 0.5).float()
  #             # print(predicted_class)

  #             print("OLD Predicted class:", predicted_class)
              
  #             ### 
              
                      
  #             # Define perturbation as a parameter or tensor that requires gradient
  #             perturbation = torch.zeros_like(feature_vector, requires_grad=True)
              
  #             printed_once = True

  #             for i in range(32):
  #               # Apply perturbation to the feature vector
  #               perturbed_vector = feature_vector + perturbation

  #               # Get output from the MLP
  #               mlp_out = mlp(perturbed_vector)
                
                
  #               target_out = torch.zeros_like(mlp_out)  # Default to zero, or choose an appropriate value

  #               # Target tensor (ensure its shape matches mlp_out's shape)
  #               # target = torch.zeros_like(mlp_out)  # Match the output size
          
  #               if target == 0:
  #                 target_out = torch.zeros_like(mlp_out) # Match the output size  
  #                 if printed_once: 
  #                   print("TARGET", target_out)
  #                   printed_once = False
                  
  #               if target == 1:
  #                 target_out = torch.ones_like(mlp_out) # Match the output size 
  #                 if printed_once: 
  #                   print("TARGET", target_out) 
  #                   printed_once = False

  #               # Calculate the loss
  #               loss = criterion(mlp_out, target_out)

  #               # print(loss, loss.requires_grad)  # Should print a scalar tensor and True

  #               # Backward pass to compute gradients w.r.t the perturbation
  #               loss.backward()

  #               # Access the gradient of the loss w.r.t. the perturbation
  #               grad_input = perturbation.grad

  #               # Define a learning rate
  #               learning_rate = 0.01

  #               # Update the perturbation in the direction of the negative gradient
  #               perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_()
                
  #             ## check new predicted class
  #             # pred = mlp(perturbation)
  #             pred = mlp(perturbed_vector)

  #             # Apply a sigmoid to convert it into a probability (between 0 and 1)
  #             probability = torch.sigmoid(pred)

  #             # Check which category it belongs to (class 0 or 1)
  #             predicted_class = (probability > 0.5).float()

  #             print("NEW Predicted class:", predicted_class.item())
              
  #             ### 
              
              
  #             new_out = perturbed_vector, output[1] # concat perturbed vector with the rest of the output

  #             # returnn perturbation
  #             return new_out
          
  #       return hook

  #   layers_to_enum = get_res_layers_to_enumerate(model)
  #   hooks = []

  #   # Register hooks for each specified layer
  #   for layer_index in layers_to_intervene:
  #       if layer_index < 0 or layer_index >= len(layers_to_enum):
  #           raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
  #       hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index, target))
  #       hooks.append(hook_handle)
    
  #   try:
  #       # model.eval()
  #       with torch.no_grad():
  #         output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)
            
  #   finally:
  #       # Ensure all hooks are removed after use
  #       for h in hooks:
  #           h.remove()
    
  #   generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
  #   # If outptut sequence repeats prompt, remove it 
  #   if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

  #   return generated_text
