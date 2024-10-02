import torch
import torch.nn as nn
import torch.optim as optim
from gen_data.get_activations import get_res_layers_to_enumerate


def get_response(model, 
                 tokenizer,  
                 prompt,
                 probe,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
                 intervention_strength=1,
                 tokens="all", add_to="prompt"):
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
    print("imported")
    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = input_ids.size(1)

    def linear_intervene(name):
        def hook(model, input, output):
            if add_to=="prompt":
                if output[0].shape[1] != 1:
                    if tokens=="last":
                        output[0][:, -1, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)
                    else:
                        output[0][:, :, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)

            else: 
                if tokens=="last":
                    # [1, 8, 3072]
                    output[0][:, -1, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)
                    # out = output[0][:, -1, :] +intervention_strength * torch.tensor(probe.coef_[0]).to(device)
                # elif tokens=="prompt":
                    # out = output[0][:, :len_prompt, :] + intervention_strength * torch.tensor(probe.coef_[0]).to(device)
                else:
                    output[0][:, :, :] += intervention_strength * torch.tensor(probe.coef_[0]).to(device)
                    # out = output[0][:, :, :] + intervention_strength * torch.tensor(probe.coef_[0]).to(device) #third colon is token position
            
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

        # for i in range(len(200)):
            #autoregressively generate the thing 
    finally:
        # Ensure all hooks are removed after use
        for h in hooks:
            h.remove()
    
    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
    if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

    return generated_text

def get_response_MLP(model, 
                 tokenizer,  
                 prompt,
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
                 target = 0,
                 loss = "MSE",
                 towards_target = True,
                 nr_perturbations=32, 
                 learning_rate = 0.1):
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
              # if loss == "BCE_new":
              mlp_out = sigmoid(mlp(perturbed_vector))
              # ! Add this back in
              # else: 
              # mlp_out = mlp(perturbed_vector)
              
              
              # target_out = torch.zeros_like(mlp_out)  # Default to zero, or choose an appropriate value

              # Target tensor (ensure its shape matches mlp_out's shape)
              # target = torch.zeros_like(mlp_out)  # Match the output size
              if target == -1:
                # Create a tensor of -1's with the same shape as the output
                target_out = torch.full_like(mlp_out, -1)
                if printed_once: 
                    print("TARGET", target_out)
                    printed_once = False
                    
              if target == 0:
                target_out = torch.zeros_like(mlp_out) # Match the output size  
                if printed_once: 
                  print("TARGET", target_out)
                  printed_once = False
                
              if target == 1:
                target_out = torch.ones_like(mlp_out) # Match the output size 
                if printed_once: 
                  print("TARGET", target_out) 
                  printed_once = False
              
              else: 
                target_out = torch.full_like(mlp_out, target)
                if printed_once: 
                    print("TARGET", target_out)
                    printed_once = False                

              # Calculate the loss
              loss = criterion(mlp_out, target_out)

              # print(loss, loss.requires_grad)  # Should print a scalar tensor and True

              # Backward pass to compute gradients w.r.t the perturbation
              loss.backward()

              # Access the gradient of the loss w.r.t. the perturbation
              grad_input = perturbation.grad

              # Define a learning rate
              # learning_rate = 0.01

              
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

            print("NEW Predicted class:", predicted_class)
            
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
  
def get_response_MLP_only_prompt(model, 
                 tokenizer,  
                 prompt,
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
                 target = 0,
                 intervention_strength=1):
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
    print("imported")
    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = input_ids.size(1)
    
    
    # Define a loss function
    criterion = nn.MSELoss()
    # criterion = nn.BCEWithLogitsLoss()


    def linear_intervene(name, target):
        def hook(model, input, output):
          if output[0].shape[1] != 1:
            print("output shape", output[0].shape) # this sohuld not be 1 at second position
          
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

              for i in range(32):
                # Apply perturbation to the feature vector
                perturbed_vector = feature_vector + perturbation

                # Get output from the MLP
                mlp_out = mlp(perturbed_vector)
                
                
                target_out = torch.zeros_like(mlp_out)  # Default to zero, or choose an appropriate value

                # Target tensor (ensure its shape matches mlp_out's shape)
                # target = torch.zeros_like(mlp_out)  # Match the output size
          
                if target == 0:
                  target_out = torch.zeros_like(mlp_out) # Match the output size  
                  if printed_once: 
                    print("TARGET", target_out)
                    printed_once = False
                  
                if target == 1:
                  target_out = torch.ones_like(mlp_out) # Match the output size 
                  if printed_once: 
                    print("TARGET", target_out) 
                    printed_once = False

                # Calculate the loss
                loss = criterion(mlp_out, target_out)

                # print(loss, loss.requires_grad)  # Should print a scalar tensor and True

                # Backward pass to compute gradients w.r.t the perturbation
                loss.backward()

                # Access the gradient of the loss w.r.t. the perturbation
                grad_input = perturbation.grad

                # Define a learning rate
                learning_rate = 0.01

                # Update the perturbation in the direction of the negative gradient
                perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_()
                
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




def get_response_MLP_random(model, 
                 tokenizer,  
                 prompt,
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15):
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
    print("random")
    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = input_ids.size(1)
    
    
    # Define a loss function
    # criterion = nn.BCEWithLogitsLoss()


    def linear_intervene(name):
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
            # get random vector from torch
            
            perturbation = torch.randn_like(feature_vector, requires_grad=True)
            
            perturbed_vector = feature_vector + perturbation
              
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
        hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index))
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
  
def get_response_MLP_logit(model, 
                 tokenizer,  
                 prompt,
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
                 target = 0,
                 towards_target = True,
                 nr_perturbations=32):
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
    print(f"logit loss")
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

              # Define a learning rate
              learning_rate = 0.01

              
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


# def get_response_linear(model, 
#                  tokenizer,  
#                  prompt,
#                  linear_probe,
#                  layers_to_intervene=[17],  # Allow multiple layers
#                  max_new_tokens=15, 
#                  target = 0,
#                  intervention_strength=1):
#     """
#     Generates a response from a model with a causal intervention applied to specific layers.

#     Args:
#         model: Pretrained language model.
#         tokenizer: Tokenizer compatible with the model.
#         prompt (str): The input text prompt.
#         probe: The intervention object (e.g., a linear model with coefficients).
#         layers_to_intervene (list of int): Indices of the model layers to apply the intervention.
#         max_new_tokens (int): Maximum number of new tokens to generate.
#         intervention_strength (float): Strength of the intervention to apply.

#     Returns:
#         str: The generated text after applying the intervention.
#     """
#     print("linear")

#     device = next(model.parameters()).device
#     input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
#     len_prompt = input_ids.size(1)
    
    
#     # Define a loss function
#     # criterion = nn.MSELoss()
#     # criterion = nn.BCEWithLogitsLoss()

#     def linear_intervene(name, target):
#         def hook(model, input, output):
          
#           W = torch.from_numpy(linear_probe.coef_).float()
#           b = torch.from_numpy(linear_probe.intercept_).float()
          
#           x = output[0][:, :, :].requires_grad_()

#           with torch.enable_grad():
            
#             print(x.requires_grad)

#             predict = linear_probe.predict(x.detach().numpy())
#             print(predict)


#             y_pred = torch.sigmoid(torch.matmul(W.view(-1,1), x) + b)

#             # Initialize perturbation 
#             perturbation = torch.ones_like(x, requires_grad=True)
#             print(perturbation.requires_grad)


#             # Define optimizer for x
#             # optimizer = torch.optim.SGD([x], lr=0.01)

#             for i in range(32):
#                 # optimizer.zero_grad()
#                 perturbed_vector = x + perturbation
                
#                 # Logistic function output
#                 y_pred = torch.sigmoid(torch.matmul(W.view(-1,1), perturbation) + b).requires_grad_()
#                 y_target = torch.zeros_like(y_pred).requires_grad_()
#                 print(y_target.requires_grad, y_pred.requires_grad)

#                 # y_pred = model.predict(x)
#                 # print(y_pred)
                
#                 # Binary cross-entropy loss
#                 loss = F.binary_cross_entropy(y_pred, y_target)
#                 loss.backward()
#                 print(loss.requires_grad)
                
#                 # Compute gradients and update x
#                 # loss.backward()
#                 grad_input = perturbation.grad
#                 print(grad_input)
#                 # x = x + W
#                           # Define a learning rate
#                 learning_rate = 1

#                           # Update the perturbation in the direction of the negative gradient
#                 perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_()
                
#             predict = linear_probe.predict(perturbed_vector.detach().numpy())
#             print(predict)
            
#             new_out = perturbed_vector, output[1] # concat perturbed vector with the rest of the output

#             # returnn perturbation
#             return new_out
          
#         return hook

#     layers_to_enum = get_res_layers_to_enumerate(model)
#     hooks = []

#     # Register hooks for each specified layer
#     for layer_index in layers_to_intervene:
#         if layer_index < 0 or layer_index >= len(layers_to_enum):
#             raise ValueError(f"Layer {layer_index} is out of bounds for the model.")
#         hook_handle = model.model.layers[layer_index].register_forward_hook(linear_intervene(layer_index, target))
#         hooks.append(hook_handle)
    
#     try:
#         # model.eval()
#         with torch.no_grad():
#           output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)
            
#     finally:
#         # Ensure all hooks are removed after use
#         for h in hooks:
#             h.remove()
    
#     generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    
#     # If outptut sequence repeats prompt, remove it 
#     if generated_text.startswith(prompt): generated_text = generated_text[len(prompt):].strip()

#     return generated_text
  
####
# ! attempt linear probe with gradient
# import torch.nn.functional as F

# W = torch.from_numpy(model.coef_).float()
# b = torch.from_numpy(model.intercept_).float()

# with torch.enable_grad():
  
#   x= torch.tensor(df[activation_column][0][0][0]).requires_grad_()
#   print(x.requires_grad)

#   predict = model.predict(x.detach().numpy())
#   print(predict)


#   y_pred = torch.sigmoid(torch.matmul(W.view(-1,1), x) + b)

#   # Initialize perturbation 
#   perturbation = torch.ones_like(x, requires_grad=True)
#   print(perturbation.requires_grad)


#   # Define optimizer for x
#   # optimizer = torch.optim.SGD([x], lr=0.01)

#   for i in range(32):
#       perturbed_vector = x + perturbation
#       # optimizer.zero_grad()
      
#       # Logistic function output
#       y_pred = torch.sigmoid(torch.matmul(W.view(-1,1), perturbation) + b).requires_grad_()
#       y_target = torch.zeros_like(y_pred).requires_grad_()
#       print(y_target.requires_grad, y_pred.requires_grad)

#       # y_pred = model.predict(x)
#       # print(y_pred)
      
#       # Binary cross-entropy loss
#       loss = F.binary_cross_entropy(y_pred, y_target)
#       loss.backward()
#       print(loss.requires_grad)
      
#       # Compute gradients and update x
#       # loss.backward()
#       grad_input = perturbation.grad
#       print(grad_input)
#       # x = x + W
#                 # Define a learning rate
#       learning_rate = 1

#                 # Update the perturbation in the direction of the negative gradient
#       perturbation = (perturbation - learning_rate * grad_input).clone().detach().requires_grad_()
      
#   predict = model.predict(perturbed_vector.detach().numpy())
#   print(predict)





####





# def get_response(model, 
#                  tokenizer,  
#                  prompt,
#                  probe,
#                  layer_to_intervene = 17, 
#                  max_new_tokens = 15, 
#                  multiply_magnitude = 1):

#     device = next(model.parameters()).device

#     #model, tokenizer = models.get_model_from_name(model_name)
#     input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
#     len_prompt = len(input_ids)

#     def linear_intervene(name):
#         def hook(model, input, output):
#             print(output[0].shape)
#             print("len", len(probe.coef_[0])) # use normalized
#             print(probe.coef_[0])
#             output[0][:, :, :] += multiply_magnitude*probe.coef_[0]
#             return output
#         return hook

#     layers_to_enum = get_res_layers_to_enumerate(model)
#     hooks = []

#     # Register hooks
#     # for i, layer in enumerate(layers_to_enum):
#     hook_handle = model.model.layers[layer_to_intervene].register_forward_hook(linear_intervene(layer_to_intervene))
#     hooks.append(hook_handle)

#     # Process the dataset
#     model.eval()
#     with torch.no_grad():
#         inputs = tokenizer(prompt, return_tensors="pt").to(device)
#         # _ = model(**inputs)
#         output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)

#     for h in hooks:
#         h.remove()

#     generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
#     return generated_text