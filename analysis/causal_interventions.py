import torch

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
    return generated_text

def get_response_MLP(model, 
                 tokenizer,  
                 prompt,
                 mlp,
                 layers_to_intervene=None,  # Allow multiple layers
                 max_new_tokens=15, 
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
    if layers_to_intervene is None:
        layers_to_intervene = [17]  # Default to layer 17 if no layers specified

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = input_ids.size(1)
    
    
    # Define a loss function
    criterion = nn.MSELoss()
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
            print(predicted_class)

            print("OLD Predicted class:", predicted_class)
            
            ### 
            
                    
            # Define perturbation as a parameter or tensor that requires gradient
            perturbation = torch.zeros_like(feature_vector, requires_grad=True)

            for i in range(32):
              # Apply perturbation to the feature vector
              perturbed_vector = feature_vector + perturbation

              # Get output from the MLP
              mlp_out = mlp(perturbed_vector)
              

              # Target tensor (ensure its shape matches mlp_out's shape)
              # target = torch.zeros_like(mlp_out)  # Match the output size
              target = torch.zeros_like(mlp_out)  # Match the output size          


              # Calculate the loss
              loss = criterion(mlp_out, target)

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
            pred = mlp(perturbation)

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
    return generated_text

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