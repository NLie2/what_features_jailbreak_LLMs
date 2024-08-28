import torch

from gen_data.get_activations import get_res_layers_to_enumerate


def get_response(model, 
                 tokenizer,  
                 prompt,
                 probe,
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