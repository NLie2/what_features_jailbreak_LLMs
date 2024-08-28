import torch

from gen_data.get_activations import get_res_layers_to_enumerate

def get_response(model, 
                 tokenizer,  
                 prompt,
                 probe,
                 layer_to_intervene = 17, 
                 max_new_tokens = 15):

    device = next(model.parameters()).device

    #model, tokenizer = models.get_model_from_name(model_name)
    input_ids = tokenizer.encode(prompt, return_tensors='pt', truncation=True).to(device)
    len_prompt = len(input_ids)

    def linear_intervene(name):
        def hook(model, input, output):
            print(output[0].shape)
            print("len", len(probe.coef_[0])) # use normalized
            print(probe.coef_[0])
            output[0][:, :, :] += probe.coef_[0]
            return output
        return hook

    layers_to_enum = get_res_layers_to_enumerate(model)
    hooks = []

    # Register hooks
    # for i, layer in enumerate(layers_to_enum):
    hook_handle = model.model.layers[layer_to_intervene].register_forward_hook(linear_intervene(layer_to_intervene))
    hooks.append(hook_handle)

    # Process the dataset
    model.eval()
    with torch.no_grad():
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        # _ = model(**inputs)
        output_sequence = model.generate(input_ids, num_return_sequences=1, max_new_tokens=max_new_tokens, do_sample=False)

    for h in hooks:
        h.remove()

    generated_text = tokenizer.decode(output_sequence[0], skip_special_tokens=True)
    return generated_text