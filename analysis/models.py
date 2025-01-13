## CONFIG
models = {
    "gemma-7b-it": {
        "num_layers": 28,
        "rating_column": "rating",
        "input_size": 3072,
        "hf-identifier": "google/gemma-7b-it",
    },
    "Llama-3.1-8B-Instruct": {
        "hf-identifier": "meta-llama/Llama-3.1-8B-Instruct",
        "num_layers": 32,
        "rating_column": "rating_output_text",
        "input_size": 4096,
        
    },
    
    "Llama-3.2-3B-Instruct": {
      "hf-identifier": "meta-llama/Llama-3.2-3B-Instruct",
      "num_layers": 28,
      "rating_column": "rating_output_text",
      "input_size": 3072,
      }, 
    "Ministral-8B-Instruct": {
        "hf-identifier": "mistralai/Ministral-8B-Instruct-2410",
        "num_layers": 36,
        "rating_column": "rating_output_text",
        "input_size": 4096,
    }
}



# Determine the model name based on the file path
def get_model_details(model):
    for model_name in models.keys():
        if model_name in str(model):
            return model_name, models[model_name]["num_layers"], models[model_name]["rating_column"], models[model_name]["hf-identifier"], models[model_name]["input_size"]
    print(f"Model {model} not found in models")
    return "unknown"  # Default if no match is found
