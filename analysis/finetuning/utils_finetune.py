import pandas as pd
import sys
from pathlib import Path
import os
import numpy as np
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM

def load_model_with_lora(
    base_model_id,
    lora_weights_path=None,
):
    # Load the base model
    model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    tokenizer.pad_token = tokenizer.eos_token
    
    if lora_weights_path is not None:
        # Load and apply LoRA weights
        model = PeftModel.from_pretrained(
            model,
            lora_weights_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    return model, tokenizer

def generate_response(model, tokenizer, prompt, max_new_tokens=512):
    # Format the prompt using the chat template
    messages = [{"role": "user", "content": prompt}]
    formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    
    # Tokenize
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    
    # Get the input length in tokens
    input_length = inputs.input_ids.shape[1]
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode only the new tokens
    response = tokenizer.decode(outputs[0][input_length:], skip_special_tokens=True)
    response = response.strip()
    if response.startswith("assistant"):
        response = response[len("assistant"):].strip()
    return response