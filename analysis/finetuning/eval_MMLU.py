import pandas as pd
import sys
from pathlib import Path
import os
from tqdm import tqdm
from utils_finetune import eval_MMLU
tqdm.pandas()

# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "google/gemma-7b-it"
if len(sys.argv) > 2:
    model_name = sys.argv[1]
    specifier = sys.argv[2]
else:
    assert False, "Please provide a model name as an argument"

print(f"\n\nModel name: {model_name}, specifier: {specifier}\n\n")
model_path = f"/home/weisser/what_features_jailbreak_LLMs/analysis/finetuning/finetuned_models/{model_name.replace('/', '_')}_{specifier}"
if specifier=="unfinetuned":
    print("Using base model")
    os.makedirs(model_path, exist_ok=True)
    lora_weights_path = None
else:
    lora_weights_path = model_path+"/final_model"
    
project_root = Path.cwd().parent.parent
os.chdir(project_root)
save_dir = model_path # f"datasets/finetune/{model_name.replace('/', '_')}"
os.makedirs(save_dir, exist_ok=True)

fname = f"{save_dir}/MMLU.csv"
if os.path.exists(fname):
    print(f"\n\nStage 3: Loading existing results from {fname}...\n")
    summary_df = pd.read_csv(fname)
    print(f"MMLU: {summary_df['MMLU_acc'].iloc[0]}")
else:
    print("\n\nStage 3: Evaluating MMLU for jailbreak prompts...\n")
    mmlu_results = eval_MMLU(model_name, lora_weights_path)['results']['mmlu']['acc,none']
    print(f"MMLU: {mmlu_results}")
    print(f"\n\nSaving results to {fname}...\n")
    summary_df = pd.DataFrame({
        'MMLU_acc': [mmlu_results],
    })
    summary_df.to_csv(fname, index=False)