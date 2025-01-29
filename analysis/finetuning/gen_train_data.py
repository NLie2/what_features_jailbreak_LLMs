import pandas as pd
import sys
from pathlib import Path
import os
from tqdm import tqdm
from utils_finetune import load_and_generate_responses_batched, grade_with_HF_batched
import numpy as np
tqdm.pandas()

# model_name = "meta-llama/Llama-3.2-3B-Instruct"
# model_name = "google/gemma-7b-it"
if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    assert False, "Please provide a model name as an argument"
specifier = "unfinetuned"

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
df_train = pd.read_csv("datasets/finetune/jailbreak_prompts.csv")

fname_stage1 = f"{save_dir}/jailbreak_prompts.csv"
if os.path.exists(fname_stage1):
    print(f"\n\nStage 1: Loading existing results from {fname_stage1}...\n")
    df_train = pd.read_csv(fname_stage1)
else:
    print("\n\nStage 1: Generating responses for jailbreak prompts.\n")
    df_train[f'output_text'] = load_and_generate_responses_batched(model_name, lora_weights_path, df_train['jailbreak_prompt_text'], batch_size=32, max_new_tokens=1024)
    print(f"\n\nStage 1: Saving results to {fname_stage1}...\n")
    df_train.to_csv(fname_stage1, index=False)

fname_stage2 = f"{save_dir}/jailbreak_prompts_rated.csv"
if os.path.exists(fname_stage2):
    print(f"\n\nStage 2: Loading existing results from {fname_stage2}...\n")
    df_train = pd.read_csv(fname_stage2)
    print(f"ASR: {(df_train[f'rating_output_text']=='Yes').mean()} = {(df_train[f'rating_output_text']=='Yes').sum()}/{len(df_train)}")
else:
    print("\n\nStage 2: Grading responses for jailbreak prompts.\n")
    # Important: We use the original prompt text, not the jailbreak prompt text
    df_train[f'rating_output_text'] = grade_with_HF_batched(df_train['original_prompt_text'], df_train[f'output_text'], batch_size=4)
    print(f"ASR : {(df_train[f'rating_output_text']=='Yes').mean()} = {(df_train[f'rating_output_text']=='Yes').sum()}/{len(df_train)}")
    print(f"\n\nStage 2: Saving results to {fname_stage2}.\n")
    df_train.to_csv(fname_stage2, index=False)
    # Create a summary DataFrame with ASR metrics

    summary_df = pd.DataFrame({
        'ASR_mean': [(df_train['rating_output_text']=='Yes').mean()],
        'total_yes': [(df_train['rating_output_text']=='Yes').sum()],
        'total_samples': [len(df_train)]
    })
    
    # Save summary to CSV
    summary_fname = f"{save_dir}/ASR.csv"
    print(f"\n\nSaving ASR summary to {summary_fname}.\n")
    summary_df.to_csv(summary_fname, index=False)


fname_stage3 = f"datasets/finetune/{model_name.replace('/', '_')}/jailbreak_with_refusals.csv"
KEEP_ORIGINAL_REFUSALS = True

refusal_strings = [
        "I apologize, but I cannot and will not provide information about harmful or dangerous activities.",
        "I cannot assist with activities that could harm others or break laws.",
        "That request involves potentially dangerous or unethical behavior that I won't help with.",
        "I must decline as that would involve helping with harmful or illegal activities.",
        "I don't engage with requests that could lead to harm or illegal behavior.",
        "I aim to be helpful while avoiding harm. I cannot assist with that request.",
        "That kind of activity could hurt people. I won't provide information about it.",
        "I have to decline as that would involve assisting with dangerous or unethical behavior.",
        "I cannot help with activities that risk harming others or breaking laws.",
        "My purpose is to help people, not cause harm. I cannot assist with that request."
    ]

if KEEP_ORIGINAL_REFUSALS:
    df_to_refuse = df_train.copy()
    df_to_refuse['golden_output'] = np.where(
        df_to_refuse[f'rating_output_text']=='Yes',
        np.random.choice(refusal_strings, size=len(df_to_refuse)),
        df_to_refuse[f'output_text']
    )
    df_to_refuse['golden_input'] = df_to_refuse['jailbreak_prompt_text']
    df_to_refuse.to_csv(fname_stage3, index=False)
else:
    df_to_refuse = df_train[df_train[f'rating_output_text']=='Yes'].copy()
    df_to_refuse['golden_output'] = np.random.choice(refusal_strings, size=len(df_to_refuse))
    df_to_refuse['golden_input'] = df_to_refuse['jailbreak_prompt_text']
    df_to_refuse.to_csv(fname_stage3, index=False)
