import pandas as pd
import sys
from pathlib import Path
import os
from tqdm import tqdm
from utils_finetune import load_and_generate_responses_batched, grade_with_HF_batched

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
df_val = pd.read_pickle("datasets/finetune/validation/validation_data.pkl")

fname_stage1 = f"{save_dir}/jailbreak_prompts_val.csv"
if os.path.exists(fname_stage1):
    print(f"\n\nStage 1: Loading existing results from {fname_stage1}...\n")
    df_val = pd.read_csv(fname_stage1)
else:
    print("\n\nStage 1: Generating responses for jailbreak prompts.\n")
    df_val[f'output_text'] = load_and_generate_responses_batched(model_name, lora_weights_path, df_val['jailbreak_prompt_text'], batch_size=16, max_new_tokens=1024)
    print(f"\n\nStage 1: Saving results to {fname_stage1}...\n")
    df_val.to_csv(fname_stage1, index=False)

fname_stage2 = f"{save_dir}/jailbreak_prompts_val_rated.csv"
if os.path.exists(fname_stage2):
    print(f"\n\nStage 2: Loading existing results from {fname_stage2}...\n")
    df_val = pd.read_csv(fname_stage2)
    print(f"ASR: {(df_val[f'rating_output_text']=='Yes').mean()} = {(df_val[f'rating_output_text']=='Yes').sum()}/{len(df_val)}")
else:
    print("\n\nStage 2: Grading responses for jailbreak prompts.\n")
    # Important: Given that the jailbreak prompt is just an original prompt and the other column doesnt exist, we use jailbreak_prompt_text
    # If original_prompt_text existed, we should use that
    df_val[f'rating_output_text'] = grade_with_HF_batched(df_val['jailbreak_prompt_text'], df_val[f'output_text'], batch_size=4)
    print(f"ASR : {(df_val[f'rating_output_text']=='Yes').mean()} = {(df_val[f'rating_output_text']=='Yes').sum()}/{len(df_val)}")
    print(f"\n\nStage 2: Saving results to {fname_stage2}.\n")
    df_val.to_csv(fname_stage2, index=False)
    # Create a summary DataFrame with ASR metrics

    summary_df = pd.DataFrame({
        'ASR_mean': [(df_val['rating_output_text']=='Yes').mean()],
        'total_yes': [(df_val['rating_output_text']=='Yes').sum()],
        'total_samples': [len(df_val)]
    })
    
    # Save summary to CSV
    summary_fname = f"{save_dir}/ASR.csv"
    print(f"\n\nSaving ASR summary to {summary_fname}.\n")
    summary_df.to_csv(summary_fname, index=False)