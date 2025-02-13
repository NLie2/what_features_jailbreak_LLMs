import json
import os

import pandas as pd
import torch
import wandb
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer

from datasets import Dataset, load_dataset


def prepare_dataset_to_messages(df):
    # Convert to the format expected by apply_chat_template
    messages = []
    for _, row in df.iterrows():
        messages.append([
            {"role": "user", "content": row['golden_input']},
            {"role": "assistant", "content": row['golden_output']}
        ])
    
    # Create dataset with conversations
    dataset = Dataset.from_pandas(pd.DataFrame({"messages": messages}))
    split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
    return split_dataset['train'], split_dataset['test']

def prepare_dataset_to_text(dataset, tokenizer):
    def apply_template(example):
        chat = example["messages"]
        return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
    # Apply the official chat template
    dataset = dataset.map(apply_template, remove_columns=dataset.column_names)
    return dataset

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

def prepare_dataset_to_tokens(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=8
    ).with_format("torch")

def get_model_and_tokenizer(model_id):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # need to set the pad token, otherwise, training will fail

    ADD_NEW_PAD_TOKEN = False
    if ADD_NEW_PAD_TOKEN:
        # if we do this, the trained model will have a different vocab size than the base model, which is not nice
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id
    else:
        # tokenizer.pad_token = tokenizer.eos_token # if this is set, the trained model will not know when to stop generating
        tokenizer.pad_token ="<|finetune_right_pad_id|>" # 128004
        # tokenizer.pad_token = tokenizer.unk_token
    
    return model, tokenizer

def setup_lora_model(model):
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = prepare_model_for_kbit_training(model)
    # Add this line to ensure parameters require gradients
    model.enable_input_require_grads()
    return get_peft_model(model, lora_config), lora_config

def save_training_args(training_args, lora_config):
    # Save training arguments to file
    args_dict = training_args.to_dict()
    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "training_args.json"), "w") as f:
        json.dump(args_dict, f, indent=2)

    with open(os.path.join(output_dir, "lora_config.txt"), "w") as f:
        f.write(lora_config.__repr__())

def proposed_response_template(model_name):
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        return "<|start_header_id|>assistant<|end_header_id|>"
    elif model_name == "google/gemma-7b-it":
        return "<start_of_turn>model"
    elif model_name == "mistralai/Ministral-8B-Instruct-2410":
        return '[/INST]'
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        return "<|start_header_id|>assistant<|end_header_id|>"
    else:
        raise ValueError(f"Model {model_name} not supported")

def main():
    # Model setup
    model_name = "google/gemma-7b-it" #"meta-llama/Llama-3.2-3B-Instruct"
    specifier = 'baseline'
    model, tokenizer = get_model_and_tokenizer(model_name)
    model, lora_config = setup_lora_model(model)
    
    # Initialize wandb for experiment tracking
    wandb.init(project="what_features_jailbreak_LLMs", name=f"{model_name.replace('/', '_')}_{specifier}")

    df1 = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/{model_name.replace('/','_')}/jailbreak_with_refusals.csv")
    df2 = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/ultrachat_10k_samples.csv")[:1000]
    df = pd.concat([df1, df2])
    # df = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/{model_name.replace('/','_')}/jailbreak_with_answer1.csv")
    df.golden_output = df.golden_output.astype(str)
    train_dataset_messages, eval_dataset_messages = prepare_dataset_to_messages(df)
    # train_dataset = train_dataset.select([1])


    # proposed_response_template = "<|im_start|>assistant\n"
    # proposed_response_template = "<|start_header_id|>assistant<|end_header_id|>" #\\n\\n"
    proposed_response_template = proposed_response_template(model_name)
    assert proposed_response_template in tokenizer.chat_template, f"The response template is not in the tokenizer's chat template. \n{proposed_response_template} \nnot in \n{tokenizer.chat_template}"

    # Create the data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        response_template=proposed_response_template
    )

    training_args = SFTConfig(
        output_dir=f"./finetuned_models/{model_name.replace('/', '_')}_{specifier}",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=5e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        report_to="wandb",
        max_seq_length=2048,
    )

    save_training_args(training_args, lora_config)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_messages,
        eval_dataset=eval_dataset_messages,
        # formatting_func=formatting_prompts_func,
        data_collator=data_collator,
    )

    trainer.train()
    # Save the final model
    trainer.save_model(f"./finetuned_models/{model_name.replace('/', '_')}_{specifier}/final_model")

if __name__ == "__main__":
    main()