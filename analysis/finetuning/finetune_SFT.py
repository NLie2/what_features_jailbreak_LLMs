from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import Dataset
import pandas as pd
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import wandb
import torch

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
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
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
    return get_peft_model(model, lora_config)

def main():
    # Initialize wandb for experiment tracking
    wandb.init(project="llama2-lora-finetuning")

    # Model setup
    model_name = "meta-llama/Llama-3.2-3B-Instruct"
    model, tokenizer = get_model_and_tokenizer(model_name)
    model = setup_lora_model(model)

    df1 = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/{model_name.replace('/','_')}/jailbreak_with_refusals.csv")
    df2 = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/ultrachat_10k_samples.csv")[:1000]
    # df = pd.concat([df1, df2])
    df = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/{model_name.replace('/','_')}/jailbreak_with_answer1.csv")
    df.golden_output = df.golden_output.astype(str)
    train_dataset_messages, eval_dataset_messages = prepare_dataset_to_messages(df)
    # train_dataset = train_dataset.select([1])


    # proposed_response_template = "<|im_start|>assistant\n"
    proposed_response_template = "<|start_header_id|>assistant<|end_header_id|>" #\\n\\n"
    assert proposed_response_template in tokenizer.chat_template, f"The response template is not in the tokenizer's chat template. \n{proposed_response_template} \nnot in \n{tokenizer.chat_template}"

    # Create the data collator
    data_collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
        response_template=proposed_response_template
    )

    training_args = SFTConfig(
        output_dir=f"./finetuned_models/{model_name.replace('/', '_')}_SFT",
        num_train_epochs=1,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=50,
        learning_rate=5e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb",
        max_seq_length=2048,
    )

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
    trainer.save_model(f"./finetuned_models/{model_name.replace('/', '_')}_SFT/final_model")

if __name__ == "__main__":
    main()