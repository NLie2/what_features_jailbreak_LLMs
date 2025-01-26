import pandas as pd
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model
)
import wandb

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
    
    # Data preparation
    df1 = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/{model_name.replace('/','_')}/jailbreak_with_refusals.csv")
    df2 = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/ultrachat_10k_samples.csv")[:1000]
    # df = pd.concat([df1, df2])
    df = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/{model_name.replace('/','_')}/jailbreak_with_answer1.csv")
    df.golden_output = df.golden_output.astype(str)
    train_dataset_messages, eval_dataset_messages = prepare_dataset_to_messages(df)
    train_dataset_text = prepare_dataset_to_text(train_dataset_messages, tokenizer)
    eval_dataset_text = prepare_dataset_to_text(eval_dataset_messages, tokenizer)
    train_dataset_tokens = prepare_dataset_to_tokens(train_dataset_text, tokenizer)
    eval_dataset_tokens = prepare_dataset_to_tokens(eval_dataset_text, tokenizer)
    
    # Training setup
    training_args = TrainingArguments(
        output_dir=f"./finetuned_models/{model_name.replace('/', '_')}",
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
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_tokens,
        eval_dataset=eval_dataset_tokens,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(f"./finetuned_models/{model_name.replace('/', '_')}/final_model")
    
if __name__ == "__main__":
    main()