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

def prepare_dataset(df, tokenizer):
    # Convert to the format expected by apply_chat_template
    conversations = []
    for _, row in df.iterrows():
        conversations.append([
            {"role": "user", "content": row['golden_input']},
            {"role": "assistant", "content": row['golden_output']}
        ])
    
    # Create dataset with conversations
    dataset = Dataset.from_pandas(pd.DataFrame({"conversations": conversations}))
    
    def apply_template(example):
        chat = example["conversations"]
        return {"text": tokenizer.apply_chat_template(chat, tokenize=False)}
    
    # Apply the official chat template
    dataset = dataset.map(apply_template, remove_columns=dataset.column_names)
    
    # Split the dataset into train and evaluation sets
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    return split_dataset['train'], split_dataset['test']

def tokenize_function(examples, tokenizer, max_length=512):
    return tokenizer(
        examples['text'],
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors='pt'
    )

def prepare_tokenized_dataset(dataset, tokenizer):
    return dataset.map(
        lambda x: tokenize_function(x, tokenizer),
        batched=True,
        remove_columns=dataset.column_names,
        batch_size=8
    ).with_format("torch")

def get_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )
    
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
    
    # Data preparation
    df = pd.read_csv(f"/home/weisser/what_features_jailbreak_LLMs/datasets/finetune/{model_name.replace('/','_')}/jailbreak_refusals.csv")
    train_dataset, eval_dataset = prepare_dataset(df, tokenizer)
    tokenized_train_dataset = prepare_tokenized_dataset(train_dataset, tokenizer)
    tokenized_eval_dataset = prepare_tokenized_dataset(eval_dataset, tokenizer)
    
    # Model preparation
    model = setup_lora_model(model)
    
    # Training setup
    training_args = TrainingArguments(
        output_dir=f"./finetuned_models/{model_name.replace('/', '_')}",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=100,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        report_to="wandb"
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    
    # Start training
    trainer.train()
    
    # Save the final model
    trainer.save_model(f"./finetuned_models/{model_name.replace('/', '_')}/final_model")
    
if __name__ == "__main__":
    main()