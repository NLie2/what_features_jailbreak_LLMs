For high level understanding, go through finetune.ipynb.

[Warning]: The code has not been cleaned up yet.

However, in practice, I now
1. Evaluate the base model. 
    1. Test if the trained model gives ok outputs by interacting with it in finetune_chat_test.ipynb
    2. Calculate ASR with: python eval_ASR.py meta-llama/Llama-3.2-3B-Instruct unfinetuned
    3. Calculate MMLU with: python eval_MMLU.py meta-llama/Llama-3.2-3B-Instruct unfinetuned
2. Generate training data to finetune the model. Run: python gen_train_data.py meta-llama/Llama-3.2-3B-Instruct
3. Train a LoRa with finetune_assistant.py (It is best practice to only train on the assistant message conditioned on the user message, rather than the full conversation). Run: python finetune_assistant.py; Remember to set a specifier, which is a name to remember the model by and will be part of the model directory name (e.g. 'all1')
4. Test the finetuned model.
    1. Test if the trained model still gives ok outputs by interacting with it in eval_chat_test.ipynb
    2. Calculate ASR with: python eval_ASR.py meta-llama/Llama-3.2-3B-Instruct all1
    3. Calculate MMLU with: python eval_MMLU.py meta-llama/Llama-3.2-3B-Instruct all1
