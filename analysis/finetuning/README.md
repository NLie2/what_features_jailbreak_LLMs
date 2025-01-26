For high level understanding, go through finetune.ipynb.

[Warning]: The code has not been cleaned up yet.

However, in practice, I now
1. Run the original model on the training jailbreaks and construct a training dataset with ?.py
2. Train a LoRa with finetune_assistant.py (It is best practice to only train on the assistant message conditioned on the user message, rather than the full conversation)
    1. python finetune_assistant.py
3. Test both the original model and the LoRa. For finetuned models, add a specifier after the model directory name. e.g. 'meta-llama/Llama-3.2-3B-Instruct' -> 'meta-llama/Llama-3.2-3B-Instruct_all1'. Then add the specifier when calling eval_ASR.py and eval_MMLU.py. 'unfinetuned' is a specifier, where the base model is used.
    1. Test if the trained model still gives ok outputs by running finetune_chat_test.ipynb
    2. Calculate ASR with: python eval_ASR.py meta-llama/Llama-3.2-3B-Instruct unfinetuned
    3. Calculate MMLU with: python eval_MMLU.py meta-llama/Llama-3.2-3B-Instruct unfinetuned
