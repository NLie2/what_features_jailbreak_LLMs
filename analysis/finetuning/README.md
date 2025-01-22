For high level understanding, go through finetune.ipynb.

However, in practice, I now
1. Test the unfinetuned model with eval_finetuned.py
2. Generate training data with the above results using finetune.ipynb and a bit of manual work
3. Finetune the model with the training data with finetune.py
4. Test the finetuned model with eval_finetuned.py