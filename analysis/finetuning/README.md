For high level understanding, go through finetune.ipynb.

However, in practice, I now
1. Test the unfinetuned model with eval_finetuned.py
2. Generate training data with the above results using finetune.ipynb and a bit of manual work to make sure I read from and save to the correct file locations.
3. Finetune the model with the training data with finetune.py. Here I have to do some manual hyperparameter tuning.
4. Test the finetuned model with eval_finetuned.py