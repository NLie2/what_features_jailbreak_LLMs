For high level understanding, go through finetune.ipynb.

[Warning]: The code has not been cleaned up yet.

However, in practice, I now
1. Run the original model on the training jailbreaks and construct a training dataset with ?.py
2. Train a LoRa with finetune.py
3. Test both the original model and the LoRa with eval_finetuned.py
    1. Test if the trained model still gives ok outputs by running finetune_chat_test.ipynb
    2. ASR on holdout dataset
    3. Calculate MMLU