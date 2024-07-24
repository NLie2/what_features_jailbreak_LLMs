'''
John James dataset creation
INPUTS: model, tokenizer, size, get_activation_function(model, tokenizer, prompt)


RETURNS: dataframe in the form:
: prompt :    label :  activations  : label_name
: string : int (1/0) : torch.tensor : "john_james" 
: string : int (1/0) : torch.tensor : "john_james" 
: string : int (1/0) : torch.tensor : "john_james" 
...

'''

import torch as t
from datasets import load_dataset
import random


def john_james(model, tokenizer, size, get_activation_function, *args, **kwargs):
    dataset = []
    for i in range(size):
        for label, name in enumerate(["john", "james"]):
            random_repeats = random.randint(1, 10)
            text = f"{name} " * random_repeats       

            dataset.append({"sentence": text, "label": label})
    random.shuffle(dataset)

    data = []
    for item in dataset:
        prompt = item["sentence"]
        if args or kwargs:
            activations = get_activation_function(model, tokenizer, item["sentence"], *args, **kwargs)
        else:
            activations = get_activation_function(model, tokenizer, item["sentence"])
        label = t.tensor([item["label"]])
    
        data.append([prompt, label[0], activations['activations'], "john_james"])

    return data

if __name__ == "__main__":
    print("hello world!")
    # t.save(train_data, "gpt-2/train_john_james.pt")
    # t.save(test_data, "gpt-2/test_john_james.pt")