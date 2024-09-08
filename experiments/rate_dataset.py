#!/usr/bin/env python
# coding: utf-8

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import sys

# Function to process the dataframe
def process_dataframe(df, model, tokenizer):
    dictionary = {}
    # loop through every row where jailbreak_prompt_name is original_prompt
    for index, row in df.iterrows():
        if row['jailbreak_prompt_name'] == 'original_prompt':
            prompt_id = row['prompt_name']
            prompt_text = row['jailbreak_prompt_text']
            dictionary[prompt_id] = prompt_text

    # now loop through every row, and add a column called "original_prompt_text" to the dataframe
    for index, row in df.iterrows():
        prompt_id = row['prompt_name']
        df.loc[index, 'original_prompt_text'] = dictionary.get(prompt_id, None)
    
    return df

def main(input_path, output_path):
    sys.path.append("/data/nathalie_maria_kirch/ERA_Fellowship")
    from gen_data.harmBench_autograder import grade_with_HF


    # Load the dataframe from the input path
    df = pd.read_pickle(input_path)
    
    # Load the HarmBench model
    model_name = "cais/HarmBench-Llama-2-13b-cls"
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, truncation_side="left", padding_side="left")

    # Process the dataframe
    if "wei" in input_path: 
      df = process_dataframe(df, model, tokenizer)
    elif "allen" in input_path:
      df = df.rename(columns={"forbidden_prompt": "original_prompt_text", "response": "output_text"})


    # Apply the grading function and add the ratings column
    df["rating"] = df.apply(lambda row: grade_with_HF(row['original_prompt_text'], [row['output_text']], model, tokenizer)[0], axis=1)

    # Save the dataframe to the output path
    df.to_pickle(output_path)
    print(output_path)

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a dataframe and grade the prompts.")
    parser.add_argument("input_path", type=str, help="Path to the input pickle file")
    parser.add_argument("output_path", type=str, help="Path to save the output pickle file")

    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.input_path, args.output_path)
