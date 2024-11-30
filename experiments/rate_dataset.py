#!/usr/bin/env python
# coding: utf-8

import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
import sys
from pathlib import Path
import os
from gen_data.harmBench_autograder import *

def main(input_path, column_to_grade):
  
    # Get the current working directory (where the notebook is)
    current_dir = Path.cwd()

    # Move up to the root directory (parent of 'experiments')
    project_root = current_dir.parent

    # os.chdir(project_root)

    # Load the dataframe from the input path
    df = pd.read_pickle(project_root / input_path)
    
    output_path = input_path.replace(".plk", "_rated.pkl")
    
    # Load the HarmBench model
    model_name = "cais/HarmBench-Llama-2-13b-cls"
    autograder = AutoModelForCausalLM.from_pretrained(model_name)
    autograder.to('cuda')
    autogr_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, truncation_side="left", padding_side="left")
    
    # Apply the grading function and add the ratings column
    df[f"rating_{column_to_grade}"] = df.apply(lambda row: grade_with_HF(row['original_prompt_text'], [row[column_to_grade]], autograder, autogr_tokenizer, verbose = True)[0], axis=1)
    print("rated")

    # Save the dataframe to the output path
    df.to_pickle(output_path)
    print(output_path)
 
if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Process a dataframe and grade the prompts.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to the input pickle file")
    parser.add_argument("--column_to_grade", type=str, required=True, help="Column to grade")
  
    args = parser.parse_args()

    # Run the main function with the provided arguments
    main(args.input_path, args.column_to_grade)
