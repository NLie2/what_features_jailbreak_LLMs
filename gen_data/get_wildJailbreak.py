from datasets import load_dataset
import pandas as pd 
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import pickle
import sys

# Ensure environment variables are loaded
load_dotenv()

repo_path = '/data/nathalie_maria_kirch/ERA_Fellowship/strongreject'
sys.path.append(repo_path)

from strongreject.strongreject_evaluator import run_autograder_on_dataset, autograde_response

from .get_activations import get_model, get_res_activations
from .helper import save_with_timestamp

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def load_dataset_with_activations(model_name, n_rows, n_start = 0, token_position = "all", dataset_name="allenai/wildjailbreak", only_harmful = False, rate=False, generate_output=False, jailbreak_cutoff=0): 
    # get model 
    model, tokenizer = get_model(model_name)

    # Load the WildJailbreak training set
    df = pd.DataFrame(load_dataset(dataset_name, "train", use_auth_token=HF_TOKEN, delimiter="\t", keep_default_na=False)["train"])

    print("Dataset loaded successfully")

    if only_harmful: 
        # get only rows with data_type that contain the word harmful
        df = df[df["data_type"].str.contains("harmful")]

        df = df[n_start:].sample(n_rows, random_state=42)
        # df = df.iloc[n_start:n_start + n_rows]

    else: 
      df = df[n_start:].sample(n_rows, random_state=42)

    # df = pd.DataFrame(dataset["train"].shuffle(seed=42).select(range(n_rows)))
    print(f"DataFrame created with {n_rows} rows")

    df['forbidden_prompt'] = np.where(df['adversarial'] != '', df['adversarial'], df['vanilla'])
    df['response'] = None
    df['res_activations'] = None
    
    counter = 0
    for row in df.iterrows():
        index = row[0]
        counter +=1 

        print(f"Processing prompt {counter + 1}/{n_rows}")
    
        adv_prompt = row[1]["forbidden_prompt"]

        output_activations = get_res_activations(model = model, tokenizer = tokenizer, token_position = token_position, prompt = adv_prompt, generate_output=generate_output)

        if generate_output:
          print(output_activations["output_text"])
          df.at[index, 'response'] = output_activations['output_text']

        df.at[index, 'res_activations'] = output_activations['res_activations']

    print("Processing complete")
    
    if not rate:
      return save_with_timestamp(model_name = model_name, dataset_name = dataset_name, save=True, df=df)

    # Filter all responses that have the word "harmful" in the datatype column
    harmful_responses = df[df['data_type'].str.contains('harmful')].copy()

    # Filter all responses that DO NOT have the word "harmful" in the datatype column
    benign_responses = df[~df['data_type'].str.contains('harmful')].copy()

    # Run the autograder on the harmful responses
    harmful_responses_rated = run_autograder_on_dataset(harmful_responses)

    df = pd.concat([harmful_responses_rated, benign_responses]).sort_index()
    df = df.fillna(0)

    # Add a binary column to the dataset that indicates whether the response was successful or not
    df['data_type_binary'] = df.apply(lambda row: row['data_type'] + ' successful' if row['score'] > 0 else row['data_type'], axis=1)

    # Save the all responses to a pickle file and return file path
    return save_with_timestamp(model_name, dataset_name, df)


def get_activations_from_df(model_name, df, prompt_column, n_rows= None, n_start = 0,  dataset_name =  "allenai/wildjailbreak", token_position = "all", generate_output = False, type_of_activations = "res_activations"): 
    # get model 
    model, tokenizer = get_model(model_name)

    print("Dataset loaded successfully with {n_rows} rows")

    df[type_of_activations] = None
    
    counter = 0 

    if n_rows == None: 
        n_rows = len(df) 
    for row in df.iloc[:n_rows].iterrows():
        counter+=1 
        index = row[0]
        print(f"Processing prompt {counter + 1}/{n_rows}")
    
        prompt = row[1][prompt_column]

        output_activations = get_res_activations(model = model, tokenizer = tokenizer, token_position = token_position, prompt = prompt, type_of_activations=type_of_activations, generate_output=generate_output)
        
        df.at[index, type_of_activations] = output_activations[type_of_activations]
        
        if generate_output: df.at[index, 'output_text'] = output_activations['output_text']

    print("Processing complete")
    
    return save_with_timestamp(model_name=model_name, dataset_name=dataset_name, save = True, df=df, add_info = type_of_activations)

