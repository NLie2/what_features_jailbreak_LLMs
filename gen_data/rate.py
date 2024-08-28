import pandas as pd
import os
from dotenv import load_dotenv
from .helper import save_with_timestamp
from strongreject.strongreject.strongreject_evaluator import run_autograder_on_dataset, autograde_response

# Ensure environment variables are loaded
load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

def rate_with_harmbench(dataset_name, model_name, df, add_info="", jailbreak_cutoff=0, save_interval=10):
    # Filter all responses that have the word "harmful" in the datatype column
    harmful_responses = df[df['data_type'].str.contains('harmful')].copy()

    # Filter all responses that DO NOT have the word "harmful" in the datatype column
    benign_responses = df[~df['data_type'].str.contains('harmful')].copy()

    save_path = save_with_timestamp(model_name=model_name, dataset_name=dataset_name, add_info="temp_"+add_info)

    # Split the harmful responses into chunks
    chunk_size = save_interval
    num_chunks = (len(harmful_responses) + chunk_size - 1) // chunk_size  # Calculate the number of chunks

    all_chunks_rated = []
    interim_save_path = save_with_timestamp(model_name=model_name, dataset_name=dataset_name, add_info=f"temp_{add_info}")

    for chunk_idx in range(num_chunks):
        print(f"progress: {chunk_idx+1}/{num_chunks}")
        chunk = harmful_responses.iloc[chunk_idx*chunk_size : (chunk_idx+1)*chunk_size]
        harmful_responses_rated_chunk = run_autograder_on_dataset(chunk)
        all_chunks_rated.append(harmful_responses_rated_chunk)
        
        # Save after every chunk
        interim_save = pd.concat(all_chunks_rated)
        print(f"Intermediate save: {interim_save_path}, Length: {len(interim_save)}")
        
        interim_save.to_pickle(interim_save_path)

    harmful_responses_rated = pd.concat(all_chunks_rated).sort_index()

    df = pd.concat([harmful_responses_rated, benign_responses]).sort_index()
    df = df.fillna(0)

    # Add a binary column to the dataset that indicates whether the response was successful or not
    df['data_type_binary'] = df.apply(lambda row: row['data_type'] + ' successful' if row['score'] > jailbreak_cutoff else row['data_type'], axis=1)
    
    final_save_path = save_with_timestamp(model_name=model_name, dataset_name=dataset_name, add_info=add_info)

    # Save the all responses to a pickle file and return file path
    df.to_pickle(final_save_path)
    
    return final_save_path
