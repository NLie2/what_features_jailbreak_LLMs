import argparse
import pandas as pd
from gen_data.get_activations import get_activations_from_df
from pathlib import Path

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Process model activations.")
parser.add_argument("--model_name", required=True, help="Name or path of the model to use.")
args = parser.parse_args()

# Assign the parsed model name
model_name = args.model_name

# Define paths
notebook_dir = Path.cwd()
project_root = notebook_dir.parent
data = pd.read_pickle(project_root / "datasets/jailbreak_success.pkl")

# Generate activations
path = get_activations_from_df(
    model_name=model_name,
    df=data,
    prompt_column="jailbreak_prompt_text",
    token_position="last",
    generate_output=True,
    type_of_activations="prompt_res_activations"
)

print(f"Generated activations saved at: {path}")
