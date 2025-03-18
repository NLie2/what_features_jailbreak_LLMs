import pandas as pd
import numpy as np

# Create the first dataframe from the provided data
data1 = {
    'asr': [28, 67, 0, 65, 61],
    'model': ['Ministral-8B-Instruct', 'Ministral-8B-Instruct', 'Ministral-8B-Instruct', 'Ministral-8B-Instruct', 'Ministral-8B-Instruct'],
    'probe_type': ['mlp', 'linear', 'control', 'linear', 'transformer'],
    'offensive_defensive': ['offensive', 'offensive', np.nan, 'defensive', 'defensive'],
    'c_strength': [15.0, 0.4, np.nan, -0.4, 40.0],
    'lr': [0.025, np.nan, np.nan, np.nan, 0.050],
    'capability': [0.639012, 0.630537, 0.638940, 0.632673, 0.633813],
    'probe_path': ['/data/nathalie_maria_kirch/ERA_Fellowship/data...', '/data/nathalie_maria_kirch/ERA_Fellowship/data...', np.nan, '/data/nathalie_maria_kirch/ERA_Fellowship/data...', '/data/nathalie_maria_kirch/ERA_Fellowship/data...'],
    'capability_stderr': [0.003830, 0.003831, 0.003831, 0.003824, 0.003832],
    'target_value': [0.63894, 0.63894, 0.63894, 0.63894, 0.63894],
    'is_best_config': [True, True, True, True, True],
    'probe_type_offensive_defensive': ['mlp_offensive', 'linear_offensive', np.nan, 'linear_defensive', 'transformer_defensive']
}

df1 = pd.DataFrame(data1)

# Create the second dataframe from the provided data
data2 = {
    'mlp and model': ['Gemma 7B Instruct', 'Llama 3.1 8B Instruct', 'Llama 3.2 3B Instruct', 'Ministral 8B Instruct'],
    'baseline_ASR': [23, 12, 15, 49],
    'baseline_MMLU': [50.6, 68.1, 60.5, np.nan],
    'finetuned_ASR': [24, 4, 9, np.nan],
    'finetuned_MMLU': [50.35, 66.24, 59.09, np.nan],
    'finetune_name': ['baseline', 'baseline', 'assistantonly_again_ep5', 'baseline'],
    'notes': ['eval loss for this finetune was high. need mor...', np.nan, np.nan, np.nan]
}

df2 = pd.DataFrame(data2)

# Standardize model names in both dataframes
def standardize_model_name(name):
    # Remove '-' and standardize spacing
    return name.replace('-', ' ')

# Apply standardization to df1
df1['model_standardized'] = df1['model'].apply(standardize_model_name)

# Apply standardization to df2 and rename the column to match df1
df2['model_standardized'] = df2['mlp and model'].apply(standardize_model_name)

# Merge the dataframes on the standardized model names
merged_df = pd.merge(df1, df2, on='model_standardized', how='inner')

# Add 'finetune_' prefix to columns from df2 (except the model column)
columns_to_rename = [col for col in df2.columns if col != 'mlp and model' and col != 'model_standardized']
rename_dict = {col: f'finetune_{col}' for col in columns_to_rename}
merged_df = merged_df.rename(columns=rename_dict)

# Drop the temporary standardized model name columns
merged_df = merged_df.drop(['model_standardized'], axis=1)

# Display the result
print("Merged DataFrame:")
print(merged_df)

# Save the merged dataframe to a CSV file
merged_df.to_csv('merged_dataframe.csv', index=False)
print("Merged dataframe saved to 'merged_dataframe.csv'") 