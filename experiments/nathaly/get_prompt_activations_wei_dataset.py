import pandas as pd
import sys

sys.path.append("/data/nathalie_maria_kirch/ERA_Fellowship")

from gen_data.get_wildJailbreak import get_activations_from_df


# data = pd.read_pickle('/data/nathalie_maria_kirch/jailbroken/jailbroken_prompts_last.pkl')
# data = pd.read_pickle("/data/nathalie_maria_kirch/jailbroken/jailbroken_prompts_gemmma_7b.pkl")
# data = pd.read_pickle("/data/nathalie_maria_kirch/jailbroken/dataframes/data_all.pkl")
data = pd.read_pickle("/data/nathalie_maria_kirch/jailbroken/dataframes/df_concat_wei.pkl")


# melt the dataframe to only have "prompt_name" as a column and make a new column called "jailbreak_prompt_name" that will be the column name
data = data.melt(id_vars=['prompt_name'], var_name='jailbreak_prompt_name', value_name='jailbreak_prompt_text')

# print unique entries in jailbreak_prompt_name
print(data['jailbreak_prompt_name'].unique())
print(data['prompt_name'].unique())


model_name = "google/gemma-7b-it"
dataset_name = "wei_dataset"



# path = get_activations_from_df(model_name = model_name, dataset_name = dataset_name, df = data, prompt_column = "jailbreak_prompt_text", token_position = "last", generate_output=True, type_of_activations="prompt_res_activations")

# print(path)

total_length = len(data)
quartile_1, quartile_2, quartile_3 = total_length//4, total_length//2, 3*total_length//4
data_quartile_1 = data[:quartile_1]
data_quartile_2 = data[quartile_1:quartile_2]
data_quartile_3 = data[quartile_2:quartile_3]
data_quartile_4 = data[quartile_3:]


path = get_activations_from_df(model_name = model_name, dataset_name = dataset_name, df = data_quartile_1, prompt_column = "jailbreak_prompt_text", token_position = "last", generate_output=True, type_of_activations="prompt_res_activations")

print(path)

path = get_activations_from_df(model_name = model_name, dataset_name = dataset_name, df = data_quartile_2, prompt_column = "jailbreak_prompt_text", token_position = "last", generate_output=True, type_of_activations="prompt_res_activations")

print(path)

path = get_activations_from_df(model_name = model_name, dataset_name = dataset_name, df = data_quartile_3, prompt_column = "jailbreak_prompt_text", token_position = "last", generate_output=True, type_of_activations="prompt_res_activations")

print(path)

path = get_activations_from_df(model_name = model_name, dataset_name = dataset_name, df = data_quartile_4, prompt_column = "jailbreak_prompt_text", token_position = "last", generate_output=True, type_of_activations="prompt_res_activations")

print(path)

