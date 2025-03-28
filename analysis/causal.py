import matplotlib.pyplot as plt
import os
import matplotlib.patches as mpatches
import pandas as pd 
import numpy as np

###########GLOBALS
# color_dict = {
#     'mlp\noffensive\nc5.5': "#ff0000",  # Dark red for offensive MLP
#     'mlp\noffensive\nc10': "#ff0000",   # Dark red for offensive MLP
#     'linear\noffensive\nstrength\n0.25': "#ffcccb",  # Light red for offensive linear
#     'linear\noffensive\nstrength\n1': "#ffcccb",  # Light red for offensive linear
#     'no\nintervention\ncontrol': "#808080",  # Control color unchanged
#     'mlp\ndefensive\nc1': "#00008B",  # Dark blue for defensive MLP
#     'linear\ndefensive\nstrength\n-0.25': "#ADD8E6",  # Light blue for defensive linear
#     'linear\ndefensive\nstrength\n-1': "#ADD8E6",  # Light blue for defensive linear
# }

# color_dict = {
#     'mlp\noffensive': "#ff0000",  # Dark red for offensive MLP
#     'linear\noffensive': "#ffcccb",  # Light red for offensive linear
#     'no\nintervention\ncontrol': "#808080",  # Control color unchanged
#     'mlp\ndefensive': "#00008B",  # Dark blue for defensive MLP
#     'linear\ndefensive': "#ADD8E6",  # Light blue for defensive linear
#     'transformer\ndefensive': "#ADD8E6",  # Light blue for defensive linear
# }

# color_dict = {
#     'mlp\noffensive': "#ff0000",    # Dark red
#     'linear\noffensive': "#ffcccb",  # Light red
#     'transformer\noffensive': "#8B0000",  # Medium-dark red
#     'no\nintervention\ncontrol': "#808080",  # Gray
#     'mlp\ndefensive': "#00008B",    # Dark blue
#     'linear\ndefensive': "#ADD8E6",  # Light blue
#     'transformer\ndefensive': "#4169E1"  # Medium blue
# }

# Update color dictionary to ensure consistent shading
color_dict = {
        'linear\noffensive': "#ffcccb",  # Lightest red
        'mlp\noffensive': "#ff0000",     # Medium red
        'transformer\noffensive': "#8B0000",  # Darkest red
        'no\nintervention\ncontrol': "#808080",  # Gray
        'linear\ndefensive': "#ADD8E6",  # Lightest blue
        'mlp\ndefensive': "#4169E1",     # Medium blue
        'transformer\ndefensive': "#00008B"  # Darkest blue
    }
    

############CALCULATIONS
def describe_asr_dfs(datasets):
    for category, dataset_group in datasets.items():
        print(f"Category: {category}")
        for name, df in dataset_group.items():
            print(f"  Dataset: {name}")
            print(f"  Len: {len(df.dropna())}")
            rating_column = df.columns[df.columns.str.contains("rating", case=False)][0]
            print("rating:", rating_column)
            df.rename(columns={rating_column: "rating"}, inplace=True)
            print(f"  Rating distribution:\n{df["rating"].value_counts(normalize=True)}")
            print()


def calculate_standard_error(row):
    asr = row['asr'] / row['len']  # Convert percentage to decimal
    n_samples = row['len']
    
    # print("hit asr calc")
    # print(asr, n_samples)
    # print( np.sqrt((asr * (1 - asr)) / n_samples))
    return np.sqrt((asr * (1 - asr)) / n_samples)


# Function to process datasets (add columns, concatenate, and analyze)
# def asr_by_probe_type(datasets):
#     # Add 'probe_type' and 'len' columns to each dataset
#     for name, df in datasets.items():
#         probe_types = ["mlp", "transformer", "linear", "control"]  
#         df["probe_type"] = next((probe for probe in probe_types if probe in name), None)
#         #df["probe_type"] = name
        
#         c_strengths = [40, 30, 20, 15, 10, 5, -1, -0.5, -0.4, -0.25, 0.25, 0.4, 0.5, 1]  # Replace with your actual list
#         df["c_strength"] = next((c for c in c_strengths if str(c) in name), None)
        
#         lrs = [0.01, 0.025, 0.05, 0.1]  # Replace with your actual list
#         df["lr"] = next((l for l in lrs if str(l) in name), None)
        
#         df["offensive_defensive"] = "offensive" if "offensive" in name else "defensive"

        
#         df["len"] = len(df)
#         # Rename the column containing "rating" to "rating"
#         # print(df.columns)
#         rating_column = [col for col in df.columns if 'rating' in col.lower()][0]
#         if rating_column:
#            #  print(rating_column)
#             df.rename(columns={rating_column[0]: "rating"}, inplace=True)
#             # print("NEW COLUMNS", df.columns)
    
#     # Concatenate all datasets with 'rating' column
#     combined_df = pd.concat(
#         [df for df in datasets.values()], ignore_index=True
#     )
    
#     # Ensure the 'probe_type' column maintains the original order
#     # probe_types_order = list(datasets.keys())  # Extract original order from dictionary keys
#     # combined_df["probe_type"] = pd.Categorical(
#     #    combined_df["probe_type"], categories=probe_types_order, ordered=True
#     # )
#     # Group by 'probe_type', 'rating', and 'len', and calculate ASR
#     combined_df = combined_df.groupby(["probe_type", "c_strength", "offensive_defensive", "lr", "rating", "len"], dropna=False).size().reset_index(name="asr").sort_values("probe_type")
    
    
#     # Filter for rows where 'rating' is "Yes"
#     combined_df = combined_df[combined_df["rating"] == "Yes"]
    
    
#     # Sort by 'asr' in descending order
#     combined_df = combined_df.sort_values(by=["asr"], ascending=False)
    
#     # calculate standard error using caclulate_standard_error function
#     combined_df['mean_error'] = combined_df.apply(calculate_standard_error, axis=1)
    
#     # represent ars as percentage
#     combined_df['asr'] = combined_df['asr'] / combined_df['len'] * 100
    
#     ##Finalize df 
#     # Reset index and rename column
#     combined_df = combined_df.reset_index(drop=True)
    
#     # Ensure the 'asr' column is properly named
#     if 0 in combined_df.columns:
#         combined_df = combined_df.rename(columns={0: "asr"})
    
#     # Filter rows where 'rating' is "Yes"
#     combined_df = combined_df[combined_df["rating"] == "Yes"]
    
#     # Calculate the standard error for 'asr' and add a 'mean_error' column
#     combined_df['mean_error'] = combined_df.apply(calculate_standard_error, axis=1)
    
#     # Format the 'probe_type' column for better readability
#     combined_df['probe_type'] = combined_df['probe_type'].str.lower()
        
#     return combined_df        

def asr_by_probe_type(datasets):
    # Define valid mappings
    probe_strengths = {
        "mlp": [40, 30, 20, 15, 10, 5],
        "transformer": [40, 30, 20, 15, 10, 5],
        "linear": [-1, -0.5, -0.4, -0.25, -0.2, 0.2, 0.25, 0.4, 0.5, 1],
        "control": [None]
    }
    
    # Only MLPs and transformers have learning rates
    probes_with_lr = ["mlp", "transformer"]
    learning_rates = [0.01, 0.025, 0.05, 0.1]
    
    for name, df in datasets.items():
        # Get probe type
        probe_type = next((probe for probe in probe_strengths.keys() if probe in name.lower()), None)
        if not probe_type:
            continue
        
        df["filename"] = name
        df["probe_type"] = probe_type
        df["c_strength"] = next((c for c in probe_strengths[probe_type] if str(c) in name), None)
        df["lr"] = next((lr for lr in learning_rates if str(lr) in name), None) if probe_type in probes_with_lr else None
        df["offensive_defensive"] = df.apply(lambda row: 
          "nan" if row["probe_type"] == "control" 
          else ("offensive" if "offensive" in name.lower() else "defensive"), 
          axis=1)
        df["len"] = len(df)
        
        # Rename rating column
        rating_columns = [col for col in df.columns if 'rating' in col.lower()]
        if rating_columns:
            df.rename(columns={rating_columns[0]: "rating"}, inplace=True)

    # Process the combined DataFrame
    combined_df = pd.concat(list(datasets.values()), ignore_index=True)
    combined_df = (combined_df.groupby(["probe_type", "c_strength", "offensive_defensive", "lr", "rating", "len", "filename"], 
                                     dropna=False)
                             .size()
                             .reset_index(name="asr")
                             .sort_values("probe_type"))
    
    # Final processing
    combined_df = combined_df[combined_df["rating"] == "Yes"]
    combined_df['asr'] = combined_df['asr'] / combined_df['len'] * 100
    combined_df['mean_error'] = combined_df.apply(calculate_standard_error, axis=1)
    combined_df = combined_df.reset_index(drop=True)
    combined_df['probe_type'] = combined_df['probe_type'].str.lower()

    return combined_df
            
def calculate_asr_ratio_or_ces(df_asr_benign, df_asr_adversarial, ratio="regular"):
    """
    This function calculates the ratio or CES between benign and adversarial ASR values.
    
    Parameters:
    - df_asr_benign (DataFrame): The DataFrame containing benign ASR values and probe_type.
    - df_asr_adversarial (DataFrame): The DataFrame containing adversarial ASR values and probe_type.
    - ratio (str): Determines whether to calculate the regular ratio ("regular") or CES ("ces"). Default is "regular".
    
    Returns:
    - DataFrame: The modified df_asr_benign with the new 'ratio' column calculated.
    """
    # Reset indices to align rows properly
    df_asr_benign.reset_index(drop=True, inplace=True)
    df_asr_adversarial.reset_index(drop=True, inplace=True)

    # Ensure probe_type consistency (strip whitespace, lowercase)
    df_asr_benign["probe_type"] = df_asr_benign["probe_type"].str.strip().str.lower()
    df_asr_adversarial["probe_type"] = df_asr_adversarial["probe_type"].str.strip().str.lower()

    # Calculate ratio or CES
    for intervention in df_asr_benign["probe_type"].unique():
        # Create boolean masks to filter rows by intervention type
        mask_benign = df_asr_benign["probe_type"] == intervention
        mask_adversarial = df_asr_adversarial["probe_type"] == intervention

        # Ensure matching rows
        if mask_benign.sum() > 0 and mask_adversarial.sum() > 0:
            if ratio == "regular":
                # Calculate the regular ratio for matching rows
                df_asr_benign.loc[mask_benign, 'ratio'] = (
                    df_asr_benign.loc[mask_benign, 'asr'].values /
                    df_asr_adversarial.loc[mask_adversarial, 'asr'].values
                )
            else:
                # Calculate CES for matching rows
                epsilon = 1e-6  # Small constant to avoid division by zero
                df_asr_benign.loc[mask_benign, 'ratio'] = (
                    (df_asr_benign.loc[mask_benign, 'asr'].values**2 /
                    (df_asr_adversarial.loc[mask_adversarial, 'asr'].values + epsilon))/100
                )
    
    # Return the modified DataFrame
    return df_asr_benign


##########PLOTS 
# Function to simplify labels based on the presence of keywords
def simplify_label(category):
    if "mlp" in category.lower() and "defensive" in category.lower():
        return "MLP Defensive"
    elif "mlp" in category.lower() and "offensive" in category.lower():
        return "MLP Offensive"
    elif "linear" in category.lower() and "defensive" in category.lower():
        return "Linear Defensive"
    elif "linear" in category.lower() and "offensive" in category.lower():
        return "Linear Offensive"
    else:
        return category  # Default to original if it doesn't match expected patterns
      
      
# MMLU Performance Plot
def plot_mmlu_performance(mmlu_performance, savepath=None):
    labels = [label.replace(' ', '\n') for label in mmlu_performance.keys()]
    print("labels", labels)
    print("values", mmlu_performance.values())
    accuracy = [metrics['acc,none'] for metrics in mmlu_performance.values()]
    stderr = [metrics['acc_stderr,none'] for metrics in mmlu_performance.values()]

    # Get the colors for each label from color_dict
    colors = [color_dict[label.lower()] for label in mmlu_performance.keys()]

    fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure size
    bars = ax.bar(labels, accuracy, yerr=stderr, color=colors, edgecolor='black', capsize=10, width=0.7)

    ax.set_ylabel('Accuracy', fontsize=36)
    # ax.set_title('MMLU Performance', fontsize=38, weight='bold', pad=20)
    
    # Increase y-tick label size
    ax.tick_params(axis='y', labelsize=40)  # Set larger size for y-ticks
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    ax.set_ylim(0, max(accuracy) + 0.05)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 0.01, f'{height:.2f}',
                ha='center', va='bottom', fontsize=32, color='black', fontweight='bold')

    plt.tight_layout()
    if savepath: 
      plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()

def plot_mmlu_performance_from_df(df, title = None, accuracy_column = "acc_none", std_err_column = "acc_stderr", savepath=None):

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 20))
    
    # Extract data
    labels = [f"{row['probe_type']}\n{row['offensive_defensive']}"
              for _, row in df.iterrows()]
    print(labels)
    # if "control" in labels, replace with "no\intervention\control"
    labels = [label.replace("control\nnone", "no\nintervention\ncontrol") for label in labels]
    labels = [label.replace("control\nnan", "no\nintervention\ncontrol") for label in labels]
    labels = [label.replace("control\ncontrol", "no\nintervention\ncontrol") for label in labels]

    print(labels)
    
    colors = [color_dict[label.lower()] for label in labels]

    accuracies = df[accuracy_column]
    accuracies = [round(accuracy * 100, 1) for accuracy in accuracies]  # Convert to percentage
    
    errors = df[std_err_column]
    
    # Create bar plot with error bars
    bars = ax.bar(labels, accuracies, 
                  yerr=errors, 
                  color=colors,
                  edgecolor='black',
                  capsize=10,
                  width=0.7)
    
    # Customize the plot
    ax.set_ylabel('Accuracy', fontsize=36)
    ax.tick_params(axis='y', labelsize=40)  # Set larger size for y-ticks
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    
    # Set y-axis limits
    ax.set_ylim(0, max(accuracies) + 0.05 *100) 
    
    # Add grid
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # add title
    ax.set_title(title, fontsize=38, weight='bold', pad=20)
    
    # Add value labels on top of each bar
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height + 0.01,
                f'{height:.2f}',
                ha='center', va='bottom',
                fontsize=32, color='black',
                fontweight='bold')
    
    plt.tight_layout()
    
    # Save if path is provided
    if savepath:
        plt.savefig(savepath, bbox_inches='tight', format='pdf')
    
    plt.show()


def plot_asr(df, title, savepath=None):
    print(df["probe_type"].unique(), df["offensive_defensive"].unique())

    # First handle the control case before concatenation
    df["offensive_defensive"] = df["offensive_defensive"].fillna("")
    df["joint_probe_type"] = df.apply(
        lambda row: "no\nintervention\ncontrol" if row["probe_type"] == "control" 
        else row["probe_type"] + "\n" + row["offensive_defensive"],
        axis=1
    )
    
    # make sure that "asr" column is int
    df["joint_probe_type"] = df["joint_probe_type"].astype(str)
    df["asr"] = df["asr"].astype(int)
    colors = [color_dict.get(label.lower(), "#808080") for label in df["joint_probe_type"]]  # Default to gray if not found

    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure size

    # Define bar width for thinner bars
    bar_width = 0.7
    bars = ax.bar(df['joint_probe_type'], df['asr'], color=colors, edgecolor="black", width=bar_width, capsize=8)

    # Adding ASR values and standard errors within each respective bar
    
    for i, bar in enumerate(bars):
      asr = df["asr"].iloc[i]
      
      if "mean_error" in df.columns:
          error = df["mean_error"].iloc[i]
          ax.text(bar.get_x() + bar.get_width() / 2, asr + 1, 
                  f'{int(asr)}%\n±{error:.2f}',  # This is the text content
                  ha='center', va='bottom', fontsize=14, 
                  color='black', fontweight='bold')
      else:
          ax.text(bar.get_x() + bar.get_width() / 2, asr + 1,
                  f'{int(asr)}%',  # This is the text content
                  ha='center', va='bottom', fontsize=14,
                  color='black', fontweight='bold')

    # Set axis labels and title
    ax.set_ylabel("ASR (%)", fontsize=16)
    ax.set_xlabel("Intervention Type", fontsize=16)
    ax.set_title(title, fontsize=18, weight='bold', pad=20)

    # Set x-tick labels
    ax.set_xticks(range(len(df['joint_probe_type'])))
    ax.set_xticklabels(df['joint_probe_type'], rotation=45, ha='right', fontsize=14)

    # Set y-axis tick parameters
    ax.tick_params(axis='y', labelsize=14)

    # Adjust y-axis limit for clarity
    ax.set_ylim(0, max(df['asr']) + 20)

    # Tight layout and save to file
    plt.tight_layout()
    
    if savepath: 
      # create parent dir if it does not exist
      os.makedirs(os.path.dirname(savepath), exist_ok=True)
      plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()



# Benign to Harmful Compliance Ratio Plot
def plot_benign_to_harmful_ratio(df, asr_or_ratio = "ratio", savepath=None):
    df['probe_type'] = df['probe_type'].str.replace('_', '\n')
    colors = [color_dict[label.lower()] for label in df["probe_type"]]
    
    fig, ax = plt.subplots(figsize=(10, 8))  # Slightly larger figure size
    bars = ax.bar(df['probe_type'], df[asr_or_ratio], color=colors, edgecolor="black", width=0.6)

    for i, bar in enumerate(bars):
        ratio = df[asr_or_ratio].iloc[i]
        error = df["mean_error"].iloc[i]
        ax.text(bar.get_x() + bar.get_width() / 2, ratio, f'{ratio:.1f}',
                ha='center', va='bottom', fontsize=32, color='black', fontweight='bold')

    # ax.axvline(x=len(compliance_decrease) + 0.5, color='black', linestyle='--', linewidth=2)
    # ax.set_title("Benign Compliance to Harmful Compliance Ratio", fontsize=30, weight='bold', pad=20)
    # ax.set_xlabel("Intervention Type", fontsize=34)
    ax.set_ylabel(asr_or_ratio , fontsize=36)
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis='y', labelsize=32)
    ax.set_ylim([0, max(df[asr_or_ratio]) + 2])

    plt.tight_layout()
    if savepath: 
      plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()
    
def create_legend_causal(savepath=None):
    color_dict_simple = {simplify_label(key): value for key, value in color_dict.items()}

    # Create a larger figure for the grid layout
    fig, ax = plt.subplots(figsize=(6, 4))

    # Create a patch (color box) for each label
    patches = [mpatches.Patch(color=color, label=label) for label, color in color_dict_simple.items()]
    
    # Display the legend in a grid format
    ax.legend(handles=patches, loc='center', frameon=False, fontsize=38, 
              ncol=5, handleheight=2, handlelength=2, borderpad=1.5, labelspacing=1.5, markerscale=2)
    
    # Hide the axis
    ax.axis('off')

    # Save the legend as a standalone image
    plt.tight_layout()
    if savepath: 
      plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()