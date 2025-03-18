import sys 
import os
import torch
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer 
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from pathlib import Path

import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression

from analysis.Probes import *



fontsize1 = 40
fontsize2 = 36
fontsize3 = 34



## DATA PREPROCESSING 
def investigate_data_distribution(df):
  data_type_counts = df['data_type'].value_counts()
  print("1. Count of each data_type:")
  print(data_type_counts)
  print("\n")

  # 2. Split between 1's and 0's for each data_type
  print("2. Split between 1's and 0's for each data_type:")
  for data_type in df['data_type'].unique():
      subset = df[df['data_type'] == data_type]
      score_counts = subset['score'].value_counts().sort_index()
      total = len(subset)
      percentage_0 = (score_counts.get(0, 0) / total) * 100
      percentage_1 = (score_counts.get(1, 0) / total) * 100
      
      print(f"\n{data_type}:")
      print(f"  0's: {score_counts.get(0, 0)} ({percentage_0:.2f}%)")
      print(f"  1's: {score_counts.get(1, 0)} ({percentage_1:.2f}%)")
      print(f"  Total: {total}")

def balance_dataset(df, colname):
  value_counts = df[colname].value_counts()
  min_value = value_counts.min()
  df = df.groupby(colname).apply(lambda x: x.sample(min_value)).reset_index(drop=True)
  # shuffle balanced dataset
  df = df.sample(frac=1).reset_index(drop=True)
  return df



def get_validation_data(data_type, layer_idx, split_idx, y_col='score'):
    '''
    input: layer_idx (int), split_idx (int)
    The split_idx is the index at which the training data ends and the testing data begins
    output: validation data
    '''

    layer = data_type['layer_' + str(layer_idx)].tolist()[split_idx:]
    y_test = data_type[y_col].tolist()[split_idx:]
    y_test = [int(i) for i in y_test]

    return layer, y_test


def split_dataset_by_jailbreak(df): 
  """
  Make a new df for every jailbreak prompt type 
  """
  jailbreak_dfs = {}
  for jailbreak in df["jailbreak_prompt_name"].unique():
      jailbreak_dfs[jailbreak] = df[df["jailbreak_prompt_name"] == jailbreak]
      jailbreak_dfs[jailbreak]
      

      # shuffle and balance all dfs
      jailbreak_dfs[jailbreak] = balance_dataset(jailbreak_dfs[jailbreak], "rating_binary")
      jailbreak_dfs[jailbreak]["rating_binary"].value_counts()

      # shuffle dataset
      jailbreak_dfs[jailbreak] = jailbreak_dfs[jailbreak].sample(frac=1).reset_index(drop=True)

      # Get value counta for for rating_binary
      print(jailbreak_dfs[jailbreak]["rating_binary"].value_counts())
      
  return jailbreak_dfs
    
  
# def extract_layer(row, layer, activation_column= "prompt_res_activations"):
#     return row[activation_column][layer][0].cpu().numpy().flatten()

def extract_layer(row, layer, activation_column="prompt_res_activations"):
        activation = row[activation_column][layer][0]
        # Average across the first dimension (4 heads) instead of flattening
        return activation.mean(dim=0).cpu().numpy()  # This will give a (3072,) array

def get_data_and_labels_multilayer(df, layers, split_idx = 0.8):
    """
    Prepare data by extracting features, melting the dataframe, and splitting into train/val sets.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing the data
        layers (list): List of layers to extract features from
        exclude_prompts (list, optional): List of prompt names to exclude. Defaults to specified prompts.
    
    Returns:
        tuple: (train_data, train_labels, val_data, val_labels) as tensors
    """
    # Extract features for each layer
    for layer in layers:
        df[f'layer_{layer}'] = df.apply(lambda row: extract_layer(row, layer), axis=1)
    
    # Melt the DataFrame
    melted_df = pd.melt(
        df,
        id_vars=["jailbreak_prompt_name", "rating_binary"],
        value_vars=[f'layer_{layer}' for layer in layers],
        var_name="layer",
        value_name="features"
    )
        
    # Prepare labels and data
    labels = melted_df["rating_binary"].to_list()
    data = melted_df["features"].to_list()
    
    # Convert to tensors
    data_tensor = torch.tensor(np.stack(data), dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)
    
    # Split the dataset
    split_idx = int(split_idx * len(melted_df))
    train_data = data_tensor[:split_idx]
    val_data = data_tensor[split_idx:]
    train_labels = labels_tensor[:split_idx]
    val_labels = labels_tensor[split_idx:]
    
    return train_data, train_labels, val_data, val_labels


def get_data_and_labels_multilayer_aggregate(df, layers, n_layers_aggregate=1, agg_method='mean', split_idx=0.8):
    # Extract features  
    for layer in layers:
        df[f'layer_{layer}'] = df.apply(lambda row: extract_layer(row, layer), axis=1)
    
    # Get feature dimension from first sample
    feature_dim = len(df[f'layer_{layers[0]}'].iloc[0])
    num_samples = len(df)
    
    # Stack all layers into 3D array: (num_layers, num_samples, feature_dim)
    layer_features = np.stack([np.stack(df[f'layer_{layer}'].to_list()) for layer in layers])
    
    if agg_method == 'mean':
        data = np.mean(layer_features, axis=0)  # (num_samples, feature_dim)
    elif agg_method == 'max':
        data = np.max(layer_features, axis=0)   # (num_samples, feature_dim)
    elif agg_method == 'concat':
        data = layer_features.transpose(1, 0, 2).reshape(num_samples, -1)  # (num_samples, num_layers*feature_dim)
        
    labels = df["rating_binary"].to_list()
    
    split_idx = int(split_idx * len(df))

    return (data[:split_idx], labels[:split_idx], 
            data[split_idx:], labels[split_idx:])


## LINEAR MODELS

def create_logistic_regression_model(train_data, train_labels, test_data, test_labels):
    '''
    Input:
        data: Features (e.g., numpy array or pandas DataFrame)
        labels: Corresponding labels for the data
        split_idx: Index at which the training data ends and the testing data begins
    Output:
        log_model: Trained Logistic Regression model
        train_accuracy: Accuracy of the model on the training set
        test_accuracy: Accuracy of the model on the test set
    '''

    # Initialize the Logistic Regression model
    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    # Train the model
    log_model.fit(train_data, train_labels)

    # Calculate training and testing accuracy
    train_accuracy = log_model.score(train_data, train_labels)
    test_accuracy = log_model.score(test_data, test_labels)

    # Return model and accuracies
    return log_model, train_accuracy, test_accuracy



def train_logistic_regression_model(data, labels, split_idx):
    '''
    input: layer_idx (int), split_idx (int)
    The split_idx is the index at which the training data ends and the testing data begins
    output: trained logistic regression model
    '''

    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)

    x_train = data[:split_idx]
    y_train = labels[:split_idx]

    log_model.fit(x_train, y_train)
    
    return log_model
  

## MLP MODELS

def create_mlp_model(train_data, train_labels, test_data, test_labels, hidden_layer_sizes=(8,), max_iter=1000):
    '''
    input: 
        layer_idx (int): Index of the layer to use for input features
        split_idx (int): Index at which the training data ends and the testing data begins
        y_col (str): Name of the column containing the target variable
        hidden_layer_sizes (tuple): Tuple representing the number of neurons in each hidden layer
        max_iter (int): Maximum number of iterations for the solver to converge
    output: trained MLP model and scaler
    '''

    # Initialize the MLP classifier
    mlp_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, max_iter=max_iter, random_state=42)

    # # Get the input features and target variable
    # x_train = data[:split_idx]
    # y_train = labels[:split_idx]
    
    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_data)

    # Train the model
    mlp_model.fit(X_train_scaled, train_labels)
    
    train_accuracy = mlp_model.score(train_data, train_labels)

    
    test_accuracy = mlp_model.score(test_data, test_labels)

    return mlp_model, train_accuracy, test_accuracy
  


## MODEL EVALUATION
def get_confidence_interval(accuracy, n_samples): 
  s_e = np.sqrt((accuracy * (1 - accuracy)) / n_samples)
  
  lower_bound = accuracy - 1.96 * s_e
  upper_bound = accuracy + 1.96 * s_e
  
  return (lower_bound, upper_bound)


def get_confusion_matrix_and_accuracy(log_model, X_test, y_test):
    '''
    input: log_model (LogisticRegression), X_test (array), y_test (array)
    output: confusion matrix (array), accuracy (float)
    '''
    y_pred = log_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, round(accuracy, 2)


def evaluate_model(model, test_data, test_labels):

    print("eval fn")
        
    accuracy = model.score(test_data, test_labels)
    
    print(accuracy)

    return accuracy
  

def probe_transfer(df, probe_type, layer): 
  jailbreak_dfs = split_dataset_by_jailbreak(df)
  results_df = pd.DataFrame(columns=jailbreak_dfs.keys(), index=jailbreak_dfs.keys())

  for jailbreak_1, train_df in jailbreak_dfs.items():
    
    # print(train_df["rating_binary"].value_counts()) 
    split_idx = int(0.8* len(train_df))
    labels = train_df["rating_binary"].to_list()
    # print(len(labels))
    train_data = train_df[f'layer_{layer}'].to_list()
    # print(len(train_data))
    
    try: 
      if probe_type == "logistic_regression":
        probe, train_accuracy, test_accuracy = create_logistic_regression_model(train_data, labels, split_idx)
      
      elif probe_type == "MLP":
        print("PROBE TYPE", probe_type)
        probe, train_accuracy, test_accuracy = create_mlp_model(train_data, labels, split_idx)

      results_df.loc[jailbreak_1, jailbreak_1] = test_accuracy
    
    except: 
      print("Error training the")

    for jailbreak_2, test_df in jailbreak_dfs.items(): 
      
      if jailbreak_1 == jailbreak_2:
        # print(jailbreak_1, jailbreak_2)
        continue

      # shuffle and balance dataset
      test_df = balance_dataset(test_df, "rating_binary")
      test_df["rating_binary"].value_counts()

      # shuffle dataset
      test_df = test_df.sample(frac=1).reset_index(drop=True)

      # Get value counta for for rating_binary
      test_df["rating_binary"].value_counts() 

      split_idx = int(0.8* len(test_df))
      labels = test_df["rating_binary"].to_list()
      # print(len(labels))

      test_data = test_df[f'layer_{layer}'].to_list()
      # print(len(train_data))
      
      if probe_type == "logistic_regression":
          accuracy = evaluate_model(probe, test_data, labels, split_idx)
          results_df.loc[jailbreak_1, jailbreak_2] = accuracy
      elif probe_type == "MLP":
          accuracy = evaluate_model(probe, test_data, labels, split_idx)
          results_df.loc[jailbreak_1, jailbreak_2] = accuracy        
    
  return results_df



def transfer_with_holdout(df, probe_type, layer):
   

  holdout_df = pd.DataFrame(columns=["Train Set", "Test Set (Holdout)"] , index=df['jailbreak_prompt_name'].unique())
  

  for i in df['jailbreak_prompt_name'].unique():
    df_holdout = df[df['jailbreak_prompt_name'] != i]
    print("Holdout: ", i)  
    # Get value counts for for rating_binary
    print("before balancing")
    print(df_holdout["rating_binary"].value_counts())
    
    # shuffle and balance all dfs
    df_holdout = balance_dataset(df_holdout, "rating_binary")
    df_holdout["rating_binary"].value_counts()

    # shuffle dataset
    df_holdout = df_holdout.sample(frac=1).reset_index(drop=True)

    print("after balancing")
    # Get value counts for for rating_binary
    print(df_holdout["rating_binary"].value_counts())
  
  
    split_idx = int(0.8* len(df_holdout))
    
    labels = df_holdout["rating_binary"].to_list()
    # print(len(labels))

    train_data = df_holdout[f'layer_{layer}'].to_list()
    # print(len(train_data))
    
    if probe_type == "logistic_regression":
      probe, train_accuracy = create_logistic_regression_model(train_data, labels, split_idx)
    
    elif probe_type == "MLP": 
      probe, train_accuracy = create_mlp_model(train_data, labels, split_idx)
  

    holdout_df.loc[i,"Train Set", ] = train_accuracy
    print(train_accuracy)
  
  
    df_holdout_test = df[df['jailbreak_prompt_name'] == i]

    # shuffle and balance dataset
    test_df = balance_dataset(df_holdout_test, "rating_binary")
    test_df["rating_binary"].value_counts()

    # shuffle dataset
    test_df = test_df.sample(frac=1).reset_index(drop=True)

    # Get value counta for for rating_binary
    print(test_df["rating_binary"].value_counts())

    split_idx = int(0.8* len(test_df))
    labels = test_df["rating_binary"].to_list()

    test_data = test_df[f'layer_{layer}'].to_list()
    # print(len(train_data))
    
    if probe_type == "logistic_regression":
      test_accuracy = evaluate_model(probe, test_data, labels, split_idx)
    elif probe_type == "MLP":
      test_accuracy = evaluate_model(probe, test_data, labels, split_idx)

    holdout_df.loc[i, "Test Set (Holdout)"] = test_accuracy
    print(test_accuracy)
  
  return holdout_df
  

def transfer_with_holdout_with_errors(df, probe_type, layers, input_size = None):
  
    # Initialize a DataFrame to store results
    holdout_df = pd.DataFrame(columns=[
        "test accuracy w/o holdout var", 
        "test accuracy SE w/o holdout var", 
        "test size w/o holdout var", 
        "test accuracy w holdout var", 
        "test accuracy SE w holdout var", 
        "test size w holdout var"
    ], index=df['jailbreak_prompt_name'].unique())
  
    for i in df['jailbreak_prompt_name'].unique():
        # Split dataset into holdout and non-holdout
        df_wo_holdout_var = df[df['jailbreak_prompt_name'] != i]
        df_w_holdout_var = df[df['jailbreak_prompt_name'] == i]
        
        print("len df wo holdout: ", len(df_wo_holdout_var), "len df w holdout: ", len(df_w_holdout_var))
        print("Holdout: ", i)
        
        # Shuffle the dataset
        df_wo_holdout_var = df_wo_holdout_var.sample(frac=1).reset_index(drop=True)

        df_wo_holdout_var = balance_dataset(df_wo_holdout_var, "rating_binary")
  
        train_data, train_labels, test_data, test_labels = get_data_and_labels_multilayer_aggregate(df = df_wo_holdout_var, layers = layers, agg_method="mean")
        print("len train data: ", len(train_data), "len test data: ", len(test_data))
              
        # Train the model and get test accuracy without the holdout group
        if probe_type == "logistic_regression":
            probe, _, test_accuracy_wo_holdout = create_logistic_regression_model(train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
            
        elif probe_type == "MLP":
            # probe, _, test_accuracy_wo_holdout = create_mlp_model(train_data=train_data, train_labels=train_labels, test_data=test_data, test_labels=test_labels)
            probe = SimpleMLP(input_size=input_size)

            probe.to("cpu")


            probe.train_MLP(train_data = train_data, train_labels = train_labels, num_epochs = 500)
            test_accuracy_wo_holdout, _ = probe.get_accuracy(test_data = test_data, test_labels = test_labels)


        elif probe_type == "transformer":      
            trans_probe = TransformerProbe(input_size=input_size)
            trans_probe.to("cuda")

            trans_probe.train_probe(train_data = train_data, train_labels = train_labels, num_epochs = 500)
            test_accuracy_wo_holdout, _ = trans_probe.get_accuracy(test_data = test_data, test_labels = test_labels)

            # Calculate the standard error for the test accuracy without the holdout group
        n_test_samples_wo_holdout = len(test_data)  # number of samples in the test set
        test_se_wo_holdout = np.sqrt((test_accuracy_wo_holdout * (1 - test_accuracy_wo_holdout)) / n_test_samples_wo_holdout)

        # Store results in holdout_df
        holdout_df.loc[i, "test accuracy w/o holdout var"] = test_accuracy_wo_holdout
        holdout_df.loc[i, "test accuracy SE w/o holdout var"] = test_se_wo_holdout
        holdout_df.loc[i, "test size w/o holdout var"] = n_test_samples_wo_holdout
        
        ##### Transfer to the holdout group #####
        # labels_w_holdout_var = df_w_holdout_var["rating_binary"].to_list()
        
        # data_w_holdout_var = df_w_holdout_var[f'layer_{layer}'].to_list()
        
        #no training data needed for holdout group
        _, _, test_data, test_labels = get_data_and_labels_multilayer_aggregate(df = df_w_holdout_var, layers = layers, split_idx = 0, agg_method="mean")
        
        print(len(test_data))
        
        #
        # Evaluate the model on the holdout set
        if probe_type == "logistic_regression":
            test_accuracy_w_holdout = evaluate_model(probe, test_data, test_labels)
        elif probe_type == "MLP":
            # test_accuracy_w_holdout = evaluate_model(probe, test_data, test_labels)
            test_accuracy_w_holdout, _ = probe.get_accuracy(test_data = test_data, test_labels = test_labels)

        elif probe_type == "transformer":      
            test_accuracy_w_holdout, _ = trans_probe.get_accuracy(test_data = test_data, test_labels = test_labels)

       
        # Calculate the standard error for the test accuracy with the holdout group
        n_test_samples_w_holdout = len(test_data)  # number of samples in the test set
        test_se_w_holdout = np.sqrt((test_accuracy_w_holdout * (1 - test_accuracy_w_holdout)) / n_test_samples_w_holdout)

        # Store results in holdout_df
        holdout_df.loc[i, "test accuracy w holdout var"] = test_accuracy_w_holdout
        holdout_df.loc[i, "test accuracy SE w holdout var"] = test_se_w_holdout
        holdout_df.loc[i, "test size w holdout var"] = n_test_samples_w_holdout
        
        # Print the results for the current group
        print(f"Test Accuracy without holdout: {test_accuracy_wo_holdout}, SE: {test_se_wo_holdout}")
        print(f"Test Accuracy with holdout: {test_accuracy_w_holdout}, SE: {test_se_w_holdout}")
        
    return holdout_df


## GRAPHS 

def plot_confusion_matrix(accuracy, probe_type, conf_matrix, class_names, dataset_name, model_name, layer, savepath=None):
    '''
    input: conf_matrix (array), class_names (list of str)
    output: None
    '''
    fontsize1 = 40
    fontsize2 = 36
    fontsize3 = 34

    plt.figure(figsize=(12, 10))  # Increased figure size
    
    # Labels for the confusion matrix (0 = no jailbreak, 1 = jailbreak)
    labels = ["No Jailbreak", "Jailbreak"] 
    
    # Plot the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
    
    # Set axis labels
    plt.xlabel('Predicted', fontsize=fontsize2)
    plt.ylabel('Actual', fontsize=fontsize2)
    
    # Adjust y-axis tick padding using tick_params
    ax = plt.gca()
    ax.tick_params(axis='y', labelsize=fontsize3, pad=10)  # Add padding to y-axis labels
    ax.tick_params(axis='x', labelsize=fontsize3)  # Adjust x-axis font size

    # Add a figure text below the plot
    plt.figtext(0.5, -0.1, f"{probe_type} Probe\n{model_name.replace('-',' ').title()} (Layer {layer}) {dataset_name}\nAccuracy {accuracy}", 
                ha='center', fontsize=fontsize1, style='italic')  # moved text slightly lower

    # Adjust layout to ensure nothing overlaps
    plt.tight_layout(pad=12.0)  # Increased padding

    # Save the plot if save=True
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')
    
    # Show the plot
    plt.show()


def holdout_transfer_graph(holdout_df, dataset_name, model_name, layer, probe_type, mean_tranfer_score = None, savepath=None):
    
    
    holdout_df = holdout_df.reset_index()
    # Rename the index column to 'Technique'
    holdout_df = holdout_df.rename(columns={'index': 'technique'})
    'test accuracy w holdout var'

    # Set 'Technique' as the index
    holdout_df.set_index('technique', inplace=True)

    # Format the y-tick labels: replace '_' with ' ' and title case
    holdout_df.index = [label.replace('_', ' ').title().replace("Gcg", "GCG").replace("Aim", "AIM") for label in holdout_df.index]
    
    # Extract accuracy values and standard errors
    test_accuracy = holdout_df['test accuracy w holdout var']
    train_accuracy = holdout_df['test accuracy w/o holdout var']
    test_se = holdout_df['test accuracy SE w holdout var']
    test_size = holdout_df['test size w holdout var']

    # Plot the bar plot with adjusted bar height and width
    ax = holdout_df[['test accuracy w/o holdout var', 'test accuracy w holdout var']].plot(kind='barh', 
                                                              figsize=(15, 12), 
                                                              color=['skyblue', 'salmon'], 
                                                              linewidth=0.5, 
                                                              width=0.85)

    # Add error bars (horizontal line, no vertical caps) to the Test Set bars
    bar_height = 0.85  # Height of each bar (set in plot above)
    y_positions = np.arange(len(test_accuracy)) + ax.patches[0].get_height() / 2  # Middle of each bar

    ax.errorbar(test_accuracy,  # x-values (accuracy)
                y_positions,    # y-values (middle of the bars)
                xerr=test_se,   # Error values (standard errors)
                fmt='none',     # No marker
                ecolor='black', # Color of the horizontal error lines
                capsize=0,      # No vertical caps
                elinewidth=2,   # Thickness of horizontal error lines
                linestyle='-')  # Ensure the line is solid

    # Annotate test sample sizes near x=0
    for i, (test, size, y_pos) in enumerate(zip(test_accuracy, test_size, y_positions)):
        ax.text(0.02, y_pos, f'n={size}',  # Fixed position near x=0
                va='center', ha='left', fontsize=fontsize3-2, color='black', fontstyle='italic')

    # Add titles and labels
    plt.xlabel('Accuracy', fontsize=fontsize2)
    plt.ylabel('Held out Category', fontsize=fontsize2)
    plt.xticks(rotation=45, ha="right", fontsize=fontsize3)
    plt.yticks(fontsize=fontsize3)

    # Set the plot title
    # plt.figtext(0.5, -0.1, f'Mean Test Accuracy {mean_tranfer_score}\n{probe_type} Accuracy by Holdout Type\n{model_name.replace("-", " ").title()} (Layer {layer})', ha='center', fontsize=fontsize1, style='italic')
    plt.figtext(0.5, -0.1, f'{probe_type}\nMean Transfer Accuracy {mean_tranfer_score}\n{model_name.replace("-", " ").title()} (Layer {layer})', ha='center', fontsize=fontsize1, style='italic')

    # Move the legend underneath the graph
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=fontsize2, ncol=1)

    # Add a red vertical line at 0.50 for accuracy threshold
    plt.axvline(x=0.50, color='red', linestyle='--', linewidth=2, label="Threshold 0.50")

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=5.0)

    # Set the x-axis limits from 0 to 1
    plt.xlim(0, 1)

    # Show or save the plot
    if savepath:
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()




def transfer_heatmap(results_df, dataset_name, model_name, probe_type, savepath=None):
    # Compute the mean of each row and each column
    row_means = results_df.mean(axis=1)
    col_means = results_df.mean(axis=0)

    # Sort the row and column indexes by their means
    sorted_rows = row_means.sort_values(ascending=False).index
    sorted_cols = col_means.sort_values(ascending=True).index

    # Reorder the DataFrame based on sorted rows and columns
    results_df = results_df.loc[sorted_rows, sorted_rows]

    # Append row means and column means to the DataFrame
    results_df['Test Mean'] = row_means[sorted_rows]
    col_means_sorted = col_means[sorted_rows]
    results_df.loc['Training Mean'] = col_means_sorted

    # Set the "row mean" of the "column means" cell to NaN to exclude it from the heatmap
    results_df.loc['Training Mean', 'Test Mean'] = np.nan

    # Replace NaN with 0, except for the special NaN we just created
    results_df = results_df.fillna(0)
    results_df.loc['Training Mean', 'Test Mean'] = np.nan

    # Create a mask for values below 0.50 with the new DataFrame shape
    mask = results_df < 0.51

    # Create a mask to highlight the row and column means
    highlight_mask = pd.DataFrame(False, index=results_df.index, columns=results_df.columns)
    highlight_mask.iloc[-1, :] = True  # Last row for row means
    highlight_mask.iloc[:, -1] = True  # Last column for column means
    
    # Mask the specific "row mean" of the "column means" cell
    highlight_mask.loc['Training Mean', 'Test Mean'] = True

    # Create a custom colormap
    cmap = sns.color_palette("Greens", as_cmap=True)
    mean_cmap = sns.light_palette("orange", n_colors=50, as_cmap=True)  # Shaded gradient for row/column means
    
    # Set up the plot
    plt.figure(figsize=(20, 17))  # Increase figure size to make cells larger

    # Plot the heatmap for the main data
    ax = sns.heatmap(results_df, annot=True, fmt=".2f", cmap=cmap, cbar=True, 
                     linewidths=.5, mask=mask & ~highlight_mask, square=True, 
                     annot_kws={"color": "black", "fontsize": fontsize3})

    # Overlay the row/column means with a different color
    sns.heatmap(results_df, annot=True, fmt=".2f", cmap=mean_cmap, cbar=False, 
                linewidths=.5, mask=~highlight_mask, square=True, 
                annot_kws={"color": "black", "fontsize": fontsize3}) 

    # Now manually annotate the masked (white) cells
    for i in range(results_df.shape[0]):
        for j in range(results_df.shape[1]):
            if mask.iloc[i, j]:  # Access the mask correctly with .iloc
                ax.text(j + 0.5, i + 0.5, f"{results_df.iloc[i, j]:.2f}",
                        ha='center', va='center', color='black', fontsize=fontsize3)
                
    # Adjust ticks and labels for readability
    plt.xticks(fontsize=fontsize3)
    plt.yticks(fontsize=fontsize3)

    # Add axis labels and title
    plt.xlabel("Test Data", fontsize=fontsize2, labelpad=15)
    plt.ylabel("Training Data", fontsize=fontsize2, labelpad=15)
    plt.title(f"Heatmap of {probe_type} Accuracy \n {dataset_name} \n {model_name}",
              fontsize=fontsize1, pad=25)
    
    # Optionally save the figure
    if savepath: 
        savepath.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(savepath, bbox_inches='tight', dpi=300)  # High DPI for better quality
    
    plt.show()
    
