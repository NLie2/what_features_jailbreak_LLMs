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
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim





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
  return df


def create_logistic_regression_model(data, labels, split_idx):
    '''
    input: layer_idx (int), split_idx (int)
    The split_idx is the index at which the training data ends and the testing data begins
    output: trained logistic regression model
    '''

    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)

    x_train = data[:split_idx]
    y_train = labels[:split_idx]
    
    print(len(x_train), len(y_train))

    log_model.fit(x_train, y_train)
    
    
    x_test = data[split_idx:]
    y_test = labels[split_idx:]
    print(len(x_test), len(y_test))
    
    accuracy = log_model.score(x_test, y_test)

    return log_model, accuracy


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
  
def evaluate_logistic_regression_model(log_model, data, labels, split_idx):
    
    x_test = data[split_idx:]
    y_test = labels[split_idx:]
    
    accuracy = log_model.score(x_test, y_test)

    return accuracy


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



def get_confusion_matrix(log_model, X_test, y_test):
    '''
    input: log_model (LogisticRegression), X_test (array), y_test (array)
    output: confusion matrix (array)
    '''
    y_pred = log_model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def plot_confusion_matrix(accuracy, conf_matrix, class_names, dataset_name, model_name, layer, save=False):
    '''
    input: conf_matrix (array), class_names (list of str)
    output: None
    '''
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix of Linear Probe with Accuracy {accuracy} | {model_name} (Layer {layer}), {dataset_name}')

    if save:
        now = datetime.datetime.now().strftime("%d%m%Y%H:%M:%S")
        save_fig_path = f'linear_probe_{dataset_name}_{now}.pdf'.replace('/', '_')  # Replace with your desired path
        plt.savefig(save_fig_path)  
    plt.show()
    
    
def create_mlp_model(data, labels, split_idx, hidden_layer_sizes=(8,), max_iter=1000):
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

    # Get the input features and target variable
    x_train = data[:split_idx]
    y_train = labels[:split_idx]
    
    # Scale the input features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(x_train)

    # Train the model
    mlp_model.fit(X_train_scaled, y_train)
    
    x_test = data[split_idx:]
    y_test = labels[split_idx:]
    
    accuracy = mlp_model.score(x_test, y_test)

    return mlp_model, accuracy
  
  
def evaluate_mlp_model(mlp_model, data, labels, split_idx): 
    x_test = data[split_idx:]
    y_test = labels[split_idx:]
    
    accuracy = mlp_model.score(x_test, y_test)
    
    return accuracy 
  


def split_dataset_by_jailbreak(df): 
  """
  Make a new df for every jailbreak prompt type 
  """
  jailbreak_dfs = {}
  for jailbreak in df["jailbreak_prompt_name"].unique():
      print(jailbreak) 
      jailbreak_dfs[jailbreak] = df[df["jailbreak_prompt_name"] == jailbreak]
      print(len(jailbreak_dfs[jailbreak]))
      jailbreak_dfs[jailbreak]
      
      # shuffle and balance all dfs
      jailbreak_dfs[jailbreak] = balance_dataset(jailbreak_dfs[jailbreak], "rating_binary")
      jailbreak_dfs[jailbreak]["rating_binary"].value_counts()

      # shuffle dataset
      jailbreak_dfs[jailbreak] = jailbreak_dfs[jailbreak].sample(frac=1).reset_index(drop=True)

      # Get value counta for for rating_binary
      print(jailbreak_dfs[jailbreak]["rating_binary"].value_counts())
      
  return jailbreak_dfs
    
  
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
        probe, accuracy = create_logistic_regression_model(train_data, labels, split_idx)
      
      elif probe_type == "MLP":
        print("PROBE TYPE", probe_type)
        probe, accuracy = create_mlp_model(train_data, labels, split_idx)

      results_df.loc[jailbreak_1, jailbreak_1] = accuracy
    
    except: 
      continue

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
          accuracy = evaluate_logistic_regression_model(probe, test_data, labels, split_idx)
          results_df.loc[jailbreak_1, jailbreak_2] = accuracy
      elif probe_type == "MLP":
          accuracy = evaluate_mlp_model(probe, test_data, labels, split_idx)
          results_df.loc[jailbreak_1, jailbreak_2] = accuracy        
    
  return results_df




def holdout_transfer_graph(holdout_df, dataset_name, probe_type, savepath=None):
    # Assuming holdout_df is your DataFrame and you want to reset the index
    holdout_df = holdout_df.reset_index()  # Reset the index

    # Rename the new column created from the index to 'technique'
    holdout_df = holdout_df.rename(columns={'index': 'technique'})

    # Select the relevant columns
    holdout_df = holdout_df[['technique', 'train set', 'test set (holdout)']]

    # Sort by the 'test set (holdout)' column in ascending order
    holdout_df = holdout_df.sort_values(by='test set (holdout)', ascending=True)

    # Set 'technique' as the index
    holdout_df.set_index('technique', inplace=True)

    # Plot the bar plot
    holdout_df.plot(kind='barh', figsize=(12, 8), color=['skyblue', 'salmon'], linewidth=0.5)

    # Add titles and labels
    plt.title(f'{probe_type} Accuracy by Holdout Type {dataset_name}', fontsize=20)
    plt.xlabel('Accuracy', fontsize=18)
    plt.ylabel('Held out Category', fontsize=18)
    plt.xticks(rotation=45, ha="right", fontsize = 14)
    plt.yticks(fontsize = 14)

    plt.legend(title='Accuracy Type', bbox_to_anchor=(1.05, 1), fontsize=14)

    # Add a red vertical line at 0.50 for accuracy threshold
    plt.axvline(x=0.50, color='red', linestyle='--', linewidth=2, label="Threshold 0.50")

    # Set the x-axis limits from 0 to 1
    plt.xlim(0, 1)

    # Show the plot
    plt.tight_layout()
    
    if savepath: 
        plt.savefig(savepath)
        
    plt.show()


def transfer_with_holdout(df, probe_type, layer):
  
  jailbreak_dfs = split_dataset_by_jailbreak(df)
   

  holdout_df = pd.DataFrame(columns=["train set", "test set (holdout)"] , index=jailbreak_dfs.keys())


  for i in df['jailbreak_prompt_name'].unique():
    df_holdout = df[df['jailbreak_prompt_name'] != i]
    print("Holdout: ", i)
    print(df_holdout['jailbreak_prompt_name'].unique())
  
    
    # shuffle and balance all dfs
    df_holdout = balance_dataset(df_holdout, "rating_binary")
    df_holdout["rating_binary"].value_counts()

    # shuffle dataset
    df_holdout = df_holdout.sample(frac=1).reset_index(drop=True)

    # Get value counta for for rating_binary
    print(df_holdout["rating_binary"].value_counts())
  
  
    split_idx = int(0.8* len(df_holdout))
    
    labels = df_holdout["rating_binary"].to_list()
    # print(len(labels))

    train_data = df_holdout[f'layer_{layer}'].to_list()
    # print(len(train_data))
    
    if probe_type == "logistic_regression":
      probe, accuracy = create_logistic_regression_model(train_data, labels, split_idx)
    
    elif probe_type == "MLP": 
      probe, accuracy = create_mlp_model(train_data, labels, split_idx)
  

    holdout_df.loc[i,"train set", ] = accuracy
    print(accuracy)
  
  
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
      accuracy = evaluate_logistic_regression_model(probe, test_data, labels, split_idx)
    elif probe_type == "MLP":
      accuracy = evaluate_mlp_model(probe, test_data, labels, split_idx)

    holdout_df.loc[i, "test set (holdout)"] = accuracy
    print(accuracy)
    
  return holdout_df
  


def transfer_heatmap(results_df, dataset_name, probe_type, savepath = None):
    
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
    plt.figure(figsize=(12, 10))

    # Plot the heatmap for the main data
    ax = sns.heatmap(results_df, annot=True, fmt=".2f", cmap=cmap, cbar=True, 
                     linewidths=.5, mask=mask & ~highlight_mask, square=True, 
                     annot_kws={"color": "black"})

    # Overlay the row/column means with a different color
    sns.heatmap(results_df, annot=True, fmt=".2f", cmap=mean_cmap, cbar=False, 
                linewidths=.5, mask=~highlight_mask, square=True, 
                annot_kws={"color": "black"}) 

    # Now manually annotate the masked (white) cells
    for i in range(results_df.shape[0]):
        for j in range(results_df.shape[1]):
            if mask.iloc[i, j]:  # Access the mask correctly with .iloc
                ax.text(j + 0.5, i + 0.5, f"{results_df.iloc[i, j]:.2f}",
                        ha='center', va='center', color='black', fontsize=10)
                
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    plt.xlabel("Test Data", fontsize=18)
    plt.ylabel("Training Data", fontsize=18)
    plt.title(f"Heatmap of {probe_type} Accuracy {dataset_name}", fontsize=20, pad=20)
    
    
    if savepath: 
        plt.savefig(savepath)
    
    plt.show()
    


  
 

