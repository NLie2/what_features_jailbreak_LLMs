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


import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim



fontsize1 = 40
fontsize2 = 36
fontsize3 = 34


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


def create_logistic_regression_model(data, labels, split_idx):
    '''
    input: layer_idx (int), split_idx (int)
    The split_idx is the index at which the training data ends and the testing data begins
    output: trained logistic regression model
    '''

    log_model = LogisticRegression(solver='lbfgs', max_iter=1000)

    x_train = data[:split_idx]
    y_train = labels[:split_idx]
    
    print("len train", len(x_train), len(y_train))

    log_model.fit(x_train, y_train)
    
    
    x_test = data[split_idx:]
    y_test = labels[split_idx:]
    print("len test", len(x_test), len(y_test))
    
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



# def get_confusion_matrix(log_model, X_test, y_test):
#     '''
#     input: log_model (LogisticRegression), X_test (array), y_test (array)
#     output: confusion matrix (array)
#     '''
#     y_pred = log_model.predict(X_test)
#     return confusion_matrix(y_test, y_pred)
def get_confusion_matrix_and_accuracy(log_model, X_test, y_test):
    '''
    input: log_model (LogisticRegression), X_test (array), y_test (array)
    output: confusion matrix (array), accuracy (float)
    '''
    y_pred = log_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    return cm, round(accuracy, 2)

def plot_confusion_matrix(accuracy, probe_type, conf_matrix, class_names, dataset_name, model_name, layer, save=False):
    '''
    input: conf_matrix (array), class_names (list of str)
    output: None
    '''
    plt.figure(figsize=(10, 8))
    
    # Labels for the confusion matrix (0 = no jailbreak, 1 = jailbreak)
    labels = ["No Jailbreak", "Jailbreak"] 
    
    # Plot the confusion matrix
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels, annot_kws={"size": 16})
    
    # Set axis labels
    plt.xlabel('Predicted', fontsize=fontsize2)
    plt.ylabel('Actual', fontsize=fontsize2)
    
    # Add a figure text below the plot
    plt.figtext(0.5, -0.1, f'{probe_type} Probe with Accuracy {accuracy:.2f} \n {model_name.replace("-"," ").title()} (Layer {layer}) {dataset_name}', 
                ha='center', fontsize=fontsize1, style='italic')  # moved text slightly lower
    
    # Adjust x and y ticks font size
    plt.xticks(fontsize=fontsize3)
    plt.yticks(fontsize=fontsize3)

    # Adjust layout to ensure nothing overlaps
    plt.tight_layout(pad=3.0)

    # Save the plot if save=True
    if save:
        save_fig_path = f'/data/nathalie_maria_kirch/ERA_Fellowship/images/confusion_matrices/{probe_type}_{dataset_name.replace("/", "_")}.pdf'  
        plt.savefig(save_fig_path, bbox_inches='tight')  # Added bbox_inches='tight'
    
    # Show the plot
    plt.show()

# def plot_confusion_matrix(accuracy, probe_type, conf_matrix, class_names, dataset_name, model_name, layer, save=False):
#     '''
#     input: conf_matrix (array), class_names (list of str)
#     output: None
#     '''
#     plt.figure(figsize=(10, 8))
    
#     # Labels for the confusion matrix (0 = no jailbreak, 1 = jailbreak)
#     labels = ["no jailbreak", "jailbreak"] 
    
#     # Plot the confusion matrix
#     sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Greens', xticklabels=labels, yticklabels=labels)
    
#     # Set axis labels
#     plt.xlabel('Predicted', fontsize=16)
#     plt.ylabel('Actual', fontsize=16)
    
#     # Add a figure text below the plot
#     plt.figtext(0.5, -0.05, f'{probe_type} probe with accuracy {accuracy:.2f} | {model_name} (Layer {layer}) {dataset_name}', 
#                 ha='center', fontsize=20, style='italic')
    
#     # Adjust x and y ticks font size
#     plt.xticks(fontsize=14)
#     plt.yticks(fontsize=14)

#     # Adjust layout to ensure nothing overlaps
#     plt.tight_layout()

#     # Save the plot if save=True
#     if save:
#         save_fig_path = f'/data/nathalie_maria_kirch/ERA_Fellowship/images/confusion_matrices/{probe_type}_{dataset_name.replace("/", "_")}.pdf'  
#         plt.savefig(save_fig_path)
    
#     # Show the plot
#     plt.show()
    
    
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
          accuracy = evaluate_logistic_regression_model(probe, test_data, labels, split_idx)
          results_df.loc[jailbreak_1, jailbreak_2] = accuracy
      elif probe_type == "MLP":
          accuracy = evaluate_mlp_model(probe, test_data, labels, split_idx)
          results_df.loc[jailbreak_1, jailbreak_2] = accuracy        
    
  return results_df




# def holdout_transfer_graph(holdout_df, dataset_name, model_name, layer, probe_type, savepath=None):
#     # Reset the index
#     holdout_df = holdout_df.reset_index()

#     # Rename the index column to 'technique'
#     holdout_df = holdout_df.rename(columns={'index': 'Technique'})

#     # Select relevant columns
#     holdout_df = holdout_df[['Technique', 'Train Set', 'Test Set (Holdout)']]

#     # Sort by the 'Test Set (Holdout)' column in ascending order
#     # holdout_df = holdout_df.sort_values(by='Test Set (Holdout)', ascending=True)

#     # Set 'technique' as the index
#     holdout_df.set_index('Technique', inplace=True)

#     # Format the y-tick labels: replace '_' with ' ' and title case
#     holdout_df.index = [label.replace('_', ' ').title() for label in holdout_df.index]

#     # Plot the bar plot with adjusted bar height and width
#     ax = holdout_df.plot(kind='barh', figsize=(15, 12), color=['skyblue', 'salmon'], linewidth=0.5, width=0.85)

#     # Add titles and labels
#     plt.xlabel('Accuracy', fontsize=fontsize2)
#     plt.ylabel('Held out Category', fontsize=fontsize2)
#     plt.xticks(rotation=45, ha="right", fontsize=fontsize3)
#     plt.yticks(fontsize=fontsize3)

#     # Set the plot title
#     plt.figtext(0.5, -0.1, f'{probe_type} Accuracy by Holdout Type\n {model_name.replace("-", " ").title()} (Layer {layer})', ha='center', fontsize=fontsize1, style='italic')

#     # Move the legend underneath the graph
#     plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=fontsize2, ncol=2)

#     # Add a red vertical line at 0.50 for accuracy threshold
#     plt.axvline(x=0.50, color='red', linestyle='--', linewidth=2, label="Threshold 0.50")

#     # Adjust layout to prevent overlap
#     plt.tight_layout(pad=1.0)

#     # Set the x-axis limits from 0 to 1
#     plt.xlim(0, 1)

#     # Show or save the plot
#     if savepath:
#         plt.savefig(savepath, bbox_inches='tight')

#     plt.show()

def holdout_transfer_graph(holdout_df, dataset_name, model_name, layer, probe_type, mean_tranfer_score = None, savepath=None):
    # Reset the index
    holdout_df = holdout_df.reset_index()

    # Rename the index column to 'Technique'
    holdout_df = holdout_df.rename(columns={'index': 'Technique'})

    # Select relevant columns
    holdout_df = holdout_df[['Technique', 'Train Set', 'Test Set (Holdout)', 'Test Set SE', 'Test Size']]

    # Set 'Technique' as the index
    holdout_df.set_index('Technique', inplace=True)

    # Format the y-tick labels: replace '_' with ' ' and title case
    holdout_df.index = [label.replace('_', ' ').title().replace("Gcg", "GCG").replace("Aim", "AIM") for label in holdout_df.index]

    # Extract accuracy values and standard errors
    test_accuracy = holdout_df['Test Set (Holdout)']
    train_accuracy = holdout_df['Train Set']
    test_se = holdout_df['Test Set SE']
    test_size = holdout_df['Test Size']

    # Plot the bar plot with adjusted bar height and width
    ax = holdout_df[['Train Set', 'Test Set (Holdout)']].plot(kind='barh', 
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
    plt.figtext(0.5, -0.1, f'Mean Test Accuracy {mean_tranfer_score}\n{model_name.replace("-", " ").title()} (Layer {layer})', ha='center', fontsize=fontsize1, style='italic')

    # Move the legend underneath the graph
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), fontsize=fontsize2, ncol=2)

    # Add a red vertical line at 0.50 for accuracy threshold
    plt.axvline(x=0.50, color='red', linestyle='--', linewidth=2, label="Threshold 0.50")

    # Adjust layout to prevent overlap
    plt.tight_layout(pad=5.0)

    # Set the x-axis limits from 0 to 1
    plt.xlim(0, 1)

    # Show or save the plot
    if savepath:
        plt.savefig(savepath, bbox_inches='tight')

    plt.show()





def transfer_with_holdout(df, probe_type, layer):
  
  jailbreak_dfs = split_dataset_by_jailbreak(df)
   

  holdout_df = pd.DataFrame(columns=["Train Set", "Test Set (Holdout)"] , index=jailbreak_dfs.keys())
  

  for i in df['jailbreak_prompt_name'].unique():
    df_holdout = df[df['jailbreak_prompt_name'] != i]
    print("Holdout: ", i)
    print(df_holdout['jailbreak_prompt_name'].unique())
  
    
    # shuffle and balance all dfs
    df_holdout = balance_dataset(df_holdout, "rating_binary")
    df_holdout["rating_binary"].value_counts()

    # shuffle dataset
    df_holdout = df_holdout.sample(frac=1).reset_index(drop=True)

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
      test_accuracy = evaluate_logistic_regression_model(probe, test_data, labels, split_idx)
    elif probe_type == "MLP":
      test_accuracy = evaluate_mlp_model(probe, test_data, labels, split_idx)

    holdout_df.loc[i, "Test Set (Holdout)"] = test_accuracy
    print(test_accuracy)
  
  return holdout_df
  

def transfer_with_holdout_with_errors(df, probe_type, layer):
  
    jailbreak_dfs = split_dataset_by_jailbreak(df)
    holdout_df = pd.DataFrame(columns=["Train Set", "Test Set (Holdout)", "Test Set SE", "Test Size"], index=jailbreak_dfs.keys())
  
    for i in df['jailbreak_prompt_name'].unique():
        df_holdout = df[df['jailbreak_prompt_name'] != i]
        print("Holdout: ", i)
    
        # Shuffle and balance all dfs
        print("len before balancing", len(df_holdout))
        df_holdout = balance_dataset(df_holdout, "rating_binary")
        print("len after balancing", len(df_holdout))

        df_holdout = df_holdout.sample(frac=1).reset_index(drop=True)
        split_idx = int(0.8 * len(df_holdout))
        labels = df_holdout["rating_binary"].to_list()
        train_data = df_holdout[f'layer_{layer}'].to_list()
        
        if probe_type == "logistic_regression":
            probe, train_accuracy = create_logistic_regression_model(train_data, labels, split_idx)
        elif probe_type == "MLP": 
            probe, train_accuracy = create_mlp_model(train_data, labels, split_idx)

        holdout_df.loc[i, "Train Set"] = train_accuracy
        print(train_accuracy)

        df_holdout_test = df[df['jailbreak_prompt_name'] == i]

        # Shuffle and balance dataset
        print("test len before balancing", len(df_holdout_test))
        test_df = balance_dataset(df_holdout_test, "rating_binary")
        print("test len after balancing", len(test_df))
        test_df = test_df.sample(frac=1).reset_index(drop=True)
        split_idx = int(0.8 * len(test_df))
        labels = test_df["rating_binary"].to_list()
        test_data = test_df[f'layer_{layer}'].to_list()

        if probe_type == "logistic_regression":
            test_accuracy = evaluate_logistic_regression_model(probe, test_data, labels, split_idx)
        elif probe_type == "MLP":
            test_accuracy = evaluate_mlp_model(probe, test_data, labels, split_idx)

        # Calculate standard error for the test accuracy
        n_test_samples = len(labels)  # number of samples in the test set
        test_se = np.sqrt((test_accuracy * (1 - test_accuracy)) / n_test_samples)

        holdout_df.loc[i, "Test Set (Holdout)"] = test_accuracy
        holdout_df.loc[i, "Test Set SE"] = test_se
        holdout_df.loc[i, "Test Size"] = n_test_samples
        
        print(f"Test Accuracy: {test_accuracy}, SE: {test_se}")
  
    return holdout_df


def transfer_heatmap(results_df, dataset_name, model_name, probe_type, savepath = None):
    
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
                        ha='center', va='center', color='black', fontsize=fontsize3)
                
    plt.xticks(fontsize=fontsize3)
    plt.yticks(fontsize=fontsize3)

    plt.xlabel("Test Data", fontsize=fontsize2)
    plt.ylabel("Training Data", fontsize=fontsize2)
    plt.title(f"Heatmap of {probe_type} Accuracy {dataset_name} for {model_name}", fontsize=fontsize1, pad=20)
    
    
    if savepath: 
        plt.savefig(savepath)
    
    plt.show()
    


  
 

