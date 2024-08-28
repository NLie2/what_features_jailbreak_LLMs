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

    log_model.fit(x_train, y_train)
    
    
    x_test = data[split_idx:]
    y_test = labels[split_idx:]
    
    accuracy = log_model.score(x_test, y_test)

    return log_model, accuracy


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


def plot_confusion_matrix(conf_matrix, class_names, save=False, dataset_name =  ""):
    '''
    input: conf_matrix (array), class_names (list of str)
    output: None
    '''
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

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