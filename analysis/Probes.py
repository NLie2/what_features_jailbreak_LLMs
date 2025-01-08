import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import torch.optim as optim
import torch.nn as nn
import numpy as np
import math



class SimpleMLP(nn.Module):
    def __init__(self, input_size=3072):
        """
        Parameters:
        - input_size (int): The size of the input features.
        """
        super(SimpleMLP, self).__init__()
        
        # Input layer to hidden layer with 8 neurons
        self.fc1 = nn.Linear(input_size, 8)
        self.relu = nn.ReLU()
        
        # Hidden layer to output layer
        self.fc2 = nn.Linear(8, 1)
        
        self.scaler = StandardScaler()
        self.accuracy = 0

    def forward(self, x):
        # x is expected to have shape [1, 8, 3072]
        # Process each of the 8 vectors separately through fc1
        x = self.fc1(x)  # Shape: [1, 8, 8]
        x = self.relu(x)
        x = self.fc2(x)  # Shape: [1, 8, 1]

        # Aggregate the outputs across the sequence dimension (dim=1)
        x = torch.mean(x, dim=1)  # Shape: [1, 1], now a scalar for each batch
        
        return x
      
    def train_MLP(self, train_data, train_labels, num_epochs = 500):
      device = next(self.parameters()).device
      
      self.train() 
      
      X_train = self.scaler.fit_transform(train_data)
      # y_train = labels[:split_idx]


      X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
      y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
  

      # Define a loss function
      # criterion = nn.MSELoss()
      criterion = nn.BCEWithLogitsLoss()
      # criterion = nn.BCELoss()

      # Define an optimizer (e.g., Stochastic Gradient Descent)
      optimizer = optim.SGD(self.parameters(), lr=0.01)

      losses = []
      accuracies = []
          
      for epoch in range(num_epochs):
          # self.train()
          y_pred = self(X_train)
          
          # Compute and print loss
          loss = criterion(y_pred, y_train)
          losses.append(loss.item())
          

          y_pred_class = torch.sigmoid(y_pred).squeeze() > 0.5  # Sigmoid and threshold for binary classification
          accuracy = accuracy_score(y_train.cpu(), y_pred_class.cpu().numpy())  # Calculate accuracy
          accuracies.append(accuracy)
          
          
          #optimizer.zero_grad()  # Zero the gradients
          optimizer.zero_grad()
          loss.backward()        # Backward pass: compute gradient of the loss with respect to model parameters
          optimizer.step()       # Update the model parameters

          if (epoch+1) % 10 == 0:  # Print every 10 epochs
              print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

      print("Training complete.")
      return losses, accuracies


    # def get_accuracy(self, test_data, test_labels):
      
    #     device = next(self.parameters()).device
        
    #     # Apply scaler if it was fitted during training
    #     if hasattr(self.scaler, 'mean_'):
    #         print("Applying fitted scaler to test data...")
    #         X_test = self.scaler.transform(test_data)
    #     else:
    #         print("No fitted scaler found, using raw data.")
    #         X_test = test_data

    #     # Convert to torch tensors
    #     X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    #     y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

    #     # Set the model to evaluation mode
    #     self.eval()

    #     # Make predictions
    #     with torch.no_grad():  # Disable gradient calculation
    #         y_pred = self(X_test)

    #         # Apply sigmoid to get probabilities
    #         y_probabilities = torch.sigmoid(y_pred)

    #         # Convert probabilities to binary labels (threshold at 0.5)
    #         y_pred_labels = (y_probabilities > 0.5).float()

    #     # Convert to numpy for compatibility with sklearn's accuracy_score
    #     y_test_np = y_test.cpu().numpy()
    #     y_pred_labels_np = y_pred_labels.cpu().numpy()

    #     # Calculate accuracy
    #     self.accuracy = accuracy_score(y_test_np, y_pred_labels_np)
    #     print(f'Accuracy: {self.accuracy:.4f}')
    #     return self.accuracy
    

    #     # Apply scaler if it was fitted during training
    #     if hasattr(self.scaler, 'mean_'):
    #         print("Applying fitted scaler to test data...")
    #         X_test = self.scaler.transform(test_data)
    #     else:
    #         print("No fitted scaler found, using raw data.")

    #     # Convert to torch tensors
    #     X_test = torch.tensor(X_test, dtype=torch.float32)
    #     y_test = torch.tensor(test_labels, dtype=torch.float32)

    #     # Set the model to evaluation mode
    #     self.eval()

    #     # Make predictions
    #     with torch.no_grad():  # Disable gradient calculation
    #         y_pred = self(X_test)

    #         # Apply sigmoid to get probabilities
    #         y_probabilities = torch.sigmoid(y_pred)

    #         # Convert probabilities to binary labels (threshold at 0.5)
    #         y_pred_labels = (y_probabilities > 0.5).float()

    #     # Convert to numpy for compatibility with sklearn's accuracy_score
    #     y_test_np = y_test.numpy()
    #     y_pred_labels_np = y_pred_labels.numpy()

    #     # Calculate accuracy
    #     self.accuracy = accuracy_score(y_test_np, y_pred_labels_np)
    #     print(f'Accuracy: {self.accuracy:.4f}')
    #     return self.accuracy
    
    def get_accuracy(self, test_data, test_labels, verbose = False):
      
      print(len(test_data), len(test_labels))
      device = next(self.parameters()).device
      
      X_test = self.scaler.transform(test_data) if hasattr(self.scaler, 'mean_') else test_data
      X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
      y_test = torch.tensor(test_labels, dtype=torch.float32).to(device)

      
      print(len(X_test), len(y_test))
      self.eval()
      with torch.no_grad():
          y_pred = self(X_test)
          y_pred_labels = (torch.sigmoid(y_pred) > 0.5).float()

      y_test_np = y_test.cpu().numpy()
      y_pred_labels_np = y_pred_labels.cpu().numpy()

      self.accuracy = accuracy_score(y_test_np, y_pred_labels_np)
      conf_matrix = confusion_matrix(y_test_np, y_pred_labels_np)
      
      if verbose:
        plt.figure(figsize=(8,6))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
        
        print(f'Accuracy: {self.accuracy:.4f}')
        
      return self.accuracy, conf_matrix
    
    def load_model(self, file_path, map_location=None):
      self.load_state_dict(torch.load(file_path, map_location=map_location))
      self.eval()
      print(f"Model weights loaded from {file_path}!")
      
    def save_model(self, file_path):
      torch.save(self.state_dict(), file_path)
      print(f"Model weights saved to {file_path}!")
      
    
    

########TRANSFORMER PROBE########

# class TransformerProbe(nn.Module):
#     def __init__(self, input_size=4096, nhead=8, num_layers=2):
#         super(TransformerProbe, self).__init__()
#         self.input_size = input_size
#         self.d_model = 256
#         self.seq_len = 11
        
#         # First project the input to a size divisible by seq_len
#         self.input_projection = nn.Linear(input_size, self.seq_len * self.d_model)
#         self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, self.d_model))
        
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=self.d_model,
#             nhead=nhead,
#             dim_feedforward=512,
#             dropout=0.1,
#             batch_first=True
#         )
        
#         self.transformer_encoder = nn.TransformerEncoder(
#             encoder_layer,
#             num_layers=num_layers
#         )
        
#         self.fc = nn.Linear(self.d_model, 1)
#         self.scaler = StandardScaler()
#         self.accuracy = 0
#         self.cuda()
   
    # def forward(self, x):
    #     batch_size = x.shape[0]
        
    #     # Project and reshape to [batch_size, seq_len, d_model]
    #     x = self.input_projection(x)
    #     x = x.view(batch_size, self.seq_len, self.d_model)
        
    #     # Add positional encoding
    #     x = x + self.pos_encoder
        
    #     # Pass through transformer
    #     x = self.transformer_encoder(x)
        
    #     # Average pooling
    #     x = torch.mean(x, dim=1)
        
    #     # Final classification
    #     x = self.fc(x)
    #     return x

    # def train_probe(self, train_data, train_labels, num_epochs=500):
    #     # Check if CUDA is available and set device
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     print(f"Using device: {device}")
        
    #     # Ensure model is on the correct device
    #     self = self.to(device)
    #     self.train()
        
    #     X_train = self.scaler.fit_transform(train_data)
    #     # Move data to device
    #     X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    #     y_train = torch.tensor(train_labels, dtype=torch.float32).to(device)
        
    #     criterion = nn.BCEWithLogitsLoss().to(device)
    #     optimizer = optim.AdamW(self.parameters(), lr=0.001)
   
class TransformerProbe(nn.Module):
    def __init__(self, input_size=4096, nhead=8, num_layers=2):
        super(TransformerProbe, self).__init__()
        self.device = torch.device("cuda")
        self.input_size = input_size
        
        # Adjust dimensions to be more GPU-friendly
        self.nhead = nhead
        # Make d_model larger and ensure it's divisible by 64 (typical GPU warp size)
        self.d_model = max(512, ((input_size // 32) // 64) * 64)
        # Keep seq_len a power of 2 but smaller
        self.seq_len = 8  # Fixed to 8 for better GPU alignment
        
        print(f"Dimensions: input_size={input_size}, d_model={self.d_model}, "
              f"seq_len={self.seq_len}, nhead={self.nhead}")
        
        self.input_projection = nn.Linear(input_size, self.seq_len * self.d_model).cuda()
        self.pos_encoder = nn.Parameter(torch.randn(1, self.seq_len, self.d_model, device=self.device))
        
        # Adjust layer normalization epsilon for better numerical stability
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.d_model * 4,
            dropout=0.1,
            batch_first=True,
            norm_first=False,  # Changed back to False
            layer_norm_eps=1e-5
        ).cuda()
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        ).cuda()
        
        self.fc = nn.Linear(self.d_model, 1).cuda()
        self.scaler = StandardScaler()
        self.accuracy = 0
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Project and reshape with explicit memory contiguous operation
        x = self.input_projection(x)
        x = x.view(batch_size, self.seq_len, self.d_model).contiguous()
        
        # Add positional encoding
        x = x + self.pos_encoder
        
        # Pass through transformer
        x = self.transformer_encoder(x)
        
        # Average pooling with explicit contiguous operation
        x = torch.mean(x.contiguous(), dim=1)
        
        # Final classification
        x = self.fc(x)
        return x

    def train_probe(self, train_data, train_labels, num_epochs=500):
        self.train()
        
        # Data preparation
        X_train = self.scaler.fit_transform(train_data)
        X_train = torch.tensor(X_train, dtype=torch.float32).cuda()
        y_train = torch.tensor(train_labels, dtype=torch.float32).cuda()
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        
        criterion = nn.BCEWithLogitsLoss().cuda()
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        
        try:
            for epoch in range(num_epochs):
                optimizer.zero_grad()
                y_pred = self(X_train)
                loss = criterion(y_pred.squeeze(), y_train)
                loss.backward()
                optimizer.step()
                
                if (epoch + 1) % 10 == 0:
                    with torch.no_grad():
                        y_pred_class = (torch.sigmoid(y_pred).squeeze() > 0.5).cpu()
                        accuracy = accuracy_score(y_train.cpu(), y_pred_class)
                        print(f'Epoch [{epoch+1}/{num_epochs}], '
                              f'Loss: {loss.item():.4f}, '
                              f'Accuracy: {accuracy:.4f}')
        except RuntimeError as e:
            print("Error during training!")
            print(f"X_train device: {X_train.device}")
            print(f"Model device: {next(self.parameters()).device}")
            print(f"Batch size: {X_train.shape[0]}")
            raise e
      
      
    def get_accuracy(self, test_data, test_labels):
        if hasattr(self.scaler, 'mean_'):
            X_test = self.scaler.transform(test_data)
            # Move data to CUDA
            X_test = torch.tensor(X_test, dtype=torch.float32).cuda()
            y_test = torch.tensor(test_labels, dtype=torch.float32).cuda()
            
            self.eval()
            with torch.no_grad():
                y_pred = self(X_test)
                y_probabilities = torch.sigmoid(y_pred)
                y_pred_labels = (y_probabilities > 0.5).float()
                
                # Move to CPU for sklearn metrics calculation
                y_test_cpu = y_test.cpu().numpy()
                y_pred_cpu = y_pred_labels.squeeze().cpu().numpy()
                
                # Calculate accuracy and confusion matrix
                self.accuracy = accuracy_score(y_test_cpu, y_pred_cpu)
                conf_matrix = confusion_matrix(y_test_cpu, y_pred_cpu)
                
                print(f'Accuracy: {self.accuracy:.4f}')
                print('\nConfusion Matrix:')
                print(conf_matrix)
                
                return self.accuracy, conf_matrix
              
    def save_probe(self, save_path):
        """
        Save the trained probe model and scaler parameters.
        
        Args:
            save_path (str): Path where the model should be saved
                            (without file extension)
        """
        # Create a dictionary containing all necessary components
        save_dict = {
            'model_state_dict': self.state_dict(),
            'scaler_mean': self.scaler.mean_,
            'scaler_scale': self.scaler.scale_,
            'scaler_var': self.scaler.var_,
            'input_size': self.input_size,
            'accuracy': self.accuracy
        }
        
        # Save the dictionary
        torch.save(save_dict, f"{save_path}.pth")
        print(f"Model saved to {save_path}.pth")

    @classmethod
    def load_probe(cls, load_path):
        """
        Load a saved probe model.
        
        Args:
            load_path (str): Path to the saved model file
                            (without file extension)
        
        Returns:
            TransformerProbe: Loaded model with restored weights and scaler
        """
        # Load the saved dictionary
        save_dict = torch.load(f"{load_path}")
        
        # Create a new instance of the model
        model = cls(input_size=save_dict['input_size'])
        
        # Load the model state dictionary
        model.load_state_dict(save_dict['model_state_dict'])
        
        # Restore the scaler parameters
        model.scaler.mean_ = save_dict['scaler_mean']
        model.scaler.scale_ = save_dict['scaler_scale']
        model.scaler.var_ = save_dict['scaler_var']
        model.accuracy = save_dict['accuracy']
        
        return model