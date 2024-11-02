import torch 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import torch.optim as optim
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        # Input layer to hidden layer with 8 neurons, process each of the 8 sequences separately
        self.fc1 = nn.Linear(3072, 8)
        self.relu = nn.ReLU()
        # Process the output from the 8 sequences, reduce to a single output
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
      
    def train_MLP(self, data, labels, split_idx, num_epochs = 500):
      self.train() 
      
      X_train = self.scaler.fit_transform(data[:split_idx])
      y_train = labels[:split_idx]


      X_train = torch.tensor(X_train, dtype=torch.float32)
      y_train = torch.tensor(y_train, dtype=torch.float32)
  

      # Define a loss function
      # criterion = nn.MSELoss()
      criterion = nn.BCEWithLogitsLoss()
      # criterion = nn.BCELoss()

      # Define an optimizer (e.g., Stochastic Gradient Descent)
      optimizer = optim.SGD(self.parameters(), lr=0.01)

      for epoch in range(num_epochs):
          # self.train()
          y_pred = self(X_train)

          losses = []
          
          # Compute and print loss
          loss = criterion(y_pred, y_train)
          losses.append(loss.item())
          
          # Calculate accuracy
          accuracies = []
          y_pred_class = torch.sigmoid(y_pred).squeeze() > 0.5  # Sigmoid and threshold for binary classification
          accuracy = accuracy_score(y_train.cpu(), y_pred_class.cpu())  # Calculate accuracy
          accuracies.append(accuracy)
          
          
          #optimizer.zero_grad()  # Zero the gradients
          loss.backward()        # Backward pass: compute gradient of the loss with respect to model parameters
          optimizer.step()       # Update the model parameters

          if (epoch+1) % 10 == 0:  # Print every 10 epochs
              print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

      print("Training complete.")
      return losses, accuracies


    def get_accuracy(self, data, labels, split_idx):
        X_test = data[split_idx:]
        y_test = labels[split_idx:]

        # Apply scaler if it was fitted during training
        if hasattr(self.scaler, 'mean_'):
            print("Applying fitted scaler to test data...")
            X_test = self.scaler.transform(X_test)
        else:
            print("No fitted scaler found, using raw data.")

        # Convert to torch tensors
        X_test = torch.tensor(X_test, dtype=torch.float32)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        # Set the model to evaluation mode
        self.eval()

        # Make predictions
        with torch.no_grad():  # Disable gradient calculation
            y_pred = self(X_test)

            # Apply sigmoid to get probabilities
            y_probabilities = torch.sigmoid(y_pred)

            # Convert probabilities to binary labels (threshold at 0.5)
            y_pred_labels = (y_probabilities > 0.5).float()

        # Convert to numpy for compatibility with sklearn's accuracy_score
        y_test_np = y_test.numpy()
        y_pred_labels_np = y_pred_labels.numpy()

        # Calculate accuracy
        self.accuracy = accuracy_score(y_test_np, y_pred_labels_np)
        print(f'Accuracy: {self.accuracy:.4f}')
        return self.accuracy
    
    
    def load_model(self, file_path):
      self.load_state_dict(torch.load(file_path))
      self.eval()  # Set to evaluation mode
      print(f"Model weights loaded from {file_path}!")
      
    def save_model(self, file_path):
      torch.save(self.state_dict(), file_path)
      print(f"Model weights saved to {file_path}!")