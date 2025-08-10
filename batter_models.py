import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy

class ffnn(nn.Module):
    
    """ 
    Description:
        a class to define a feed-forward neural network that works with predicting offensive outcomes (without embeddings) 
    """
   
    def __init__(self, input_dim: int, num_classes=1):
        
        """
        Description:
            Initializes the architecture for the feed-forward neural network.
            
        Inputs:
            input_dim (int): how many columns are in the input (X) data
            num_classes (int): how many outputs are expected
        """
              
        super().__init__()
        
        # define model architecture
        self.model = nn.Sequential(
            
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 32),
            nn.ReLU(),  

            nn.Linear(32, num_classes)  
        )
    
    def forward(self, X):
        """
        Description:
            Defines the behavior for the forward pass.
            
        Inputs:
            X (torch.tensor): the raw input data
            
        Outputs:
            output (torch.tensor): the result from passing the data though the model
        """

        # pass input data through model architecture 
        output = self.model(X)

        return output


    
    
def train(model, train_loader, test_loader, epochs=50, patience=10, loss_fn=None, optimizer=None):
    """
    Description:
        Trains a provided feed-forward neural network using provided training and testing data loaders
        
    Inputs:
        model (batter_models.ffnn): the input model, which has the same architecture as defined in batter_models.ffnn
        train_loader (torch.DataLoader): the training data, in DataLoader format
        test_loader (torch.DataLoader): the testing data, in DataLoader format
        epochs (int): how many times the training loop should iterate through the DataLoaders
        patience (int): how many epochs the training loop will wait before terminating without an improvement in validation loss
        loss_fn: the function that will compute the loss
        optimizer: the optimizer that should be used by the training loop
    """
    
    # auto fills the loss function and optimizer if not specified
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()
    
    if optimizer is None:
        optimizer = optim.Adam(model.parameters(), lr=0.001)

    # initialize lists to store the training and validation losses
    train_losses, val_losses = [], []
    
    # initialize variables to help with early stopping and version caching
    best_val_loss = float('inf')
    best_model_weights = deepcopy(model.state_dict())
    epochs_no_improve = 0

    for epoch in range(epochs):

        ### training portion ###
        
        model.train() # set model to train mode
        
        cumulative_train_loss = 0
        for X_batch, y_batch in train_loader:
            
            optimizer.zero_grad() # get gradients
            outputs = model(X_batch) # get predictions
            loss = loss_fn(outputs, y_batch) # compute loss and back propogate
            loss.backward() # adjust weights
            optimizer.step() # update gradients
            cumulative_train_loss += loss.item() * X_batch.size(0) # add to the loss

        train_losses.append(cumulative_train_loss/len(train_loader.dataset)) # save train loss for the epoch
        
        ### evaluation portion ###
        
        model.eval() # set model to evaluation mode
        
        cumulative_val_loss = 0
        with torch.no_grad():
            for X_val, y_val in test_loader:
                y_pred = model(X_val) # get predictions
                val_loss = loss_fn(y_pred, y_val)  # calculate loss of predictions
                cumulative_val_loss += val_loss.item() * X_val.size(0) # scale loss and add to cumulative loss
                
        val_losses.append(cumulative_val_loss / len(test_loader.dataset))  # save validation loss for the epoch
        
        # check if the most recent validation loss is better than the best seen, and update early stopping variables accordingly
        if val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            best_model_weights = deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
          
        # print the loss to the console
        print(f"Epoch {epoch+1}/{epochs} - Training Loss: {train_losses[-1]:.4f} - Validation Loss: {val_losses[-1]:.4f}")
        
        # checks if the early stopping condition is met
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break
        
    # revert the model to the state that produced the best validation score
    model.load_state_dict(best_model_weights)
        
    return train_losses, val_losses




if __name__=="__main__":
    pass
