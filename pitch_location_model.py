import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
from copy import deepcopy
import torch.nn.functional as F

def bivariate_loss(x, z, mu_x, mu_z, sigma_x, sigma_z, rho):
    """
    Description:
        defines a custom loss function for bivariate loss
        
    Inputs:
        x (torch.tensor): x coordinates of pitches
        z (torch.tensor): z coordinates of pitches
        mu_x (torch.tensor): the mean x coordinates of a distribution
        mu_z (torch.tensor): the mean z coordinates of a distribution
        sigma_x (torch.tensor): the standard deviation of x coordinates of a distribution
        sigma_z (torch.tensor): the standard deviation of z coordinates of a distribution
        rho (torch.tensor): the correlation between the x and z coordinates in a distribution
    
    Outputs:
        loss (torch.tensor): the calculated loss for each instance fed into the function
        
    """
    
    dx = (x.unsqueeze(-1) - mu_x) / sigma_x
    dz = (z.unsqueeze(-1) - mu_z) / sigma_z
    
    one_minus_r2 = (1 - rho**2).clamp(1e-6, 1.0)
    z_term = (dx**2 - 2*rho*dx*dz + dz**2) / one_minus_r2
    log_norm = torch.log(2 * torch.pi * sigma_x * sigma_z * torch.sqrt(one_minus_r2))
    
    loss = -0.5 * z_term - log_norm
    
    return loss

def mdn_nll(x, z, mu_x, mu_z, sx, sz, rho, pi):
    
    log_prob_k = bivariate_loss(x, z, mu_x, mu_z, sx, sz, rho) 
    log_mix = torch.log(pi.clamp_min(1e-9)) + log_prob_k       
    log_sum = torch.logsumexp(log_mix, dim=-1)   
    
    loss = -log_sum.mean()
    
    return loss

class PitchLocDataset(torch.utils.data.Dataset):
    """
    Description:
        A class that builds a custom PyTorch Dataset that stores pitcher IDs and pitch types in addition to X and y features.
    """
    def __init__(self, X, pitcher_ids, pitch_types, y):
        """
        Description:
            Creates a PitchTypeDataset object
            
        Inputs:
            X (torch.tensor): a tensor of numeric X features
            pitcher_ids (torch.tensor): a tensor of pre-encoded pitcher IDs
            pitch_types (torch.tensor): a tensor of pre-encoded pitch types
            y (torch.tensor): a tensor of target y features
        
        Outpus:
            None
        """
        
        self.X = X
        self.pitcher_ids = pitcher_ids
        self.pitch_types = pitch_types
        self.y = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.pitcher_ids[idx], self.pitch_types[idx], self.y[idx]

class ffnn(nn.Module):
    
    """ 
    Description:
        a class to define a feed-forward neural network that works with predicting x and z coordinates of a certain pitch type 
    """
   
    def __init__(self, input_dim, num_pitchers, num_pitch_types, num_classes=1, pitcher_emb_dim=4, pitch_type_emb_dim=8):
        
        """
        Description:
            Initializes the architecture for the feed-forward neural network.
            
        Inputs:
            input_dim (int): how many columns are in the input (X) data
            num_pitchers (int): how many pitchers are in the dataset
            num_pitch_types (int): how many pitch types are in the dataset
            num_classes (int): how many outputs are expected
            pitcher_embedding_dim (int): how many dimensions each pitcher ID should be split into
            pitch_type_emb_dim (int): how many dimensions each pitch type should be split into
            
        """
        
        # update the input dimension to account for the size of the pitcher embeddings and the pitch_type embeddings
        input_dim += pitcher_emb_dim + pitch_type_emb_dim
              
        super().__init__()
        
        # create embedding layers
        self.pitcher_embedding = nn.Embedding(num_embeddings=num_pitchers, embedding_dim=pitcher_emb_dim)
        self.pitch_type_embedding = nn.Embedding(num_embeddings=num_pitch_types, embedding_dim=pitch_type_emb_dim)
        
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
    
    def forward(self, X, pitcher_id, pitch_type):
        
        """
        Description:
            Defines the behavior for the forward pass.
            
        Inputs:
            X (torch.tensor): the raw input data
            pitcher_id (torch.tensor): the pitcher IDs corresponding to each row of the raw input data
            pitch_type (torch.tensor): the pitch type corresponding to each row of the raw input data
            
        Outputs:
            output (torch.tensor): the result from passing the data though the model
        """
        
        # get embeddings for the pitcher ID and pitch type
        pitcher_emb = self.pitcher_embedding(pitcher_id)
        pitch_type_emb = self.pitch_type_embedding(pitch_type)
        
        # add embeddings to the raw X data
        X_embeddings = torch.cat([X, pitcher_emb, pitch_type_emb], dim=1)
        
        # pass input data through model architecture 
        output = self.model(X_embeddings)
        
        return output
    
        

def train(model, train_loader, test_loader, epochs=50, patience=10, loss_fn=None, optimizer=None):
    """
    Description:
        Trains a provided feed-forward neural network using provided training and testing data loaders
        
    Inputs:
        model (batter_models.ffnn): the input model, which has the same architecture as defined in pitcher_location_model.ffnn
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
        for X_batch, pitchers_batch, pitch_types_batch, y_batch in train_loader:
            
            optimizer.zero_grad() # get gradients
            outputs = model(X_batch, pitchers_batch, pitch_types_batch) # get predictions
            loss = loss_fn(outputs, y_batch) # compute loss and back propogate
            loss.backward() # adjust weights
            optimizer.step() # update gradients
            cumulative_train_loss += loss.item() * X_batch.size(0) # add to the loss

        train_losses.append(cumulative_train_loss/len(train_loader.dataset)) # save train loss for the epoch
        
        ### evaluation portion ###
        
        model.eval() # set model to evaluation mode
        
        cumulative_val_loss = 0
        with torch.no_grad():
            for X_val, pitchers_val, pitch_types_val, y_val in test_loader:
                
                # get predictions
                y_pred = model(X_val, pitchers_val, pitch_types_val)
                
                # calculate loss of predictions
                val_loss = loss_fn(y_pred, y_val)
                
                # scale loss and add to cumulative loss
                cumulative_val_loss += val_loss.item() * X_val.size(0) 
                
        val_losses.append(cumulative_val_loss / len(test_loader.dataset)) # save validation loss for the epoch
        
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