import pandas as pd
import numpy as np
from helper_functions import adjust_stat
from copy import copy
import pickle
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.nn.functional as F
from multiprocessing import Pool, cpu_count

def sample_pitch_type(row, models):
    """
    Take in various features about a pitcher/batter matchup and use a pre-trained pitch type model to sample the next pitch type that the pitcher will throw next. 
    
    Inputs:
        row (dict): various informatoin regarding a specific pitcher/batter matchup
        models (dict): a dict of dicts that stores various objects in a tree structure
        
    Outputs:
        none (function modifies row in place)
    """

    # load all necessary scalers, label encoders, and models
    ptm_scaler = models['scalers']['ptm_scaler']
    ptm_pitcher_le = models['label_encoders']['ptm_pitcher_le']
    ptm_pitch_type_le = models['label_encoders']['ptm_pitch_type_le']
    pitch_type_model = models['models']['pitch_type_model']

    # extract and scale the X features and encode the pitcher ID
    X = ptm_scaler.transform([[row[feat] for feat in models['features']['pitch_X_feats']]])
    p_enc = ptm_pitcher_le.transform([row['pitcher']])

    # convert the X array and encoded pitcher ID into tensors
    X_tensor = torch.from_numpy(X).float()
    p_enc_tensor = torch.tensor(p_enc, dtype=torch.int32)

    # run the model
    pitch_type_logits = pitch_type_model(X_tensor, p_enc_tensor).detach()
    
    # eliminates the possibility of sampling any pitch that the pitcher does not throw
    has_ptype_mask = torch.tensor([row[f'has_{ptype}'] for ptype in ptm_pitch_type_le.classes_], dtype=torch.bool)
    pitch_type_logits[0][~has_ptype_mask] = float('-inf')

    # convert the raw logits to probabilities and samples from said probabilities
    pitch_type_probs = F.softmax(pitch_type_logits, dim=1)  
    sampled_idx = torch.multinomial(pitch_type_probs, num_samples=1).item()
    
    # add the sampled pitch type to the row
    row['pitch_type'] = ptm_pitch_type_le.classes_[sampled_idx]
    
    # no return (modifies dict in place)
    
def sample_pitch_location(row, models, data): 
    """
    Take in various features about a pitcher/batter matchup and use a pre-trained pitch location model to sample the location and pitch characteristics that the pitcher will throw next. 
    
    Inputs:
        row (dict): various informatoin regarding a specific pitcher/batter matchup
        models (dict): a dict of dicts that stores various objects in a tree structure
        data (dict): stores various dataframes containing processed data
        
    Outputs:
        none (function modifies row in place)
    """
    
    # load all necessary data, scalers, label encoders, and models
    pitch_characteristics_df = data['pitch_characteristics_df']
    plm_scaler = models['scalers']['plm_scaler']
    plm_pitcher_le = models['label_encoders']['plm_pitcher_le']
    plm_pitch_type_le = models['label_encoders']['plm_pitch_type_le']
    pitch_location_model = models['models']['pitch_location_model']

    # extract and scale the X features and encode both the pitcher ID and pitch type
    X = plm_scaler.transform([[row[feat] for feat in models['features']['pitch_X_feats']]])
    p_enc = plm_pitcher_le.transform([row['pitcher']])
    pt_enc = plm_pitch_type_le.transform([row['pitch_type']])

    # convert the X array, encoded pitcher ID, and encoded pitch type into tensors
    X_tensor = torch.from_numpy(X).float()
    p_enc_tensor = torch.tensor(p_enc, dtype=torch.int32)
    pt_enc_tensor = torch.tensor(pt_enc, dtype=torch.int32)

    # run the model
    pitch_location_logits = pitch_location_model(X_tensor, p_enc_tensor, pt_enc_tensor).detach()
    
    # convert the raw logits to probabilities and samples from said probabilities
    pitch_location_probs = F.softmax(pitch_location_logits, dim=1)  
    sampled_idx = torch.multinomial(pitch_location_probs, num_samples=1).item()
    
    # define pitch zone dimensions
    x_edges = np.linspace(-2, 2, 11)
    z_edges = np.linspace(0, 5, 11)

    # convert sampled_index back to row and column indices
    sampled_row = sampled_idx // 10
    sampled_col = sampled_idx % 10

    # define sampled zone edges
    x_min, x_max = x_edges[sampled_col], x_edges[sampled_col + 1]
    z_min, z_max = z_edges[sampled_row], z_edges[sampled_row + 1]

    # sample from the selected zone
    x_sample = np.random.uniform(x_min, x_max)
    z_sample = np.random.uniform(z_min, z_max)

    # update the input dict with the sampled pitch locations
    row['plate_x'] = x_sample
    row['plate_z'] = z_sample

    # define the pitch characteristics to be sampled
    pitch_chars = ['arm_angle', 'release_pos_x', 'release_pos_y', 'release_extension', 'release_speed', 
                             'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z']

    # locate for the current pitcher and pitch type the mean and standard deviation for each of the pitch characteristics 
    pitch_data = pitch_characteristics_df.loc[
        (pitch_characteristics_df['pitcher'] == row['pitcher']) &
        (pitch_characteristics_df['pitch_type'] == row['pitch_type'])
    ]

    # extract the means and standard deviations
    means = torch.tensor([pitch_data[f'{char}_mean'].values[0] for char in pitch_chars],dtype=torch.float32)
    stds  = torch.tensor([pitch_data[f'{char}_std'].values[0] for char in pitch_chars],dtype=torch.float32)

    # sample each of the pitch characteristics
    samples = torch.normal(means, stds)

    # update the input dict with the sampled pitch characteristics
    for idx, pitch_char in enumerate(pitch_chars):
        row[pitch_char] = samples[idx].item()
        
    # no return (modifies dict in place)
    
def sample_swing(row, models):
    """
    Take in various features about a pitcher/batter matchup and use a pre-trained swing model to sample whether or not the batter swings. 
    
    Inputs:
        row (dict): various informatoin regarding a specific pitcher/batter matchup
        models (dict): a dict of dicts that stores various objects in a tree structure
        
    Outputs:
        swing (int): whether or not the batter swings (1 = swing)
    """
    
    # load the necessary scaler and model
    bsm_scaler = models['scalers']['bsm_scaler']
    swing_model = models['models']['swing_model']

    # extract the scaled X features and convert them to a tensor
    X = bsm_scaler.transform([[row[feat] for feat in models['features']['swing_X_feats']]])
    X_tensor = torch.from_numpy(X).float()

    # run the model and convert the raw logits into probabilities
    swing_logits = swing_model(X_tensor).detach()
    swing_probs = F.softmax(swing_logits, dim=1)

    # sample from the probabilities
    swing = torch.multinomial(swing_probs, num_samples=1).item()

    return swing

def sample_contact(row, models):
    """
    Take in various features about a pitcher/batter matchup and use a pre-trained swing model to sample whether or not the batter makes contact, given that he swings. 
    
    Inputs:
        row (dict): various informatoin regarding a specific pitcher/batter matchup
        models (dict): a dict of dicts that stores various objects in a tree structure
        
    Outputs:
        contact (int): whether or not the batter makes contact (1 = makes contact)
    """
    
    # load the necessary scaler and model
    bcm_scaler = models['scalers']['bcm_scaler']
    contact_model = models['models']['contact_model']

    # extract the scaled X features and convert them to a tensor
    X = bcm_scaler.transform([[row[feat] for feat in models['features']['swing_X_feats']]])
    X_tensor = torch.from_numpy(X).float()

    # run the model and convert the raw logits into probabilities
    contact_logits = contact_model(X_tensor).detach()
    contact_probs = F.softmax(contact_logits, dim=1)

    # sample from the probabilities
    contact = torch.multinomial(contact_probs, num_samples=1).item()

    return contact

def sample_event(row, models):
    """
    Take in various features about a pitcher/batter matchup and use a pre-trained swing model to sample the outcome of a batter's swing, given that he makes contact. 
    
    Inputs:
        row (dict): various informatoin regarding a specific pitcher/batter matchup
        models (dict): a dict of dicts that stores various objects in a tree structure
        
    Outputs:
        event (int): the sampled result of the swing
    """
    
    # load the necessary scaler, label encoder, and model
    bem_scaler = models['scalers']['bem_scaler']
    bem_le = models['label_encoders']['bem_le']
    event_model = models['models']['event_model']

    # extract the scaled X features and convert them to a tensor
    X = bem_scaler.transform([[row[feat] for feat in models['features']['swing_X_feats']]])
    X_tensor = torch.from_numpy(X).float()

    # run the model and convert the raw logits into probabilities
    event_logits = event_model(X_tensor).detach()
    event_probs = F.softmax(event_logits, dim=1)

    # sample from the probabilities and convert to text using the label encoder
    event = bem_le.classes_[torch.multinomial(event_probs, num_samples=1).item()]

    return event

def sample_pitch(batter_id, pitcher_id, balls, strikes, models, data):
    """
    Provided a situation (batter/pitcher/count), simulates what the result of the next pitch will be by sampling up to 5 different models.
    
    Inputs:
        batter_id (int): the MLBAM ID corresponding to the batter
        pitcher_id (int): the MLBAM ID corresponding to the pitcher
        balls (int): how many balls are in the count
        strikes (int): how many strikes are in the count
        models (dict): a dict of dicts that stores various objects in a tree structure
        data (dict): stores various dataframes containing processed data
        
    Outputs:
        row (dict): a dictionary containing the batter/pitcher data as well as the outcome of the pitch
    """
    
    # load the necessary dataframes
    batter_df = data['batter_df']
    pitcher_df = data['pitcher_df']

    # pull the relevant batter- and pitcher-specific information
    row = batter_df.loc[batter_id].to_dict()
    row.update(pitcher_df.loc[pitcher_id].to_dict())

    # add the count
    row['balls'] = balls
    row['strikes'] = strikes

    # one-hot encode the pitcher and batter handedness
    row['p_throws_l'] = 1 if row['p_throws'] == 'L' else 0
    row['b_stands_l'] = (
        0 if row['stand'] == 'R' else
        0 if row['stand'] == 'S' and row['p_throws'] == 'L' else
        1
    )
    
    # sample the pitch type and pitch location of the next pitch
    sample_pitch_type(row, models)
    sample_pitch_location(row, models, data)
    
    # sample whether or not the batter swings
    swing = sample_swing(row, models)
    
    # if the batter swings...
    if swing:
        
        # samples whether or not the batter makes contact
        contact = sample_contact(row, models)
        
        # if the batter makes contact...
        if contact:
            
            # samples the batted ball outcome
            row['outcome'] = sample_event(row, models)
            
        # if the batter doesn't make contact...
        else:
            row['outcome'] = 'swinging_strike'
      
    # if the batter doesn't swing and the pitch is in the strike zone...
    elif row['plate_x']>=-.83 and row['plate_x']<=.83 and row['plate_z']>=row['sz_bot'] and row['plate_z']<=row['sz_top']:
        row['outcome'] = 'called_strike'
       
    # if the batter doesn't swing and the pitch is not in the strike zone...
    else:
        row['outcome'] = 'ball'
    
    return row

def sample_single_pa(batter_id, pitcher_id, models, data):
    """
    Provided a matchup (batter/pitcher), simulates an entire plate appearance by simulating individual pitches. 
    
    Inputs:
        batter_id (int): the MLBAM ID corresponding to the batter
        pitcher_id (int): the MLBAM ID corresponding to the pitcher
        models (dict): a dict of dicts that stores various objects in a tree structure
        data (dict): stores various dataframes containing processed data
        
    Outputs:
        pa (list): a list of dictionaries containing the batter/pitcher data as well as the outcome of each pitch
    """
    
    # initialize a list to store individual pitch dicts (rows)
    pa = []
    
    # initialize count
    balls = 0
    strikes = 0
    
    # define plate-appearance-ending outcomes
    terminal_outcomes = {'ground_ball', 'fly_ball', 'line_drive', 'popup', 'single', 'double', 'triple', 'home_run'}
    
    # loops untl a terminating outcome is reached
    while True:
        
        # add a new simulated pitch to the plate appearance list
        pa.append(sample_pitch(batter_id, pitcher_id, balls, strikes, models, data))
        
        # extract the outcome of the previous pitch
        prev_outcome = pa[-1]['outcome']
        
        # end the at bat if a terminal outcome is reached
        if prev_outcome in terminal_outcomes:
            pa[-1]['terminal_outcome'] = prev_outcome
            break
            
        # check if the batter struck out
        elif prev_outcome in {'swinging_strike', 'called_strike'}:
            if strikes == 2:
                pa[-1]['terminal_outcome'] = 'strikeout'
                break
            strikes += 1
            
        # check if the batter walked
        elif prev_outcome == 'ball':
            if balls == 3:
                pa[-1]['terminal_outcome'] = 'walk'
                break
            balls += 1
            
        # only increments strikes for a foul if there are fewer than 2 strikes
        elif prev_outcome == 'foul' and strikes < 2:
            pa[-1]['terminal_outcome'] = np.nan
            strikes += 1
            
        else:
            pa[-1]['terminal_outcome'] = np.nan
            
    return pa

def sample_pas(batter_id, pitcher_id, models, data, epochs=1000):
    """
    Repeatedly simulates plate appearances for a specific batter/pitcher matchup.
    
    Inputs:
        batter_id (int): the MLBAM ID corresponding to the batter
        pitcher_id (int): the MLBAM ID corresponding to the pitcher
        models (dict): a dict of dicts that stores various objects in a tree structure
        data (dict): stores various dataframes containing processed data
        epochs (int): how many times the simulation should be run
        
    Outputs:
        all_pas_df (pd.DataFrame): a dataframe containing the batter/pitcher data as well as the outcome of each pitch
    """
    
    all_pas = []
    
    for epoch in range(epochs):
        all_pas += sample_single_pa(batter_id, pitcher_id, models, data)
            
    all_pas_df = pd.DataFrame(all_pas)

    # add count column for downstream count-specific filtering
    all_pas_df["count"] = all_pas_df["balls"].astype(str) + "-" + all_pas_df["strikes"].astype(str)
    
    all_pas_df = all_pas_df.loc[:, ['pitcher', 'batter', 'count', 'pitch_type', 'plate_x', 'plate_z', 'sz_top', 'sz_bot', 'release_speed', 'release_spin_rate', 'pfx_x', 'pfx_z', 'outcome', 'terminal_outcome']]
            
    return all_pas_df

if __name__=="__main__":
    pass