import pandas as pd
import numpy as np
import torch
import pickle

from helper_functions import adjust_stat
import batter_models as bm
import pitch_type_model as ptm
import pitch_location_model as plm

def preprocess_data():
    """
    Perform various preprocessing steps required to prepare the data to be transformed into batter and pitcher dataframes. 
    
    Inputs:
        none
        
    Outputs:
        df (pd.DataFrame): a modified dataframe with all necessary changes made
    """
    
    # load in raw statcast data
    df = pd.read_parquet('data/data.parquet')
    
    # creates various helper columns
    df['year'] = df['game_date'].dt.year
    df['is_strike'] = np.where(df['zone'].isin([1, 2, 3, 4, 5, 6, 7, 8, 9]), 1, 0)

    # determine whether the row is a swing, then whether it was an in-zone swing or an out-of-zone swing
    df['swing'] = np.where(df['description'].isin(['hit_into_play', 'foul', 'swinging_strike', 'foul_tip', 
                                                   'swinging_strike_blocked', 'missed_bunt', 'foul_bunt', 
                                                   'bunt_foul_tip']), 1, 0)
    df['zswing'] = np.where((df['swing'] == 1) & (df['is_strike'] == 1), 1, 0)
    df['oswing'] = np.where((df['swing'] == 1) & (df['is_strike'] == 0), 1, 0)

    # determine whether the batter made contact, then whether it was in-zone contact or out-of-zone contact
    df['contact'] = np.where(df['description'].isin(['hit_into_play', 'foul']), 1, 0)
    df['zcontact'] = np.where((df['contact'] == 1) & (df['is_strike'] == 1), 1, 0)
    df['ocontact'] = np.where((df['contact'] == 1) & (df['is_strike'] == 0), 1, 0)

    # determine whether the hit quality was a barrel or hard hit
    df['barrel'] = np.where(df['launch_speed_angle'].fillna(0) == 6, 1, 0)
    df['hard_hit'] = np.where((df['launch_speed'].fillna(0) >= 95) & (df['description']=='hit_into_play'), 1, 0)

    # determine whether the row was a first pitch and whether there was a first pitch swing
    df['FP'] = np.where((df['balls']==0) & (df['strikes']==0), 1, 0)
    df['FP_swing'] = np.where((df['FP']==1) & (df['swing']==1), 1, 0)

    # determine whether or not the ball was hit into play
    df['in_play'] = np.where(df['description']=='hit_into_play', 1, 0)
    
    return df

def get_batter_df():
    """
    Takes a preprocessed dataframe and computes several batter-specific metrics.
    
    Inputs:
        none
        
    Outputs:
        batter_df (pd.DataFrame): each row corresponds to a different batter ID, with the various columns representing different behavioral stats about the player (e.g., plate discipline or swing metrics)
    """
    
    # load processed data
    df = preprocess_data()

    # aggregate (mostly sum) all the raw counts, grouped by batter
    batter_pd = df.groupby('batter').agg({'is_strike':['count', 'mean', 'sum'], 
                                          'swing':'sum', 'zswing':'sum', 'oswing':'sum',
                                          'contact':'sum', 'zcontact':'sum', 'ocontact':'sum', 
                                          'barrel':'sum', 'hard_hit':'sum', 
                                          'FP': 'sum', 'FP_swing':'sum', 
                                          'in_play':'sum'}).reset_index()

    # rename columns to avoid multi-indexing
    batter_pd.columns = ['batter', 'pitches', 'zone%', 'strikes', 'total_swings', 'zswings', 'oswings', 'contacts', 
                         'zcontacts', 'ocontacts', 'barrels', 'hard_hits', 'FPs', 'FP_swings', 'in_plays']

    # compute various rate statistics using Bayesian smoothing
    batter_pd['swing%'] = adjust_stat(batter_pd, batter_pd['total_swings'], batter_pd['pitches'], 'swing%')
    batter_pd['zswing%'] = adjust_stat(batter_pd, batter_pd['zswings'], batter_pd['strikes'], 'zswing%')
    batter_pd['oswing%'] = adjust_stat(batter_pd, batter_pd['oswings'], (batter_pd['pitches'] - batter_pd['strikes']), 
                                       'oswing%')
    batter_pd['contact%'] = adjust_stat(batter_pd, batter_pd['contacts'], batter_pd['total_swings'], 'contact%')
    batter_pd['zcontact%'] = adjust_stat(batter_pd, batter_pd['zcontacts'], batter_pd['zswings'], 'zcontact%')
    batter_pd['ocontact%'] = adjust_stat(batter_pd, batter_pd['ocontacts'], batter_pd['oswings'], 'ocontact%')
    batter_pd['barrel%'] = adjust_stat(batter_pd, batter_pd['barrels'], batter_pd['in_plays'], 'barrel%')
    batter_pd['hard_hit%'] = adjust_stat(batter_pd, batter_pd['hard_hits'], batter_pd['in_plays'], 'hard_hit%')
    batter_pd['fswing%'] = adjust_stat(batter_pd, batter_pd['FP_swings'], batter_pd['FPs'], 'fswing%')

    # compute average (and p80 where applicable) swing metrics
    avg_metrics = df.loc[df['description']=='hit_into_play', 
                         ['batter', 'description', 'bat_speed', 'swing_length', 'attack_angle', 'attack_direction', 
                          'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches', 
                          'intercept_ball_minus_batter_pos_y_inches']].dropna().groupby('batter').agg(
        {'description':'count', 
         'bat_speed': ['median', lambda x: x.quantile(0.8)],
         'swing_length':'median', 
         'attack_angle':'median',
         'attack_direction':'median',
         'swing_path_tilt':'median',
         'intercept_ball_minus_batter_pos_x_inches':'median',
         'intercept_ball_minus_batter_pos_y_inches':'median'}
    ).reset_index()

    # rename columns to remove multi-indexing
    avg_metrics.columns = ['batter', 'in_play_swings', 'avg_bat_speed', 'p80_bat_speed', 'avg_swing_length', 
                           'avg_attack_angle', 'avg_attack_direction', 'avg_swing_path_tilt', 'avg_bat_x_intercept', 
                           'avg_bat_y_intercept']

    # compute the average strike zone tops and bottoms for each batter and handedness
    batter_df = df.groupby(['batter', 'stand']).agg({'sz_top':'mean', 'sz_bot':'mean'}).reset_index()

    # determine the handedness of the batter by creating two one-hot encoded columns
    batter_df['b_stands_l'] = np.where(batter_df['stand']=='L', 1, 0)
    batter_df['b_stands_r'] = np.where(batter_df['stand']=='R', 1, 0)

    # group by batter and recompute the strike zone tops and bottoms
    batter_df = batter_df.groupby('batter').agg({'sz_top':'mean', 'sz_bot':'mean', 'b_stands_l':'sum', 'b_stands_r':'sum'}).reset_index()

    # determine whether the batter is a switch hitter or not
    batter_df['stand'] = np.where((batter_df['b_stands_l']==1) & (batter_df['b_stands_r']==1), 'S', 
                                  np.where(batter_df['b_stands_l']==1, 'L', 'R'))

    # select only the necessary columns from the batter dataframe
    batter_df = batter_df.loc[:, ['batter', 'stand', 'sz_top', 'sz_bot']]

    # add in plate discipline and swing metrics to the batter dataframe
    batter_df = batter_df.merge(batter_pd, on='batter')
    batter_df = batter_df.merge(avg_metrics, on='batter')

    # create an index for the batter ID
    batter_df.set_index('batter', drop=False, inplace=True)
    
    return batter_df


def get_pitcher_df():
    """
    Takes a preprocessed dataframe and computes several pitcher-specific metrics.
    
    Inputs:
        none
        
    Outputs:
        pitcher_df (pd.DataFrame): each row corresponds to a different pitcher ID, with the various columns representing what pitches the pitcher throws
    """

    # load processed data
    df = preprocess_data()

    # calculate the number of pitchers thrown by each pitcher
    pitcher_df = df.groupby(['pitcher', 'p_throws']).agg({'batter':'count'}).reset_index()
    pitcher_df.columns = ['pitcher', 'p_throws', 'num_pitches']
    
    # get the pitch types that each pitcher throw and add to df as columns
    pitch_counts = pd.crosstab(df['pitcher'], df['pitch_type'])
    arsenals = pitch_counts.ge(2).astype(int)
    arsenals.columns = ['has_' + col for col in arsenals.columns]
    arsenals['arsenal_size'] = arsenals.sum(axis=1)
    arsenals.reset_index(inplace=True)
    pitcher_df = pitcher_df.merge(arsenals)

    # only keep pitchers with at least 50 thrown pitches
    pitcher_df = pitcher_df.loc[pitcher_df['num_pitches'] >= 50, :]
    
    # create an index for the batter ID
    pitcher_df.set_index('pitcher', drop=False, inplace=True)

    return pitcher_df

    
def get_pitch_characteristics_df():
    """
    Takes a preprocessed dataframe and computes several pitch-specific metrics for each pitcher. 
    
    Inputs:
        none
        
    Outputs:
        pitch_characteristics_df (pd.DataFrame): each row corresponds to a different combination of pitch type and pitcher ID, with the various columns representing the mean and standard deviation for that pitch's metrics
    """

    # load processed data
    df = preprocess_data()

    # calculate mean and standard deviation for each pitcher and pitch type combination
    pitch_characteristics_df = df.groupby(['pitcher', 'pitch_type']).agg({'arm_angle':['mean', 'std'], 
                                                                          'release_pos_x':['mean', 'std'], 
                                                                          'release_pos_y':['mean', 'std'], 
                                                                          'release_extension':['mean', 'std'], 
                                                                          'release_speed':['mean', 'std'], 
                                                                          'release_spin_rate':['mean', 'std'], 
                                                                          'spin_axis':['mean', 'std'], 
                                                                          'pfx_x':['mean', 'std'], 
                                                                          'pfx_z':['mean', 'std']}).reset_index()

    # rename columns
    pitch_characteristics_df.columns = ['pitcher', 'pitch_type', 
                                        'arm_angle_mean', 'arm_angle_std', 
                                        'release_pos_x_mean', 'release_pos_x_std', 
                                        'release_pos_y_mean', 'release_pos_y_std',
                                        'release_extension_mean', 'release_extension_std',
                                        'release_speed_mean', 'release_speed_std',
                                        'release_spin_rate_mean', 'release_spin_rate_std',
                                        'spin_axis_mean', 'spin_axis_std',
                                        'pfx_x_mean', 'pfx_x_std',
                                        'pfx_z_mean', 'pfx_z_std']
    
    for stat in ['arm_angle_mean', 'arm_angle_std', 'release_pos_x_mean', 'release_pos_x_std', 'release_pos_y_mean', 
                 'release_pos_y_std', 'release_extension_mean', 'release_extension_std']:
        pitch_characteristics_df[stat] = pitch_characteristics_df[stat].fillna(
            pitch_characteristics_df.groupby('pitcher')[stat].transform('mean')
        )
        
    pitch_characteristics_df = pitch_characteristics_df.dropna()
    
    return pitch_characteristics_df


def load_models():
    """
    Load various features, models, scalers, and label encoders to be used in pitch simulations. 
    
    Inputs:
        none
        
    Outputs:
        models (dict): a dict of dicts that stores various objects in a tree structure that can be used later
    """
    
    ### features ##
    
    pitch_X_feats = ['balls', 'strikes', 'p_throws_l', 'b_stands_l', 'avg_bat_speed', 'p80_bat_speed', 
                     'avg_swing_length', 'avg_attack_angle', 'avg_attack_direction', 'avg_swing_path_tilt', 
                     'avg_bat_x_intercept', 'avg_bat_y_intercept', 'swing%', 'zswing%', 'oswing%', 'contact%', 
                     'zcontact%', 'ocontact%', 'barrel%', 'hard_hit%', 'fswing%', 'has_CH', 'has_CS', 'has_CU', 
                     'has_EP', 'has_FA', 'has_FC', 'has_FF', 'has_FO', 'has_FS', 'has_KC', 'has_KN', 'has_PO', 
                     'has_SC', 'has_SI', 'has_SL', 'has_ST', 'has_SV', 'arsenal_size']
    
    swing_X_feats = ['balls', 'strikes', 'p_throws_l', 'b_stands_l', 'arm_angle', 'release_pos_x', 'release_pos_y',
                     'release_extension', 'plate_x', 'plate_z', 'release_speed', 'release_spin_rate', 'spin_axis', 
                     'pfx_x', 'pfx_z', 'avg_bat_speed', 'p80_bat_speed', 'avg_swing_length', 'avg_attack_angle', 
                     'avg_attack_direction', 'avg_swing_path_tilt', 'avg_bat_x_intercept', 'avg_bat_y_intercept', 
                     'swing%', 'zswing%', 'oswing%', 'contact%', 'zcontact%', 'ocontact%', 'barrel%', 'hard_hit%', 
                     'fswing%']
    
    ### models ###
    
    ptm_model = torch.load('model_weights/pitch_type_model_weights.pth', weights_only=False)
    pitch_type_model = ptm.ffnn(**ptm_model['model_params'])
    pitch_type_model.load_state_dict(ptm_model['model_state_dict'])
    pitch_type_model.eval()
    
    plm_model = torch.load('model_weights/pitch_location_model_weights.pth', weights_only=False)
    pitch_location_model = plm.ffnn(**plm_model['model_params'])
    pitch_location_model.load_state_dict(plm_model['model_state_dict'])
    pitch_location_model.eval()
    
    bsm_model = torch.load('model_weights/swing_model_weights.pth', weights_only=False)
    swing_model = bm.ffnn(**bsm_model['model_params'])
    swing_model.load_state_dict(bsm_model['model_state_dict'])
    swing_model.eval()
    
    bcm_model = torch.load('model_weights/contact_model_weights.pth', weights_only=False)
    contact_model = bm.ffnn(**bcm_model['model_params'])
    contact_model.load_state_dict(bcm_model['model_state_dict'])
    contact_model.eval()
    
    bem_model = torch.load('model_weights/event_model_weights.pth', weights_only=False)
    event_model = bm.ffnn(**bem_model['model_params'])
    event_model.load_state_dict(bem_model['model_state_dict'])
    event_model.eval()
    
    ### scalers ###
    
    with open('scalers_encoders/ptm_scaler.pkl', 'rb') as f:
        ptm_scaler = pickle.load(f)
        
    with open('scalers_encoders/plm_scaler.pkl', 'rb') as f:
        plm_scaler = pickle.load(f)
        
    with open('scalers_encoders/bsm_scaler.pkl', 'rb') as f:
        bsm_scaler = pickle.load(f)
        
    with open('scalers_encoders/bcm_scaler.pkl', 'rb') as f:
        bcm_scaler = pickle.load(f)
        
    with open('scalers_encoders/bem_scaler.pkl', 'rb') as f:
        bem_scaler = pickle.load(f)
        
    ### label encoders ###

    with open('scalers_encoders/ptm_pitcher_le.pkl', 'rb') as f:
        ptm_pitcher_le = pickle.load(f)

    with open('scalers_encoders/ptm_pitch_type_le.pkl', 'rb') as f:
        ptm_pitch_type_le = pickle.load(f)
        
    with open('scalers_encoders/plm_pitcher_le.pkl', 'rb') as f:
        plm_pitcher_le = pickle.load(f)

    with open('scalers_encoders/plm_pitch_type_le.pkl', 'rb') as f:
        plm_pitch_type_le = pickle.load(f)
        
    with open('scalers_encoders/bem_le.pkl', 'rb') as f:
        bem_le = pickle.load(f)
    
    # create tree structure for final return
    models = {
        'features':{
            'pitch_X_feats':pitch_X_feats,
            'swing_X_feats':swing_X_feats
        },
        'models':{
            'pitch_type_model':pitch_type_model, 
            'pitch_location_model':pitch_location_model, 
            'swing_model':swing_model, 
            'contact_model':contact_model,
            'event_model':event_model
        }, 
        'scalers':{
            'ptm_scaler':ptm_scaler,
            'plm_scaler':plm_scaler,
            'bsm_scaler':bsm_scaler,
            'bcm_scaler':bcm_scaler,
            'bem_scaler':bem_scaler}, 
        'label_encoders':
        {
            'ptm_pitcher_le':ptm_pitcher_le,
            'ptm_pitch_type_le':ptm_pitch_type_le, 
            'plm_pitcher_le':plm_pitcher_le,
            'plm_pitch_type_le':plm_pitch_type_le, 
            'bem_le':bem_le
        }
    }
    
    return models


if __name__=="__main__":
    pass