from copy import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
    

def adjust_stat(df, num, den, stat_name):
    """
    Description:
        Applies Bayesian smoothing (Bayesian shrinking) to a provided rate statistics to account for small sample sizes.
        
    Inputs:
        df (pd.DataFrame): raw (relatively) Statcast data
        num (pd.Series): a column of data to be the numerator of the rate statistic
        den (pd.Series): a column of data to be the denominator of the rate statistic
        stat_name (string): a name to apply to the statistic
        
    Outputs:
        df_copy[stat_name] (pd.Series): a column from the modified data that corresponds to the resulting statistic
    """
    
    # calculate statistic without accounting for sample size
    df['stat'] = num / den

    # compute mean and variance from naive-ly calculated statistic
    mu = df['stat'].mean()
    var = df['stat'].var()
    
    # compute relevant metrics using mean and variance
    common = mu * (1 - mu) / var - 1
    alpha0 = mu * common
    beta0 = (1 - mu) * common

    # recalculate stat while accounting for sample size differences
    df[stat_name] = (alpha0 + num) / (alpha0 + beta0 + den)
    
    return df[stat_name]


def calc_avg_swing_metrics(df):
    """
    Description:
        Calculate from and add to a provided DataFrame of Statcast data various columns related to a batter's average swing
        metrics (e.g., bat speed, attack angle, etc.).
        
    Inputs:
        df (pd.DataFrame): raw (relatively) Statcast data
        
    Outputs:
        df_swing (pd.DataFrame): the input Statcast data with added columns for batter average swing metrics 
    """
    
    # extract necessary columns from the input data
    df_swing = df.loc[:, ['pitcher', 'batter', 'pitch_type', 'balls', 'strikes', 'p_throws', 'stand', 'arm_angle',
                          'release_pos_x', 'release_pos_y', 'release_extension', 'plate_x', 'plate_z', 'release_speed',
                          'release_spin_rate', 'spin_axis', 'pfx_x', 'pfx_z', 'bat_speed', 'swing_length', 'attack_angle',
                          'attack_direction', 'swing_path_tilt', 'intercept_ball_minus_batter_pos_x_inches',
                          'intercept_ball_minus_batter_pos_y_inches', 'swing%', 'zswing%', 'oswing%', 'contact%', 'zcontact%',
                          'ocontact%', 'barrel%', 'hard_hit%', 'fswing%', 'bb_type', 'swing', 'contact', 'events']]
    
    # compute average (and p80 where applicable) swing metrics
    avg_metrics = df.loc[df['description']=='hit_into_play', 
                         df_swing.columns].dropna().groupby('batter').agg(
        {'balls':'count', 
         'bat_speed': ['median', lambda x: x.quantile(0.8)],
         'swing_length':'median', 
         'attack_angle':'median',
         'attack_direction':'median',
         'swing_path_tilt':'median',
         'intercept_ball_minus_batter_pos_x_inches':'median',
         'intercept_ball_minus_batter_pos_y_inches':'median'}
    ).reset_index()

    # rename columns to remove multi-indexing
    avg_metrics.columns = ['batter', 'swings', 'avg_bat_speed', 'p80_bat_speed', 'avg_swing_length', 'avg_attack_angle',
                           'avg_attack_direction', 'avg_swing_path_tilt', 'avg_bat_x_intercept', 'avg_bat_y_intercept']

    # combine input data with average swing metrics
    df_swing = df_swing.merge(avg_metrics)

    return df_swing


def calc_pd_stats(df):
    """
    Description: 
        Given a DataFrame of Statcast data (df), computes various plate discipline metrics for each batter, such as swing%,
        contact%, etc.
    
    Inputs:
        df (pd.DataFrame): Statcast data
    
    Outputs:
        df_with_pd (pd.DataFrame): The original data with additional columns for plate discipline stats.
    """
    
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

    # aggregate (mostly sum) all the raw counts, grouped by batter
    batter_pd = df.groupby('batter').agg({'is_strike':['count', 'mean', 'sum'], 
                                          'swing':'sum', 'zswing':'sum', 'oswing':'sum',
                                          'contact':'sum', 'zcontact':'sum', 'ocontact':'sum', 
                                          'barrel':'sum', 'hard_hit':'sum', 
                                          'FP': 'sum', 'FP_swing':'sum', 
                                          'in_play':'sum'}).reset_index()

    # rename columns to avoid multi-indexing
    batter_pd.columns = ['batter', 'pitches', 'zone%', 'strikes', 'swings', 'zswings', 'oswings', 'contacts', 
                         'zcontacts', 'ocontacts', 'barrels', 'hard_hits', 'FPs', 'FP_swings', 'in_plays']

    # compute various rate statistics using Bayesian smoothing
    batter_pd['swing%'] = adjust_stat(batter_pd, batter_pd['swings'], batter_pd['pitches'], 'swing%')
    batter_pd['zswing%'] = adjust_stat(batter_pd, batter_pd['zswings'], batter_pd['strikes'], 'zswing%')
    batter_pd['oswing%'] = adjust_stat(batter_pd, batter_pd['oswings'], (batter_pd['pitches'] - batter_pd['strikes']), 'oswing%')
    batter_pd['contact%'] = adjust_stat(batter_pd, batter_pd['contacts'], batter_pd['swings'], 'contact%')
    batter_pd['zcontact%'] = adjust_stat(batter_pd, batter_pd['zcontacts'], batter_pd['zswings'], 'zcontact%')
    batter_pd['ocontact%'] = adjust_stat(batter_pd, batter_pd['ocontacts'], batter_pd['oswings'], 'ocontact%')
    batter_pd['barrel%'] = adjust_stat(batter_pd, batter_pd['barrels'], batter_pd['in_plays'], 'barrel%')
    batter_pd['hard_hit%'] = adjust_stat(batter_pd, batter_pd['hard_hits'], batter_pd['in_plays'], 'hard_hit%')
    batter_pd['fswing%'] = adjust_stat(batter_pd, batter_pd['FP_swings'], batter_pd['FPs'], 'fswing%')

    # add calculated statistics to the raw data
    df_with_pd = df.merge(batter_pd[['batter', 'swing%', 'zswing%', 'oswing%', 'contact%', 'zcontact%', 
                                     'ocontact%', 'barrel%', 'hard_hit%', 'fswing%']])
    
    return df_with_pd


def plot_loss(train_losses, val_losses):
    """ 
    Description:
        Visualize how the training and testing losses change as the number of epochs increases.
        
    Inputs:
        train_losses (list): a list of training losses where the value at one index corresponds to the loss from that epoch
        val_losses (list): a list of validation losses where the value at one index corresponds to the loss from that epoch
        
    Outputs:
        none
    """
    
    # creates a list of epoch numbers going from 1 to the number of epochs
    epochs = range(1, len(train_losses) + 1)
    
    # create a plot and add lines for both the training and validation losses
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, 'bo-', label='Training Loss')
    plt.plot(epochs, val_losses, 'ro-', label='Validation Loss')
    
    # add labels to the plot
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    
    # stylize the plot
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__=="__main__":
    pass