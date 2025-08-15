import pandas as pd
import numpy as np

def safe_div(num, den):
    if den == 0:
        return np.nan
    else:
        return num / den

def get_summary_stats(df):

    fg = pd.read_parquet('data/fangraphs.parquet')

    df['is_strike'] = np.where((df['plate_x'] <= 0.83) & 
                               (df['plate_x'] >= -0.83) & 
                               (df['plate_z'] <= df['sz_top']) & 
                               (df['plate_z'] >= df['sz_bot']), 1, 0)
    df['is_swing'] = np.where(df['outcome'].isin(['called_strike', 'ball']), 0, 1)
    df['is_contact'] = np.where(df['outcome'].isin(['called_strike', 'ball', 'swinging_strike']), 0, 1)

    total_pitches = df.shape[0]
    pa = sum(df['terminal_outcome'].notna())
    ab = sum(df['terminal_outcome'].notna()) - sum(df['terminal_outcome']=='walk')
    avg = safe_div(sum(df['terminal_outcome'].isin(['single', 'double', 'triple', 'home_run'])), ab)
    obp = safe_div(sum(df['terminal_outcome'].isin(['single', 'double', 'triple', 'home_run', 'walk'])), pa)
    slg = safe_div(sum(df['terminal_outcome']=='single') + \
                   sum(df['terminal_outcome']=='double') * 2 + \
                   sum(df['terminal_outcome']=='triple') * 3 + \
                   sum(df['terminal_outcome']=='home_run') * 4, ab)
    ops = obp + slg
    woba = safe_div((sum(df['terminal_outcome']=='walk') * fg.loc['wBB', 'value'] + \
                     sum(df['terminal_outcome']=='single') * fg.loc['w1B', 'value'] + \
                     sum(df['terminal_outcome']=='double') * fg.loc['w2B', 'value'] + \
                     sum(df['terminal_outcome']=='triple') * fg.loc['w3B', 'value'] + \
                     sum(df['terminal_outcome']=='home_run') * fg.loc['wHR', 'value']), pa)
    p600_1B = safe_div(sum(df['terminal_outcome']=='single') * 600, pa)
    p600_2B = safe_div(sum(df['terminal_outcome']=='double') * 600, pa)
    p600_3B = safe_div(sum(df['terminal_outcome']=='triple') * 600, pa)
    p600_HR = safe_div(sum(df['terminal_outcome']=='home_run') * 600, pa)
    p600_BB = safe_div(sum(df['terminal_outcome']=='walk') * 600, pa)
    p600_K = safe_div(sum(df['terminal_outcome']=='strikeout') * 600, pa)
    k_pct = safe_div(sum(df['terminal_outcome']=='strikeout'), pa)
    bb_pct = safe_div(sum(df['terminal_outcome']=='walk'), pa)
    swing_pct = safe_div(sum(df['is_swing']), total_pitches)
    zswing_pct = safe_div(sum((df['is_swing']==1) & (df['is_strike']==1)), sum(df['is_strike']==1))
    oswing_pct = safe_div(sum((df['is_swing']==1) & (df['is_strike']==0)), sum(df['is_strike']==0))
    contact_pct = safe_div(sum(df['is_contact']), sum(df['is_swing']))
    zcontact_pct = safe_div(sum((df['is_contact']==1) & (df['is_strike']==1)), sum((df['is_swing']==1) & (df['is_strike']==1)))
    ocontact_pct = safe_div(sum((df['is_contact']==1) & (df['is_strike']==0)), sum((df['is_swing']==1) & (df['is_strike']==0)))
    swstr_pct = safe_div(sum(df['outcome']=='swinging_strike'), total_pitches)
    zone_pct = safe_div(sum(df['is_strike']), df.shape[0])
    batted_balls = sum(~df['outcome'].isin(['swinging_strike', 'called_strike', 'ball', 'foul']))
    swings = sum(df['is_swing'])

    summary_stats = {'total_pitches':total_pitches, 'batted_balls':batted_balls, 'swings':swings, 'avg':avg, 'obp':obp, 'slg':slg, 
                     'ops':ops, 'woba':woba, '1B_per_600':p600_1B, '2B_per_600':p600_2B, '3B_per_600':p600_3B, 
                     'HR_per_600':p600_HR, 'BB_per_600':p600_BB, 'K_per_600':p600_K, 'k_pct':k_pct, 
                     'bb_pct':bb_pct, 'swing_pct':swing_pct, 'zswing_pct':zswing_pct, 'oswing_pct':oswing_pct, 
                     'contact_pct':contact_pct, 'zcontact_pct':zcontact_pct, 'ocontact_pct':ocontact_pct, 
                     'swstr_pct':swstr_pct, 'zone_pct':zone_pct}
    
    return summary_stats


def get_pitches_summary(df):

    pitches_summary = df.groupby(['pitch_type']).agg({'pitcher':'count', 'release_speed':'mean',
                                                      'release_spin_rate':'mean', 'pfx_x':'mean', 'pfx_z':'mean'}
                                                    ).reset_index().rename(columns={'pitcher':'pitches'}
                                                                          ).sort_values(by='pitches', 
                                                                                        ascending=False)
    pitches_summary['pct'] = pitches_summary['pitches'] / pitches_summary['pitches'].sum()
    return pitches_summary


if __name__=="__main__":
    pass