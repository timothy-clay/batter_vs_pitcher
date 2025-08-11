from sample_pitches import sample_pas
from datetime import datetime
import pandas as pd
from data_processing import get_batter_df, get_pitcher_df, get_pitch_characteristics_df, load_models

batter_df = get_batter_df()
pitcher_df = get_pitcher_df()
pitch_characteristics_df = get_pitch_characteristics_df()

models = load_models()
data = {'batter_df':batter_df, 
        'pitcher_df':pitcher_df,
        'pitch_characteristics_df':pitch_characteristics_df}


print(f'Starting 1st Test at {datetime.now().strftime("%H:%M:%S")}')
test1 = sample_pas(572191, 669387, models, data)
print(f'Finished 1st Test at {datetime.now().strftime("%H:%M:%S")}')

print(f'Starting 2nd Test at {datetime.now().strftime("%H:%M:%S")}')
test2 = sample_pas(677347, 681810, models, data)
print(f'Finished 2nd Test at {datetime.now().strftime("%H:%M:%S")}')

print(f'Starting 3rd Test at {datetime.now().strftime("%H:%M:%S")}')
test3 = sample_pas(641779, 642547, models, data)
print(f'Finished 3rd Test at {datetime.now().strftime("%H:%M:%S")}')

print(f'Starting 4th Test at {datetime.now().strftime("%H:%M:%S")}')
test4 = sample_pas(694384, 663556, models, data)
print(f'Finished 4th Test at {datetime.now().strftime("%H:%M:%S")}')

print(f'Starting 5th Test at {datetime.now().strftime("%H:%M:%S")}')
test5 = sample_pas(657656, 656302, models, data)
print(f'Finished 5th Test at {datetime.now().strftime("%H:%M:%S")}')