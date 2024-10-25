### Load libraries
import pandas as pd
import numpy as np
import csv
from datetime import datetime, timedelta
import seaborn

# Neural network libraries
import statsmodels.api as sm
import matplotlib.pyplot as plt
import tensorflow
tensorflow.random.set_seed(1)
#from tensorflow.python.keras.layers import Dense
#from tensorflow.python.keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import keras
from keras import layers

### Read in data
def year_match_reader(year):
    df = pd.read_csv('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_'+ str(year) + '.csv', encoding = 'latin-1', quoting=csv.QUOTE_NONE)
    df['year'] = year
    return df

atp_matches = pd.concat([year_match_reader(y) for y in range(2020, 2024)])
atp_matches['tourney_date'] = pd.to_datetime(atp_matches['tourney_date'], format='%Y%m%d')

### Score cleaner
def score_cleaner(score, player_is_winner=True):
    return int(score.split('(')[0].split('-')[1-player_is_winner])

atp_matches['winner_name'].loc[atp_matches['winner_name']=='Mischa Zverev'] = 'Alexander Zverev'
atp_matches['loser_name'].loc[atp_matches['loser_name']=='Mischa Zverev'] = 'Alexander Zverev'
atp_matches['score'] = [s.split('[')[0].strip() for s in atp_matches['score']]
atp_matches = atp_matches.loc[(['RET' not in s for s in atp_matches['score']])]
atp_matches = atp_matches.loc[(['W/O' not in s for s in atp_matches['score']])]
atp_matches = atp_matches.loc[(['Def.' not in s for s in atp_matches['score']])]
atp_matches = atp_matches.loc[(['DEF' not in s for s in atp_matches['score']])]
atp_matches['winner_games_won'] = atp_matches['score'].apply(lambda x: np.sum([score_cleaner(score = s, player_is_winner=True) for s in x.split(' ') if s not in ('RET', 'W/O', 'Def.')]))
atp_matches['loser_games_won'] = atp_matches['score'].apply(lambda x: np.sum([score_cleaner(score = s, player_is_winner=False) for s in x.split(' ') if s not in ('RET', 'W/O', 'Def.')]))
atp_matches['games_played'] = atp_matches['winner_games_won'] + atp_matches['loser_games_won']
atp_matches['GmWinPercWinner'] = atp_matches['winner_games_won']/(atp_matches['winner_games_won'] + atp_matches['loser_games_won'])
atp_matches['GmWinPercLoser'] = atp_matches['loser_games_won']/(atp_matches['winner_games_won'] + atp_matches['loser_games_won'])
atp_matches = atp_matches.loc[(atp_matches['games_played']>0)]

atp_matches['round_num'] = 0
atp_matches['round_num'].loc[atp_matches['round']=='R128'] = 1
atp_matches['round_num'].loc[atp_matches['round']=='R64'] = 2
atp_matches['round_num'].loc[atp_matches['round']=='R32'] = 3
atp_matches['round_num'].loc[atp_matches['round']=='R16'] = 4
atp_matches['round_num'].loc[atp_matches['round']=='QF'] = 5
atp_matches['round_num'].loc[atp_matches['round']=='SF'] = 6
atp_matches['round_num'].loc[atp_matches['round']=='F'] = 7

atp_matches = atp_matches.merge(atp_matches[['tourney_id', 'round_num']].groupby('tourney_id', as_index=False).min(), on = 'tourney_id', suffixes=['','_min'])
atp_matches['match_date'] = atp_matches.apply(lambda x: x['tourney_date'] + timedelta(days=2*(x['round_num'] - x['round_num_min'])), axis = 1)
atp_matches['match_id'] = atp_matches.apply(lambda x: x['match_date'].strftime("%Y%m%d") + '_' + x['winner_name']+ '_' + x['loser_name'], axis=1)
atp_matches['surface'] = atp_matches['surface'].str.lower()
atp_matches['surface'] = atp_matches['surface'].fillna('hard')
atp_matches = atp_matches.sort_values('match_date')

### Elo loop
def logistict_function(x):
    return 1/(1+np.e**(-10*(x-0.5)))
    
players_elo_df = pd.DataFrame({
        'player': np.sort(np.unique(atp_matches['winner_name'].unique().tolist() + atp_matches['loser_name'].unique().tolist())),
        'ds': atp_matches['match_date'].min()-timedelta(days=1),
        'outcome': None,
        'opponent': None,
        'surface': None,
        'best_of': None,
        'point_win_perc': np.nan,
        'elo_rating_1.1_pre': 1000, # general, slow learner
        'elo_rating_1.2_pre': 1000, # general, fast learner
        'elo_rating_2.1_pre': 1000, # surface: clay
        'elo_rating_2.2_pre': 1000, # surface: grass
        'elo_rating_2.3_pre': 1000, # general: hard
        'elo_rating_3.1_pre': 1000, # margin, slow learner
        'elo_rating_3.2_pre': 1000, # margin, fast learner
        'elo_sample_all_pre': 0,
        'elo_sample_clay_pre': 0,
        'elo_sample_grass_pre': 0,
        'elo_sample_hard_pre': 0,
        'total_points_won_pre': 0,
        'total_points_played_pre': 0,
        'total_point_win_perc_pre': 0.5,
        'elo_rating_1.1': 1000, # general, slow learner
        'elo_rating_1.2': 1000, # general, fast learner
        'elo_rating_2.1': 1000, # surface: clay
        'elo_rating_2.2': 1000, # surface: grass
        'elo_rating_2.3': 1000, # general: hard
        'elo_rating_3.1': 1000, # margin, slow learner
        'elo_rating_3.2': 1000, # margin, fast learner
        'elo_sample_all': 0,
        'elo_sample_clay': 0,
        'elo_sample_grass': 0,
        'elo_sample_hard': 0,
        'total_points_won': 0,
        'total_points_played': 0,
        'total_point_win_perc': 0.5
    })

def process_match(match_id, ds, player_1, player_2, surface):
    tmp_match = atp_matches.loc[atp_matches['match_id']==match_id]
    margin = tmp_match['GmWinPercWinner'].values[0]
    player_1_elo_row = players_elo_df.loc[(players_elo_df['ds'] < ds) & (players_elo_df['player'] == player_1)].sort_values('ds').drop_duplicates('player', keep='last')
    player_2_elo_row = players_elo_df.loc[(players_elo_df['ds'] < ds) & (players_elo_df['player'] == player_2)].sort_values('ds').drop_duplicates('player', keep='last')
     
    if surface=='clay':
        # player_1
        player_1_win_prob_11 = 1/(1+10**((player_1_elo_row['elo_rating_1.1'].values[0]-player_2_elo_row['elo_rating_1.1'])/400))
        player_1_win_prob_12 = 1/(1+10**((player_1_elo_row['elo_rating_1.2'].values[0]-player_2_elo_row['elo_rating_1.2'])/400))
        player_1_win_prob_21 = 1/(1+10**((player_1_elo_row['elo_rating_2.1'].values[0]-player_2_elo_row['elo_rating_2.1'])/400))
        player_1_win_prob_31 = 1/(1+10**((player_1_elo_row['elo_rating_3.1'].values[0]-player_2_elo_row['elo_rating_3.1'])/400))
        player_1_win_prob_32 = 1/(1+10**((player_1_elo_row['elo_rating_3.2'].values[0]-player_2_elo_row['elo_rating_3.2'])/400))
        
        player_1_outcome = 1
        player_1_point_win_perc = tmp_match['GmWinPercWinner'].values[0]
        
        player_1_new_ratinig_11 = player_1_elo_row['elo_rating_1.1'].values[0] + 10 * (player_1_outcome - player_1_win_prob_11)
        player_1_new_ratinig_12 = player_1_elo_row['elo_rating_1.2'].values[0] + 32 * (player_1_outcome - player_1_win_prob_12)
        player_1_new_ratinig_21 = player_1_elo_row['elo_rating_2.1'].values[0] + 10 * (player_1_outcome - player_1_win_prob_21)
        player_1_new_ratinig_31 = player_1_elo_row['elo_rating_3.1'].values[0] + 10 * (2*margin) * (player_1_outcome - player_1_win_prob_31)
        player_1_new_ratinig_32 = player_1_elo_row['elo_rating_3.2'].values[0] + 32 * (2*logistict_function(margin)) * (player_1_outcome - player_1_win_prob_32)

        player_1_elo_row_new = pd.DataFrame({
                'player': player_1_elo_row['player'].values[0],
                'ds': ds,
                'outcome': 'W' if player_1_outcome==1 else 'L',
                'opponent': player_2,
                'surface': surface,
                'best_of': tmp_match['best_of'].values[0],
                'point_win_perc': player_1_point_win_perc,
                'elo_rating_1.1_pre': player_1_elo_row['elo_rating_1.1'].values[0], 
                'elo_rating_1.2_pre': player_1_elo_row['elo_rating_1.2'].values[0],
                'elo_rating_2.1_pre': player_1_elo_row['elo_rating_2.1'].values[0],
                'elo_rating_2.2_pre': player_1_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3_pre': player_1_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1_pre': player_1_elo_row['elo_rating_3.1'].values[0],
                'elo_rating_3.2_pre': player_1_elo_row['elo_rating_3.2'].values[0],
                'elo_sample_all_pre': player_1_elo_row['elo_sample_all'].values[0],
                'elo_sample_clay_pre': player_1_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass_pre': player_1_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard_pre': player_1_elo_row['elo_sample_hard'].values[0],
                'total_points_won_pre': player_1_elo_row['total_points_won'].values[0],
                'total_points_played_pre': player_1_elo_row['total_points_played'].values[0],
                'total_point_win_perc_pre': player_1_elo_row['total_points_won'].values[0]/player_1_elo_row['total_points_played'].values[0] if player_1_elo_row['total_points_won'].values[0] > 0 else 0.5,
                'elo_rating_1.1': player_1_new_ratinig_11,
                'elo_rating_1.2': player_1_new_ratinig_12,
                'elo_rating_2.1': player_1_new_ratinig_21,
                'elo_rating_2.2': player_1_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3': player_1_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1': player_1_new_ratinig_31,
                'elo_rating_3.2': player_1_new_ratinig_32,
                'elo_sample_all': player_1_elo_row['elo_sample_all'].values[0]+1,
                'elo_sample_clay': player_1_elo_row['elo_sample_clay'].values[0]+1,
                'elo_sample_grass': player_1_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard': player_1_elo_row['elo_sample_hard'].values[0],
                'total_points_won': tmp_match['winner_games_won'].values[0]+player_1_elo_row['total_points_won'].values[0],
                'total_points_played': tmp_match['games_played'].values[0]+player_1_elo_row['total_points_played'].values[0],
                'total_point_win_perc': (tmp_match['winner_games_won'].values[0]+player_1_elo_row['total_points_won'].values[0])/(tmp_match['games_played'].values[0]+player_1_elo_row['total_points_played'].values[0])
            })
        
        # player_2
        player_2_win_prob_11 = 1/(1+10**((player_2_elo_row['elo_rating_1.1'].values[0]-player_1_elo_row['elo_rating_1.1'])/400))
        player_2_win_prob_12 = 1/(1+10**((player_2_elo_row['elo_rating_1.2'].values[0]-player_1_elo_row['elo_rating_1.2'])/400))
        player_2_win_prob_21 = 1/(1+10**((player_2_elo_row['elo_rating_2.1'].values[0]-player_1_elo_row['elo_rating_2.1'])/400))
        player_2_win_prob_31 = 1/(1+10**((player_2_elo_row['elo_rating_3.1'].values[0]-player_1_elo_row['elo_rating_3.1'])/400))
        player_2_win_prob_32 = 1/(1+10**((player_2_elo_row['elo_rating_3.2'].values[0]-player_1_elo_row['elo_rating_3.2'])/400))
        
        player_2_outcome = 0
        player_2_point_win_perc = tmp_match['GmWinPercLoser'].values[0]
        
        player_2_new_ratinig_11 = player_2_elo_row['elo_rating_1.1'].values[0] + 10 * (player_2_outcome - player_2_win_prob_11)
        player_2_new_ratinig_12 = player_2_elo_row['elo_rating_1.2'].values[0] + 32 * (player_2_outcome - player_2_win_prob_12)
        player_2_new_ratinig_21 = player_2_elo_row['elo_rating_2.1'].values[0] + 10 * (player_2_outcome - player_2_win_prob_21)
        player_2_new_ratinig_31 = player_2_elo_row['elo_rating_3.1'].values[0] + 10 * (2*margin) * (player_2_outcome - player_2_win_prob_31)
        player_2_new_ratinig_32 = player_2_elo_row['elo_rating_3.2'].values[0] + 32 * (2*logistict_function(margin)) * (player_2_outcome - player_2_win_prob_32)

        player_2_elo_row_new = pd.DataFrame({
                'player': player_2_elo_row['player'].values[0],
                'ds': ds,
                'outcome': 'W' if player_2_outcome==1 else 'L',
                'opponent': player_1,
                'surface': surface,
                'best_of': tmp_match['best_of'].values[0],
                'point_win_perc': player_2_point_win_perc,
                'elo_rating_1.1_pre': player_2_elo_row['elo_rating_1.1'].values[0], 
                'elo_rating_1.2_pre': player_2_elo_row['elo_rating_1.2'].values[0],
                'elo_rating_2.1_pre': player_2_elo_row['elo_rating_2.1'].values[0],
                'elo_rating_2.2_pre': player_2_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3_pre': player_2_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1_pre': player_2_elo_row['elo_rating_3.1'].values[0],
                'elo_rating_3.2_pre': player_2_elo_row['elo_rating_3.2'].values[0],
                'elo_sample_all_pre': player_2_elo_row['elo_sample_all'].values[0],
                'elo_sample_clay_pre': player_2_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass_pre': player_2_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard_pre': player_2_elo_row['elo_sample_hard'].values[0],
                'total_points_won_pre': player_2_elo_row['total_points_won'].values[0],
                'total_points_played_pre': player_2_elo_row['total_points_played'].values[0],
                'total_point_win_perc_pre': player_2_elo_row['total_points_won'].values[0]/player_2_elo_row['total_points_played'].values[0] if player_2_elo_row['total_points_won'].values[0] > 0 else 0.5,
                'elo_rating_1.1': player_2_new_ratinig_11,
                'elo_rating_1.2': player_2_new_ratinig_12,
                'elo_rating_2.1': player_2_new_ratinig_21,
                'elo_rating_2.2': player_2_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3': player_2_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1': player_2_new_ratinig_31,
                'elo_rating_3.2': player_2_new_ratinig_32,
                'elo_sample_all': player_2_elo_row['elo_sample_all'].values[0]+1,
                'elo_sample_clay': player_2_elo_row['elo_sample_clay'].values[0]+1,
                'elo_sample_grass': player_2_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard': player_2_elo_row['elo_sample_hard'].values[0],
                'total_points_won': tmp_match['loser_games_won'].values[0]+player_2_elo_row['total_points_won'].values[0],
                'total_points_played': tmp_match['games_played'].values[0]+player_2_elo_row['total_points_played'].values[0],
                'total_point_win_perc': (tmp_match['loser_games_won'].values[0]+player_2_elo_row['total_points_won'].values[0])/(tmp_match['games_played'].values[0]+player_2_elo_row['total_points_played'].values[0])
            })
        
    elif surface=='grass':
        # player_1
        player_1_win_prob_11 = 1/(1+10**((player_1_elo_row['elo_rating_1.1'].values[0]-player_2_elo_row['elo_rating_1.1'])/400))
        player_1_win_prob_12 = 1/(1+10**((player_1_elo_row['elo_rating_1.2'].values[0]-player_2_elo_row['elo_rating_1.2'])/400))
        player_1_win_prob_22 = 1/(1+10**((player_1_elo_row['elo_rating_2.2'].values[0]-player_2_elo_row['elo_rating_2.2'])/400))
        player_1_win_prob_31 = 1/(1+10**((player_1_elo_row['elo_rating_3.1'].values[0]-player_2_elo_row['elo_rating_3.1'])/400))
        player_1_win_prob_32 = 1/(1+10**((player_1_elo_row['elo_rating_3.2'].values[0]-player_2_elo_row['elo_rating_3.2'])/400))
        
        player_1_outcome = 1
        player_1_point_win_perc = tmp_match['GmWinPercWinner'].values[0]
        
        player_1_new_ratinig_11 = player_1_elo_row['elo_rating_1.1'].values[0] + 10 * (player_1_outcome - player_1_win_prob_11)
        player_1_new_ratinig_12 = player_1_elo_row['elo_rating_1.2'].values[0] + 32 * (player_1_outcome - player_1_win_prob_12)
        player_1_new_ratinig_22 = player_1_elo_row['elo_rating_2.2'].values[0] + 10 * (player_1_outcome - player_1_win_prob_22)
        player_1_new_ratinig_31 = player_1_elo_row['elo_rating_3.1'].values[0] + 10 * (2*margin) * (player_1_outcome - player_1_win_prob_31)
        player_1_new_ratinig_32 = player_1_elo_row['elo_rating_3.2'].values[0] + 32 * (2*logistict_function(margin)) * (player_1_outcome - player_1_win_prob_32)

        player_1_elo_row_new = pd.DataFrame({
                'player': player_1_elo_row['player'].values[0],
                'ds': ds,
                'outcome': 'W' if player_1_outcome==1 else 'L',
                'opponent': player_2,
                'surface': surface,
                'best_of': tmp_match['best_of'].values[0],
                'point_win_perc': player_1_point_win_perc,
                'elo_rating_1.1_pre': player_1_elo_row['elo_rating_1.1'].values[0], 
                'elo_rating_1.2_pre': player_1_elo_row['elo_rating_1.2'].values[0],
                'elo_rating_2.1_pre': player_1_elo_row['elo_rating_2.1'].values[0],
                'elo_rating_2.2_pre': player_1_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3_pre': player_1_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1_pre': player_1_elo_row['elo_rating_3.1'].values[0],
                'elo_rating_3.2_pre': player_1_elo_row['elo_rating_3.2'].values[0],
                'elo_sample_all_pre': player_1_elo_row['elo_sample_all'].values[0],
                'elo_sample_clay_pre': player_1_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass_pre': player_1_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard_pre': player_1_elo_row['elo_sample_hard'].values[0],
                'total_points_won_pre': player_1_elo_row['total_points_won'].values[0],
                'total_points_played_pre': player_1_elo_row['total_points_played'].values[0],
                'total_point_win_perc_pre': player_1_elo_row['total_points_won'].values[0]/player_1_elo_row['total_points_played'].values[0] if player_1_elo_row['total_points_won'].values[0] > 0 else 0.5,
                'elo_rating_1.1': player_1_new_ratinig_11,
                'elo_rating_1.1': player_1_new_ratinig_11,
                'elo_rating_1.2': player_1_new_ratinig_12,
                'elo_rating_2.1': player_1_elo_row['elo_rating_2.1'].values[0], 
                'elo_rating_2.2': player_1_new_ratinig_22,
                'elo_rating_2.3': player_1_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1': player_1_new_ratinig_31,
                'elo_rating_3.2': player_1_new_ratinig_32,
                'elo_sample_all': player_1_elo_row['elo_sample_all'].values[0]+1,
                'elo_sample_clay': player_1_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass': player_1_elo_row['elo_sample_grass'].values[0]+1,
                'elo_sample_hard': player_1_elo_row['elo_sample_hard'].values[0],
                'total_points_won': tmp_match['winner_games_won'].values[0]+player_1_elo_row['total_points_won'].values[0],
                'total_points_played': tmp_match['games_played'].values[0]+player_1_elo_row['total_points_played'].values[0],
                'total_point_win_perc': (tmp_match['winner_games_won'].values[0]+player_1_elo_row['total_points_won'].values[0])/(tmp_match['games_played'].values[0]+player_1_elo_row['total_points_played'].values[0])
            })
        
        # player_2
        player_2_win_prob_11 = 1/(1+10**((player_2_elo_row['elo_rating_1.1'].values[0]-player_1_elo_row['elo_rating_1.1'])/400))
        player_2_win_prob_12 = 1/(1+10**((player_2_elo_row['elo_rating_1.2'].values[0]-player_1_elo_row['elo_rating_1.2'])/400))
        player_2_win_prob_22 = 1/(1+10**((player_2_elo_row['elo_rating_2.2'].values[0]-player_1_elo_row['elo_rating_2.2'])/400))
        player_2_win_prob_31 = 1/(1+10**((player_2_elo_row['elo_rating_3.1'].values[0]-player_1_elo_row['elo_rating_3.1'])/400))
        player_2_win_prob_32 = 1/(1+10**((player_2_elo_row['elo_rating_3.2'].values[0]-player_1_elo_row['elo_rating_3.2'])/400))
        
        player_2_outcome = 0
        player_2_point_win_perc = tmp_match['GmWinPercLoser'].values[0]
        
        player_2_new_ratinig_11 = player_2_elo_row['elo_rating_1.1'].values[0] + 10 * (player_2_outcome - player_2_win_prob_11)
        player_2_new_ratinig_12 = player_2_elo_row['elo_rating_1.2'].values[0] + 32 * (player_2_outcome - player_2_win_prob_12)
        player_2_new_ratinig_22 = player_2_elo_row['elo_rating_2.2'].values[0] + 10 * (player_2_outcome - player_2_win_prob_22)
        player_2_new_ratinig_31 = player_2_elo_row['elo_rating_3.1'].values[0] + 10 * (2*margin) * (player_2_outcome - player_2_win_prob_31)
        player_2_new_ratinig_32 = player_2_elo_row['elo_rating_3.2'].values[0] + 32 * (2*logistict_function(margin)) * (player_2_outcome - player_2_win_prob_32)

        player_2_elo_row_new = pd.DataFrame({
                'player': player_2_elo_row['player'].values[0],
                'ds': ds,
                'outcome': 'W' if player_2_outcome==1 else 'L',
                'opponent': player_1,
                'surface': surface,
                'best_of': tmp_match['best_of'].values[0],
                'point_win_perc': player_2_point_win_perc,
                'elo_rating_1.1_pre': player_2_elo_row['elo_rating_1.1'].values[0], 
                'elo_rating_1.2_pre': player_2_elo_row['elo_rating_1.2'].values[0],
                'elo_rating_2.1_pre': player_2_elo_row['elo_rating_2.1'].values[0],
                'elo_rating_2.2_pre': player_2_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3_pre': player_2_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1_pre': player_2_elo_row['elo_rating_3.1'].values[0],
                'elo_rating_3.2_pre': player_2_elo_row['elo_rating_3.2'].values[0],
                'elo_sample_all_pre': player_2_elo_row['elo_sample_all'].values[0],
                'elo_sample_clay_pre': player_2_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass_pre': player_2_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard_pre': player_2_elo_row['elo_sample_hard'].values[0],
                'total_points_won_pre': player_2_elo_row['total_points_won'].values[0],
                'total_points_played_pre': player_2_elo_row['total_points_played'].values[0],
                'total_point_win_perc_pre': player_2_elo_row['total_points_won'].values[0]/player_2_elo_row['total_points_played'].values[0] if player_2_elo_row['total_points_won'].values[0] > 0 else 0.5,
                'elo_rating_1.1': player_2_new_ratinig_11,
                'elo_rating_1.2': player_2_new_ratinig_12,
                'elo_rating_2.1': player_2_elo_row['elo_rating_2.1'].values[0],
                'elo_rating_2.2': player_2_new_ratinig_22,
                'elo_rating_2.3': player_2_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1': player_2_new_ratinig_31,
                'elo_rating_3.2': player_2_new_ratinig_32,
                'elo_sample_all': player_2_elo_row['elo_sample_all'].values[0]+1,
                'elo_sample_clay': player_2_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass': player_2_elo_row['elo_sample_grass'].values[0]+1,
                'elo_sample_hard': player_2_elo_row['elo_sample_hard'].values[0],
                'total_points_won': tmp_match['loser_games_won'].values[0]+player_2_elo_row['total_points_won'].values[0],
                'total_points_played': tmp_match['games_played'].values[0]+player_2_elo_row['total_points_played'].values[0],
                'total_point_win_perc': (tmp_match['loser_games_won'].values[0]+player_2_elo_row['total_points_won'].values[0])/(tmp_match['games_played'].values[0]+player_2_elo_row['total_points_played'].values[0])
            })
        
    elif surface=='hard':
        # player_1
        player_1_win_prob_11 = 1/(1+10**((player_1_elo_row['elo_rating_1.1'].values[0]-player_2_elo_row['elo_rating_1.1'])/400))
        player_1_win_prob_12 = 1/(1+10**((player_1_elo_row['elo_rating_1.2'].values[0]-player_2_elo_row['elo_rating_1.2'])/400))
        player_1_win_prob_23 = 1/(1+10**((player_1_elo_row['elo_rating_2.3'].values[0]-player_2_elo_row['elo_rating_2.3'])/400))
        player_1_win_prob_31 = 1/(1+10**((player_1_elo_row['elo_rating_3.1'].values[0]-player_2_elo_row['elo_rating_3.1'])/400))
        player_1_win_prob_32 = 1/(1+10**((player_1_elo_row['elo_rating_3.2'].values[0]-player_2_elo_row['elo_rating_3.2'])/400))
        
        player_1_outcome = 1
        player_1_point_win_perc = tmp_match['GmWinPercWinner'].values[0]
        
        player_1_new_ratinig_11 = player_1_elo_row['elo_rating_1.1'].values[0] + 10 * (player_1_outcome - player_1_win_prob_11)
        player_1_new_ratinig_12 = player_1_elo_row['elo_rating_1.2'].values[0] + 32 * (player_1_outcome - player_1_win_prob_12)
        player_1_new_ratinig_23 = player_1_elo_row['elo_rating_2.3'].values[0] + 10 * (player_1_outcome - player_1_win_prob_23)
        player_1_new_ratinig_31 = player_1_elo_row['elo_rating_3.1'].values[0] + 10 * (2*margin) * (player_1_outcome - player_1_win_prob_31)
        player_1_new_ratinig_32 = player_1_elo_row['elo_rating_3.2'].values[0] + 32 * (2*logistict_function(margin)) * (player_1_outcome - player_1_win_prob_32)

        player_1_elo_row_new = pd.DataFrame({
                'player': player_1_elo_row['player'].values[0],
                'ds': ds,
                'outcome': 'W' if player_1_outcome==1 else 'L',
                'opponent': player_2,
                'surface': surface,
                'best_of': tmp_match['best_of'].values[0],
                'point_win_perc': player_1_point_win_perc,
                'elo_rating_1.1_pre': player_1_elo_row['elo_rating_1.1'].values[0], 
                'elo_rating_1.2_pre': player_1_elo_row['elo_rating_1.2'].values[0],
                'elo_rating_2.1_pre': player_1_elo_row['elo_rating_2.1'].values[0],
                'elo_rating_2.2_pre': player_1_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3_pre': player_1_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1_pre': player_1_elo_row['elo_rating_3.1'].values[0],
                'elo_rating_3.2_pre': player_1_elo_row['elo_rating_3.2'].values[0],
                'elo_sample_all_pre': player_1_elo_row['elo_sample_all'].values[0],
                'elo_sample_clay_pre': player_1_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass_pre': player_1_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard_pre': player_1_elo_row['elo_sample_hard'].values[0],
                'total_points_won_pre': player_1_elo_row['total_points_won'].values[0],
                'total_points_played_pre': player_1_elo_row['total_points_played'].values[0],
                'total_point_win_perc_pre': player_1_elo_row['total_points_won'].values[0]/player_1_elo_row['total_points_played'].values[0] if player_1_elo_row['total_points_won'].values[0] > 0 else 0.5,
                'elo_rating_1.1': player_1_new_ratinig_11,
                'elo_rating_1.2': player_1_new_ratinig_12, 
                'elo_rating_2.1': player_1_elo_row['elo_rating_2.1'].values[0], 
                'elo_rating_2.2': player_1_elo_row['elo_rating_2.2'].values[0], 
                'elo_rating_2.3': player_1_new_ratinig_23, 
                'elo_rating_3.1': player_1_new_ratinig_31, 
                'elo_rating_3.2': player_1_new_ratinig_32, 
                'elo_sample_all': player_1_elo_row['elo_sample_all'].values[0]+1,
                'elo_sample_clay': player_1_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass': player_1_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard': player_1_elo_row['elo_sample_hard'].values[0]+1,
                'total_points_won': tmp_match['loser_games_won'].values[0]+player_1_elo_row['total_points_won'].values[0],
                'total_points_played': tmp_match['games_played'].values[0]+player_1_elo_row['total_points_played'].values[0],
                'total_point_win_perc': (tmp_match['loser_games_won'].values[0]+player_1_elo_row['total_points_won'].values[0])/(tmp_match['games_played'].values[0]+player_1_elo_row['total_points_played'].values[0])
            })
        
        # player_2
        player_2_win_prob_11 = 1/(1+10**((player_2_elo_row['elo_rating_1.1'].values[0]-player_1_elo_row['elo_rating_1.1'])/400))
        player_2_win_prob_12 = 1/(1+10**((player_2_elo_row['elo_rating_1.2'].values[0]-player_1_elo_row['elo_rating_1.2'])/400))
        player_2_win_prob_23 = 1/(1+10**((player_2_elo_row['elo_rating_2.3'].values[0]-player_1_elo_row['elo_rating_2.3'])/400))
        player_2_win_prob_31 = 1/(1+10**((player_2_elo_row['elo_rating_3.1'].values[0]-player_1_elo_row['elo_rating_3.1'])/400))
        player_2_win_prob_32 = 1/(1+10**((player_2_elo_row['elo_rating_3.2'].values[0]-player_1_elo_row['elo_rating_3.2'])/400))
        
        player_2_outcome = 0
        player_2_point_win_perc = tmp_match['GmWinPercLoser'].values[0]
        
        player_2_new_ratinig_11 = player_2_elo_row['elo_rating_1.1'].values[0] + 10 * (player_2_outcome - player_2_win_prob_11)
        player_2_new_ratinig_12 = player_2_elo_row['elo_rating_1.2'].values[0] + 32 * (player_2_outcome - player_2_win_prob_12)
        player_2_new_ratinig_23 = player_2_elo_row['elo_rating_2.3'].values[0] + 10 * (player_2_outcome - player_2_win_prob_23)
        player_2_new_ratinig_31 = player_2_elo_row['elo_rating_3.1'].values[0] + 10 * (2*margin) * (player_2_outcome - player_2_win_prob_31)
        player_2_new_ratinig_32 = player_2_elo_row['elo_rating_3.2'].values[0] + 32 * (2*logistict_function(margin)) * (player_2_outcome - player_2_win_prob_32)

        player_2_elo_row_new = pd.DataFrame({
                'player': player_2_elo_row['player'].values[0],
                'ds': ds,
                'outcome': 'W' if player_2_outcome==1 else 'L',
                'opponent': player_1,
                'surface': surface,
                'best_of': tmp_match['best_of'].values[0],
                'point_win_perc': player_2_point_win_perc,
                'elo_rating_1.1_pre': player_2_elo_row['elo_rating_1.1'].values[0], 
                'elo_rating_1.2_pre': player_2_elo_row['elo_rating_1.2'].values[0],
                'elo_rating_2.1_pre': player_2_elo_row['elo_rating_2.1'].values[0],
                'elo_rating_2.2_pre': player_2_elo_row['elo_rating_2.2'].values[0],
                'elo_rating_2.3_pre': player_2_elo_row['elo_rating_2.3'].values[0],
                'elo_rating_3.1_pre': player_2_elo_row['elo_rating_3.1'].values[0],
                'elo_rating_3.2_pre': player_2_elo_row['elo_rating_3.2'].values[0],
                'elo_sample_all_pre': player_2_elo_row['elo_sample_all'].values[0],
                'elo_sample_clay_pre': player_2_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass_pre': player_2_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard_pre': player_2_elo_row['elo_sample_hard'].values[0],
                'total_points_won_pre': player_2_elo_row['total_points_won'].values[0],
                'total_points_played_pre': player_2_elo_row['total_points_played'].values[0],
                'total_point_win_perc_pre': player_2_elo_row['total_points_won'].values[0]/player_2_elo_row['total_points_played'].values[0] if player_2_elo_row['total_points_won'].values[0] > 0 else 0.5,
                'elo_rating_1.1': player_2_new_ratinig_11, 
                'elo_rating_1.2': player_2_new_ratinig_12,
                'elo_rating_2.1': player_2_elo_row['elo_rating_2.1'].values[0], 
                'elo_rating_2.2': player_2_elo_row['elo_rating_2.2'].values[0], 
                'elo_rating_2.3': player_2_new_ratinig_23, 
                'elo_rating_3.1': player_2_new_ratinig_31, 
                'elo_rating_3.2': player_2_new_ratinig_32, 
                'elo_sample_all': player_2_elo_row['elo_sample_all'].values[0]+1,
                'elo_sample_clay': player_2_elo_row['elo_sample_clay'].values[0],
                'elo_sample_grass': player_2_elo_row['elo_sample_grass'].values[0],
                'elo_sample_hard': player_2_elo_row['elo_sample_hard'].values[0]+1,
                'total_points_won': tmp_match['loser_games_won'].values[0]+player_2_elo_row['total_points_won'].values[0],
                'total_points_played': tmp_match['games_played'].values[0]+player_2_elo_row['total_points_played'].values[0],
                'total_point_win_perc': (tmp_match['loser_games_won'].values[0]+player_2_elo_row['total_points_won'].values[0])/(tmp_match['games_played'].values[0]+player_2_elo_row['total_points_played'].values[0])
            })

    else:
        print('new surface detected: '+surface)
        
    
    return pd.concat([player_1_elo_row_new, player_2_elo_row_new])
      
### Loop through matches
for i in range(atp_matches.shape[0]):
    players_elo_df = pd.concat([
            players_elo_df,
            process_match(match_id=atp_matches['match_id'].iloc[i], ds=atp_matches['match_date'].iloc[i], player_1=atp_matches['winner_name'].iloc[i], player_2=atp_matches['loser_name'].iloc[i], surface=atp_matches['surface'].iloc[i])
        ]).drop_duplicates(['player', 'ds'])
    
    print('[' + str(round(100*i/atp_matches.shape[0], 2)) + '%] ||| ||| ' + atp_matches['match_id'].iloc[i])
    
### Base to 0
for j in [col for col in list(players_elo_df) if 'elo_rating'  in col]:
    players_elo_df[j] = players_elo_df[j]-1000

### Elo slope
def elo_slope(player, ds, rating_system):
    tmp = players_elo_df.loc[(players_elo_df['player']==player) & (players_elo_df['ds'] <= ds)].sort_values('ds', ascending=False)
    slope = tmp.iloc[0]['elo_rating_'+ rating_system +'_pre'] - tmp.iloc[np.min([4, tmp.shape[0]-1])]['elo_rating_'+ rating_system +'_pre']
    return slope

players_elo_df['elo_rating_3.1_slope_pre'] = players_elo_df.apply(lambda x: elo_slope(player=x['player'], ds=x['ds'], rating_system='3.1'), axis=1)
players_elo_df['elo_rating_3.2_slope_pre'] = players_elo_df.apply(lambda x: elo_slope(player=x['player'], ds=x['ds'], rating_system='3.2'), axis=1)

### Save out
players_elo_df.to_csv('~/Documents/Personal Code/Project_Baseline/men/csv/players_elo_df.csv', index=False)

##################
### Neural Net ###
##################

players_elo_df = pd.read_csv('~/Documents/Personal Code/Project_Baseline/men/csv/players_elo_df.csv')
players_elo_df = players_elo_df.dropna()
players_elo_df.loc[players_elo_df['best_of']==3]

players_elo_df['elo_rating_2.x_pre'] = players_elo_df.apply(lambda x: x['elo_rating_2.1_pre'] if x['surface']=='clay' else (x['elo_rating_2.2_pre'] if x['surface']=='grass' else x['elo_rating_2.3_pre']), axis = 1)
players_elo_df['elo_sample_surface_pre'] = players_elo_df.apply(lambda x: x['elo_sample_'+x['surface']+'_pre'], axis = 1)
players_elo_df['ds_since_2010'] = (pd.to_datetime(players_elo_df['ds']) - datetime(2010,1,1)).dt.days

### Neural Networks
match_win_prob_nn_df = players_elo_df.merge(players_elo_df[['ds','opponent','elo_rating_1.1_pre','elo_rating_3.1_slope_pre','elo_rating_1.2_pre','elo_rating_3.2_slope_pre','elo_rating_2.x_pre','elo_rating_3.1_pre','elo_rating_3.2_pre','elo_sample_all_pre','elo_sample_surface_pre','total_point_win_perc_pre']], left_on = ['player', 'ds'], right_on = ['opponent', 'ds'], suffixes=['', '_opponent']).drop('opponent_opponent', axis=1)
match_win_prob_nn_df['win'] = (match_win_prob_nn_df['outcome']=='W').astype(int)

# clay
clay_regressors = [
        'elo_rating_1.1_pre',
        'elo_rating_1.2_pre',
        'elo_rating_3.1_pre',
        'elo_rating_3.1_slope_pre',
        'elo_rating_3.2_pre',
        'elo_sample_all_pre',
        'elo_rating_2.x_pre',
        'elo_sample_surface_pre',
        'total_point_win_perc_pre',
        'elo_rating_1.1_pre_opponent',
        'elo_rating_1.2_pre_opponent',
        'elo_rating_2.x_pre_opponent',
        'elo_rating_3.1_pre_opponent',
        'elo_rating_3.1_slope_pre_opponent',
        'elo_rating_3.2_pre_opponent',
        'elo_sample_all_pre_opponent',
        'elo_sample_surface_pre_opponent',
        'total_point_win_perc_pre_opponent'
    ]

clay_nn_set = match_win_prob_nn_df.loc[match_win_prob_nn_df['surface']=='clay'].dropna(subset = clay_regressors + ['win']).drop_duplicates()

y1_clay = np.array(clay_nn_set['win'])
x1_clay = np.column_stack(([clay_nn_set[i] for i in clay_regressors]))
x1_clay = sm.add_constant(x1_clay, prepend=True)

X_train_clay, X_val_clay, y_train_clay, y_val_clay = train_test_split(x1_clay, y1_clay)

y_train_clay=np.reshape(y_train_clay, (-1,1))
y_val_clay=np.reshape(y_val_clay, (-1,1))
scaler_x_clay = MinMaxScaler()
scaler_y_clay = MinMaxScaler()
print(scaler_x_clay.fit(X_train_clay))
xtrain_scale_clay=scaler_x_clay.transform(X_train_clay)
xval_scale_clay=scaler_x_clay.transform(X_val_clay)
print(scaler_y_clay.fit(y_train_clay))
ytrain_scale_clay=scaler_y_clay.transform(y_train_clay)
yval_scale_clay=scaler_y_clay.transform(y_val_clay)

model_clay = keras.Sequential()
model_clay.add(layers.Dense(X_train_clay.shape[1], input_dim=X_train_clay.shape[1], kernel_initializer='normal', activation='relu'))
#model_clay.add(Dense(X_train_clay.shape[1], input_dim=X_train_clay.shape[1], kernel_initializer='normal', activation='relu'))
#model_clay.add(Dense(X_train_clay.shape[0], activation='relu'))
model_clay.add(layers.Dense(1, activation='sigmoid')) # 'linear'
model_clay.summary()

model_clay.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_clay=model_clay.fit(xtrain_scale_clay, ytrain_scale_clay, epochs=150, batch_size=150, verbose=1, validation_split=0.5)
predictions_clay = model_clay.predict(xval_scale_clay)

print(history_clay.history.keys())

# "Loss"
plt.plot(history_clay.history['loss'])
plt.plot(history_clay.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions_clay = scaler_y_clay.inverse_transform(predictions_clay)
mean_absolute_error(y_val_clay, predictions_clay)
seaborn.scatterplot([y[0] for y in predictions_clay], [y[0] for y in y_val_clay])
seaborn.distplot([y[0] for y in predictions_clay])

# Confusion matrix
threshold = 0.5
predicted_binary = [1 if p >= threshold else 0 for p in predictions_clay]
confusion = confusion_matrix(y_val_clay, predicted_binary)
print(confusion)
# 10/6/23 accuracy: 64.5%

# grass
grass_regressors = [
        'elo_rating_1.1_pre',
        'elo_rating_1.2_pre',
        'elo_rating_3.1_pre',
        'elo_rating_3.1_slope_pre',
        'elo_rating_3.2_pre',
        'elo_sample_all_pre',
        'elo_rating_2.x_pre',
        'elo_sample_surface_pre',
        'total_point_win_perc_pre',
        'elo_rating_1.1_pre_opponent',
        'elo_rating_1.2_pre_opponent',
        'elo_rating_2.x_pre_opponent',
        'elo_rating_3.1_pre_opponent',
        'elo_rating_3.1_slope_pre_opponent',
        'elo_rating_3.2_pre_opponent',
        'elo_sample_all_pre_opponent',
        'elo_sample_surface_pre_opponent',
        'total_point_win_perc_pre_opponent'
    ]

grass_nn_set = match_win_prob_nn_df.loc[match_win_prob_nn_df['surface']=='grass'].dropna(subset = grass_regressors + ['win']).drop_duplicates()

y1_grass = np.array(grass_nn_set['win'])
x1_grass = np.column_stack(([grass_nn_set[i] for i in grass_regressors]))
x1_grass = sm.add_constant(x1_grass, prepend=True)

X_train_grass, X_val_grass, y_train_grass, y_val_grass = train_test_split(x1_grass, y1_grass)

y_train_grass=np.reshape(y_train_grass, (-1,1))
y_val_grass=np.reshape(y_val_grass, (-1,1))
scaler_x_grass = MinMaxScaler()
scaler_y_grass = MinMaxScaler()
print(scaler_x_grass.fit(X_train_grass))
xtrain_scale_grass=scaler_x_grass.transform(X_train_grass)
xval_scale_grass=scaler_x_grass.transform(X_val_grass)
print(scaler_y_grass.fit(y_train_grass))
ytrain_scale_grass=scaler_y_grass.transform(y_train_grass)
yval_scale_grass=scaler_y_grass.transform(y_val_grass)

model_grass = keras.Sequential()
model_grass.add(layers.Dense(X_train_grass.shape[1], input_dim=X_train_grass.shape[1], kernel_initializer='normal', activation='relu'))
#model_grass.add(Dense(X_train_grass.shape[1], input_dim=X_train_grass.shape[1], kernel_initializer='normal', activation='relu'))
#model_grass.add(Dense(X_train_grass.shape[0], activation='relu'))
model_grass.add(layers.Dense(1, activation='sigmoid')) # 'linear'
model_grass.summary()

model_grass.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_grass=model_grass.fit(xtrain_scale_grass, ytrain_scale_grass, epochs=150, batch_size=150, verbose=1, validation_split=0.5)
predictions_grass = model_grass.predict(xval_scale_grass)

print(history_grass.history.keys())

# "Loss"
plt.plot(history_grass.history['loss'])
plt.plot(history_grass.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions_grass = scaler_y_grass.inverse_transform(predictions_grass)
mean_absolute_error(y_val_grass, predictions_grass)
seaborn.scatterplot([y[0] for y in predictions_grass], [y[0] for y in y_val_grass])
seaborn.distplot([y[0] for y in predictions_grass])

# Confusion matrix
threshold = 0.5
predicted_binary = [1 if p >= threshold else 0 for p in predictions_grass]
confusion = confusion_matrix(y_val_grass, predicted_binary)
print("Confusion Matrix:")
print(confusion)
# 10/6/23 accuracy: 68.9%

# hard
hard_regressors = [
        'elo_rating_1.1_pre',
        'elo_rating_1.2_pre',
        'elo_rating_3.1_pre',
        'elo_rating_3.1_slope_pre',
        'elo_rating_3.2_pre',
        'elo_sample_all_pre',
        'elo_rating_2.x_pre',
        'elo_sample_surface_pre',
        'total_point_win_perc_pre',
        'elo_rating_1.1_pre_opponent',
        'elo_rating_1.2_pre_opponent',
        'elo_rating_2.x_pre_opponent',
        'elo_rating_3.1_pre_opponent',
        'elo_rating_3.1_slope_pre_opponent',
        'elo_rating_3.2_pre_opponent',
        'elo_sample_all_pre_opponent',
        'elo_sample_surface_pre_opponent',
        'total_point_win_perc_pre_opponent'
    ]

hard_nn_set = match_win_prob_nn_df.loc[match_win_prob_nn_df['surface']=='hard'].dropna(subset = hard_regressors + ['win']).drop_duplicates()

y1_hard = np.array(hard_nn_set['win'])
x1_hard = np.column_stack(([hard_nn_set[i] for i in hard_regressors]))
x1_hard = sm.add_constant(x1_hard, prepend=True)

X_train_hard, X_val_hard, y_train_hard, y_val_hard = train_test_split(x1_hard, y1_hard)

y_train_hard=np.reshape(y_train_hard, (-1,1))
y_val_hard=np.reshape(y_val_hard, (-1,1))
scaler_x_hard = MinMaxScaler()
scaler_y_hard = MinMaxScaler()
print(scaler_x_hard.fit(X_train_hard))
xtrain_scale_hard=scaler_x_hard.transform(X_train_hard)
xval_scale_hard=scaler_x_hard.transform(X_val_hard)
print(scaler_y_hard.fit(y_train_hard))
ytrain_scale_hard=scaler_y_hard.transform(y_train_hard)
yval_scale_hard=scaler_y_hard.transform(y_val_hard)

model_hard = keras.Sequential()
model_hard.add(layers.Dense(X_train_hard.shape[1], input_dim=X_train_hard.shape[1], kernel_initializer='normal', activation='relu'))
#model_hard.add(Dense(X_train_hard.shape[1], input_dim=X_train_hard.shape[1], kernel_initializer='normal', activation='relu'))
#model_hard.add(Dense(X_train_hard.shape[0], activation='relu'))
model_hard.add(layers.Dense(1, activation='sigmoid')) # 'linear'
model_hard.summary()

model_hard.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
history_hard=model_hard.fit(xtrain_scale_hard, ytrain_scale_hard, epochs=150, batch_size=150, verbose=1, validation_split=0.5)
predictions_hard = model_hard.predict(xval_scale_hard)

print(history_hard.history.keys())

# "Loss"
plt.plot(history_hard.history['loss'])
plt.plot(history_hard.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

predictions_hard = scaler_y_hard.inverse_transform(predictions_hard)
mean_absolute_error(y_val_hard, predictions_hard)
seaborn.scatterplot([y[0] for y in predictions_hard], [y[0] for y in y_val_hard])
seaborn.distplot([y[0] for y in predictions_hard])

# Confusion matrix
threshold = 0.5
predicted_binary = [1 if p >= threshold else 0 for p in predictions_hard]
confusion = confusion_matrix(y_val_hard, predicted_binary)
print(confusion)
# 10/6/23 accuracy: 58.3%

######################################
### Current Tournament Predictions ###
######################################
currrent_tourn = pd.read_csv('~/Documents/Personal Code/Project_Baseline/men/csv/current_tournament_draw.csv')
currrent_tourn['player'] = currrent_tourn['player'].apply(lambda x: x.replace('_',''))
currrent_tourn.to_csv('~/Documents/Personal Code/Project_Baseline/men/csv/current_tournament_draw.csv', index=False)

currrent_tourn = currrent_tourn.merge(players_elo_df.drop(['surface','outcome','opponent','point_win_perc']+[c for c in list(players_elo_df) if '_pre' in c], axis=1).sort_values('ds').drop_duplicates('player', keep='last'), on = 'player', how = 'left')
currrent_tourn['ds'] = currrent_tourn['ds'].fillna(pd.to_datetime(currrent_tourn['ds']).max())
currrent_tourn['ds_since_2010'] = currrent_tourn['ds_since_2010'].fillna(pd.to_datetime(currrent_tourn['ds_since_2010']).max())

for c in [c for c in list(currrent_tourn) if (('elo_' in c) | ('total_' in c))]:
    currrent_tourn[c] = currrent_tourn[c].fillna(0)
    
currrent_tourn['total_point_win_perc'] = currrent_tourn['total_point_win_perc'].fillna(0.5)

### Elo slope
def elo_slope_pred(player, rating_system):
    tmp = players_elo_df.loc[(players_elo_df['player']==player)].sort_values('ds', ascending=False)
    if tmp.shape[0]>0:
        slope = tmp.iloc[0]['elo_rating_'+ rating_system +'_pre'] - tmp.iloc[np.min([4, tmp.shape[0]-1])]['elo_rating_'+ rating_system +'_pre']
    else:
        slope=0
        
    return slope
   
currrent_tourn['elo_rating_3.1_slope'] = currrent_tourn.apply(lambda x: elo_slope_pred(player=x['player'], rating_system='3.1'), axis=1)
currrent_tourn['elo_rating_3.2_slope'] = currrent_tourn.apply(lambda x: elo_slope_pred(player=x['player'], rating_system='3.2'), axis=1)
    
currrent_tourn.columns = [c+'_pre' if ('elo_' in c or 'win_perc' in c) else c for c in list(currrent_tourn)]        
currrent_tourn['elo_rating_2.x_pre'] = currrent_tourn.apply(lambda x: x['elo_rating_2.1_pre'] if x['surface']=='clay' else (x['elo_rating_2.2_pre'] if x['surface']=='grass' else x['elo_rating_2.3_pre']), axis = 1)
currrent_tourn['elo_sample_surface_pre'] = currrent_tourn.apply(lambda x: x['elo_sample_'+x['surface']+'_pre'], axis = 1)
  
matchups = currrent_tourn.drop(['index', 'ds'], axis = 1).merge(currrent_tourn.drop(['index', 'ds'], axis = 1), on = 'surface', suffixes=['', '_opponent']).drop('surface', axis=1)
matchups = matchups.loc[matchups['player'] != matchups['player_opponent']]
matchups = matchups.drop_duplicates()

### Make predictions
if currrent_tourn['surface'].unique()[0]=='clay':
    x1 = np.column_stack(([matchups[i] for i in clay_regressors]))
    x1 = sm.add_constant(x1, prepend=True)
    x1_scale=scaler_x_clay.transform(x1)
    predictions_clay = model_clay.predict(x1_scale)
    predictions_clay = scaler_y_clay.inverse_transform(predictions_clay)
    matchups['win_prob'] = predictions_clay
    
elif currrent_tourn['surface'].unique()[0]=='grass':
    x1 = np.column_stack(([matchups[i] for i in grass_regressors]))
    x1 = sm.add_constant(x1, prepend=True)
    x1_scale=scaler_x_grass.transform(x1)
    predictions_grass = model_grass.predict(x1_scale)
    predictions_grass = scaler_y_grass.inverse_transform(predictions_grass)
    matchups['win_prob'] = predictions_grass

elif currrent_tourn['surface'].unique()[0]=='hard':
    x1 = np.column_stack(([matchups[i] for i in hard_regressors]))
    x1 = sm.add_constant(x1, prepend=True)
    x1_scale=scaler_x_hard.transform(x1)
    predictions_hard = model_hard.predict(x1_scale)
    predictions_hard = scaler_y_hard.inverse_transform(predictions_hard)
    matchups['win_prob'] = predictions_hard
    
else:
    pass

### Average win prob
matchups = matchups[['player','player_opponent','win_prob']].merge(matchups[['player','player_opponent','win_prob']], left_on = ['player', 'player_opponent'], right_on = ['player_opponent', 'player'], suffixes=['', '_y']).drop(['player_y', 'player_opponent_y'], axis=1)
matchups['win_prob_mean'] = matchups.apply(lambda x: np.mean([x['win_prob'], 1-x['win_prob_y']]), axis=1)

### Set byes to 100%/0% win proob
matchups['win_prob_mean'].loc[matchups['player']=='BYE'] = 0.00
matchups['win_prob_mean'].loc[matchups['player_opponent']=='BYE'] = 1.00

### Save
matchups[['player', 'player_opponent', 'win_prob_mean']].to_csv('~/Documents/Personal Code/Project_Baseline/men/csv/current_tourn_matchups_win_prob.csv', index=False)
