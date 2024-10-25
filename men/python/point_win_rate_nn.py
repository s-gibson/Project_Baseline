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
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

### Read in data
matches = pd.read_csv('~/Documents/Personal Code/Project_Baseline/csv/matches.csv')
points = pd.read_csv('~/Documents/Personal Code/Project_Baseline/csv/points.csv')
overview = pd.read_csv('~/Documents/Personal Code/Project_Baseline/csv/overview.csv')

def year_match_reader(year):
    df = pd.read_csv('https://raw.githubusercontent.com/JeffSackmann/tennis_atp/master/atp_matches_'+ str(year) + '.csv', encoding = 'latin-1', quoting=csv.QUOTE_NONE)
    df['year'] = year
    return df

atp_matches = year_match_reader(2023)

### Clean data
# Names
matches['Player 1'] = matches['Player 1'].str.strip()
matches['Player 2'] = matches['Player 2'].str.strip()

# Date
matches['Date'] = pd.to_datetime(matches['Date'], format='%Y%m%d')

### Match outcomes
match_winner = points[['match_id', 'Pt', 'PtWinner']].sort_values(['match_id', 'Pt']).drop_duplicates(subset='match_id', keep='last')[['match_id', 'PtWinner']]
match_winner.columns = ['match_id', 'MatchWinner']
points['PtWinner1'] = (points['PtWinner']==1).astype(int)
points['PtWinner2'] = (points['PtWinner']==2).astype(int)
points_won = points[['match_id', 'PtWinner1', 'PtWinner2']].groupby(['match_id'], as_index = False).sum()
points_won['PtsPlayed'] = (points_won['PtWinner1']+points_won['PtWinner2'])
points_won['PtWinPerc1'] = points_won['PtWinner1']/points_won['PtsPlayed']
points_won['PtWinPerc2'] = points_won['PtWinner2']/points_won['PtsPlayed']
matches = matches.merge(match_winner, on = 'match_id').merge(points_won, on = 'match_id')
matches = matches.sort_values('Date')
matches['year'] = pd.to_datetime(matches['Date']).dt.year
matches['MatchWinnerName'] = matches.apply(lambda x: x['Player 1'] if x['MatchWinner']==1 else x['Player 2'], axis = 1)
matches['MatchLoserName'] = matches.apply(lambda x: x['Player 1'] if x['MatchWinner']==2 else x['Player 2'], axis = 1)

### Merge
atp_matches['tourney_date'] = pd.to_datetime(atp_matches['tourney_date'], format='%Y%m%d')
merged_matches = atp_matches.merge(matches[['year', 'MatchWinnerName','MatchLoserName','Date','PtWinPerc1','PtWinPerc2','Player 1', 'Player 2']], left_on=['year','winner_name','loser_name'], right_on = ['year', 'MatchWinnerName','MatchLoserName'])
merged_matches['date_diff'] = (merged_matches['Date'] - merged_matches['tourney_date']).dt.days
merged_matches = merged_matches.loc[(merged_matches['date_diff']>=0) & (merged_matches['date_diff']<=30)].sort_values('date_diff').drop_duplicates(['tourney_date','winner_name','loser_name'])

### Score cleaner
def score_cleaner(score, player_is_winner=True):
    return int(score.split('(')[0].split('-')[1-player_is_winner])
        
merged_matches['winner_games_won'] = merged_matches['score'].apply(lambda x: np.sum([score_cleaner(score = s, player_is_winner=True) for s in x.split(' ') if s != 'RET']))
merged_matches['loser_games_won'] = merged_matches['score'].apply(lambda x: np.sum([score_cleaner(score = s, player_is_winner=False) for s in x.split(' ') if s != 'RET']))
merged_matches['winner_point_win_perc'] = merged_matches.apply(lambda x: x['PtWinPerc1'] if x['Player 1']==x['MatchWinnerName'] else x['PtWinPerc2'], axis = 1)
merged_matches['loser_point_win_perc'] = merged_matches.apply(lambda x: x['PtWinPerc1'] if x['Player 1']==x['MatchLoserName'] else x['PtWinPerc2'], axis = 1)

### Neural Network
concat_matches = pd.concat([
        pd.DataFrame({
                'point_win_perc': merged_matches['winner_point_win_perc'],
                'games_won': merged_matches['winner_games_won'],
                'games_lost': merged_matches['loser_games_won'],
                'sets': merged_matches['best_of']
            }),
        pd.DataFrame({
                'point_win_perc': merged_matches['loser_point_win_perc'],
                'games_won': merged_matches['loser_games_won'],
                'games_lost': merged_matches['winner_games_won'],
                'sets': merged_matches['best_of']
            })
    ])

regressors = [
        'games_won',
        'games_lost',
        'sets'
    ]

nn_set = concat_matches

y1 = np.array(nn_set['point_win_perc'])
x1 = np.column_stack(([nn_set[i] for i in regressors]))
x1 = sm.add_constant(x1, prepend=True)

X_train, X_val, y_train, y_val = train_test_split(x1, y1)

y_train=np.reshape(y_train, (-1,1))
y_val=np.reshape(y_val, (-1,1))
scaler_x = MinMaxScaler()
scaler_y = MinMaxScaler()
print(scaler_x.fit(X_train))
xtrain_scale=scaler_x.transform(X_train)
xval_scale=scaler_x.transform(X_val)
print(scaler_y.fit(y_train))
ytrain_scale=scaler_y.transform(y_train)
yval_scale=scaler_y.transform(y_val)

model = Sequential()
model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#model.add(Dense(X_train.shape[1], input_dim=X_train.shape[1], kernel_initializer='normal', activation='relu'))
#model.add(Dense(X_train.shape[0], activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
history=model.fit(xtrain_scale, ytrain_scale, epochs=90, batch_size=150, verbose=1, validation_split=0.5)
predictions = model.predict(xval_scale)

print(history.history.keys())

# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

########################
### Make Predictions ###
########################
atp_matches['score'] = [s.split('[')[0].strip() for s in atp_matches['score']]
atp_matches['winner_games_won'] = atp_matches['score'].apply(lambda x: np.sum([score_cleaner(score = s, player_is_winner=True) for s in x.split(' ') if s not in ('RET', 'W/O', 'Def.')]))
atp_matches['loser_games_won'] = atp_matches['score'].apply(lambda x: np.sum([score_cleaner(score = s, player_is_winner=False) for s in x.split(' ') if s not in ('RET', 'W/O', 'Def.')]))

prediction_concat_matches = pd.concat([
        pd.DataFrame({
                'tourney_id': atp_matches['tourney_id'],
                'match_num': atp_matches['match_num'],
                'side': 'winner',
                'games_won': atp_matches['winner_games_won'],
                'games_lost': atp_matches['loser_games_won'],
                'sets': atp_matches['best_of']
            }),
        pd.DataFrame({
                'tourney_id': atp_matches['tourney_id'],
                'match_num': atp_matches['match_num'],
                'side': 'loser',
                'games_won': atp_matches['loser_games_won'],
                'games_lost': atp_matches['winner_games_won'],
                'sets': atp_matches['best_of']
            })
    ])

x1 = np.column_stack(([prediction_concat_matches[i] for i in regressors]))
x1 = sm.add_constant(x1, prepend=True)
x1_scale=scaler_x.transform(x1)
predictions = model.predict(x1_scale)
predictions = scaler_y.inverse_transform(predictions)
prediction_concat_matches['pred_point_win_perc'] = predictions

