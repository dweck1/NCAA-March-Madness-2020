"""
Author: David Weck
Date Started: 3/10/2020

This part of the project consists of gathering the 68 NCAA Men's basketball
teams in the 2020 March Madness tournament. I will take their season
averages for the features involved in the model and use them to predict win
probability using the logistic regression model found in the Log Reg Fitting
script and the Gradient Boosting Classifier model found in the 
GradientBoosting Fit script

"""

#Importing required packages
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

##PART 1: Reading in cleaned dataset and preparing for model fitting
#Reading in the data and creating feature and target dfs
stats = pd.read_csv(r'.\stats_final.csv', index_col=0)
features = stats[['eFG%', 'TO%', 'OR%', 'FTR']]
target = stats['Result']

#Scaling all features
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

#Splitting data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(features_scaled, target, 
                                                    test_size=.2,
                                                    random_state=123
                                                    )

##PART2: Scraping web data for each team in the 2020 March Madness Tournament
#Logging in to KenPom
login_url = 'https://kenpom.com/handlers/login_handler.php'

#Creating email and password
payload = {'email' : 'email',
           'password' : 'password'
           }

session = requests.session()
session.get(login_url)
session.post(login_url, data=payload)

#Finding table header
header_url = 'https://kenpom.com/gameplan.php?team=Kansas'
header_page = session.get(header_url)
header_soup = BeautifulSoup(header_page.text, 'html.parser')
header = header_soup.find('tr', class_='thead2')

#Pulling in column names and keeping only necessary columns
columns_temp = [col.get_text() for col in header.find_all('th')]
to_keep = [7, 8, 9, 10]
columns = [columns_temp[i] for i in to_keep]

#Adding a team column
columns.append('Team')

#Creating empty data frame where I will put data
team_data = pd.DataFrame(columns=columns)

#Creating list of teams in the tournament
teams = ['Kansas', 'Florida', 'Duke', 'Maryland',
         'Gonzaga', 'Dayton','Baylor', 'Xavier', 
         'Wisconsin', 'Michigan+St.', 'Creighton',
         'Florida+St.', 'San+Diego+St.']

#Looping through teams and getting data
for team in teams:
        url = f'https://kenpom.com/gameplan.php?team={team}'
        page = session.get(url)
        game_soup = BeautifulSoup(page.text, 'html.parser')
        
        #Finding all data
        raw_game_data = game_soup.find_all('tr', class_=['l', 'w'])
        
        #Pulling stats into a list
        for data in raw_game_data:
            stats_temp= [stat.get_text() for stat in data.find_all('td')]
            
            #Selecting only D1 games
            if stats_temp[1] == 'NR':
                stats_temp.clear()
                continue
            
            #Selecting only necessary stats and adding team
            stats_to_keep = [8,9,10,11]
            stats = [stats_temp[i] for i in stats_to_keep]
            stats.append(team)
            
            #Creating temporary df to hold data
            temp_df = pd.DataFrame(stats).transpose()
            temp_df.columns = columns
            
            #Concatenating into final df
            team_data = pd.concat([team_data, temp_df], ignore_index=True)
    
#Converting all numeric columns to numeric
team_data = team_data.apply(pd.to_numeric, errors='ignore')

#Getting each team's season average for each stat
team_data_avg = team_data.groupby(by='Team', sort=False).mean()

##PART 3: Fitting models
#Models were selected using cross-validation 
#Details can be found in 'Log Reg Fitting' and 'GradientBoosting Fit' scripts

#Scaling the teams stats using training data scale
X_teams = scaler.transform(team_data_avg)

#Instantiating each classifier
logreg_clf = LogisticRegression(C=.6, solver = 'lbfgs')

GB_clf = GradientBoostingClassifier(learning_rate=.2,
                                    n_estimators=200,
                                    min_samples_split=.1,
                                    min_samples_leaf=.04,
                                    max_features=3,
                                    max_depth=3,
                                    subsample=.9)

#Fitting each classifier to the training data
logreg_clf.fit(X_train, y_train)

GB_clf.fit(X_train, y_train)

#Predicting probability of winning using each model
log_reg_probs = logreg_clf.predict_proba(X_teams)

GBC_probs = GB_clf.predict_proba(X_teams)

#Selecting win probabilities from each model
log_reg_win_prob = np.around(log_reg_probs[:,-1], 5)
GBC_win_prob = np.around(GBC_probs[:,-1], 5)

#Creating final table of teams sorted by highest win probability
teams_array = np.array(teams)

final_probs = np.vstack((teams_array, log_reg_win_prob, GBC_win_prob))
final_probs = pd.DataFrame(final_probs.transpose())

final_probs.columns = ['Team', 'Log Reg Win Prob', 'GBC Win Prob']

final_probs.sort_values(by='Log Reg Win Prob', ascending=False, 
                        inplace=True, ignore_index=True)
print(final_probs)

