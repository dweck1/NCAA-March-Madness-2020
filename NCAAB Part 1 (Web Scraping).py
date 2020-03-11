"""
Author: David Weck
Date Started: 2/7/2020

This part of the project consists of scraping some data from the web
and cleaning it into a DF that I will use to determine win probabilities
for each team in the 2020 NCAA Basketball March Madness Tournament.
"""

#Importing required packages
import pandas as pd
import requests
from bs4 import BeautifulSoup

#Logging in to KenPom
login_url = 'https://kenpom.com/handlers/login_handler.php'

#Creating email and password
payload = {'email' : 'email',
           'password' : 'password'
           }

session = requests.session()
session.get(login_url)
session.post(login_url, data=payload)

##SCRAPING PART 1: Gathering all D1 teams and cleaning into proper URL format
#Setting URL with all D1 teams and pulling source code
teams_url = 'https://kenpom.com'
teams_page = requests.get(teams_url)
teams_soup = BeautifulSoup(teams_page.text, 'html.parser')

#Finding teams in the soup
teams_html = teams_soup.find_all('td', class_ = 'next_left')

#Pulling in team names and putting into URL format
teams = [item.get_text() for item in teams_html]
teams_series = pd.Series(teams)
teams_series.str.replace(' ', '%20' ).str.replace("'", "%27").str.replace('&', '%26')


##SCRAPING PART 2: Looping through all teams and pulling game data
#Setting URL to grab the header from the table
header_url = 'https://kenpom.com/gameplan.php?team=Kansas'
header_page = session.get(header_url)
header_soup = BeautifulSoup(header_page.text, 'html.parser')

#Finding table header
header = header_soup.find('tr', class_='thead2')

#Pulling in column names and keeping only necessary columns
columns_temp = [col.get_text() for col in header.find_all('th')]
to_keep = [3, 7, 8, 9, 10]
columns = [columns_temp[i] for i in to_keep]

#Adding a team column
columns.append('Team')

#Creating empty data frame with column headers
stats_df = pd.DataFrame(columns=columns)
    
##SCRAPING PART 3: Pulling all game data for the past 5 years
years = [2015, 2016, 2017, 2018, 2019]

for year in years:
    for team in teams_series:
        url = f'https://kenpom.com/gameplan.php?team={team}&y={year}'
        page = session.get(url)
        game_soup = BeautifulSoup(page.text, 'html.parser')
        
        #Finding all game data
        raw_game_data = game_soup.find_all('tr', class_=['l', 'w'])
        
        #Pulling game stats into a list
        for data in raw_game_data:
            stats_temp= [stat.get_text() for stat in data.find_all('td')]
            
            #Selecting only D1 games
            if stats_temp[1] == 'NR':
                stats_temp.clear()
                continue
            
            #Selecting only necessary stats and adding team
            stats_to_keep = [3,8,9,10,11]
            stats = [stats_temp[i] for i in stats_to_keep]
            stats.append(team)
            
            ##Converting W/L to binary 1/0
            if stats[0][0] == 'W':
                stats[0] = 1
            else:
                stats[0] = 0
            
            #Creating temporary df to hold each game's data
            temp_df = pd.DataFrame(stats).transpose()
            temp_df.columns = columns
            
            #Concatenating each game's data into final df
            stats_df = pd.concat([stats_df, temp_df], ignore_index=True)
    
#Converting all numeric columns to numeric
stats_df = stats_df.apply(pd.to_numeric, errors='ignore')
    
#Exporting data frame to CSV
stats_df.to_csv(r'.\stats_df.csv')
