import pandas as pd
import os
from datetime import datetime

current_year = datetime.now().year

current_year -= 1

url = f"https://www.pro-football-reference.com/years/{current_year}/games.htm"

# Read all tables from the page
tables = pd.read_html(url)

# Main schedule table
schedule = tables[0]

# Droping the repeated header rows
schedule = schedule[schedule['Week'] != 'Week']

# The home/away might be named 'Unnamed: 5'
home_away_col = 'Unnamed: 5' if 'Unnamed: 5' in schedule.columns else 'Location'

# Keeping the columns we care about
schedule = schedule.rename(columns={home_away_col: 'Home/Away'})

# Fixing Home/Away: blank means home team, '@' means away team
schedule['Home/Away'] = schedule['Home/Away'].apply(lambda x: 'Away' if x == '@' else 'Home')

# Identifying the boxscore column
box_score_col = 'Unnamed: 7' if 'Unnamed: 7' in schedule.columns else 'BoxScore'

# Drop the boxscore column
schedule = schedule.drop(columns=[box_score_col])

# Result for the team listed in 'Team' column is always a win
schedule['Result'] = 'Win'

# Creating the opponent view
opponent_schedule = schedule.copy()
opponent_schedule.rename(columns={'Team':'Opponent', 'Opponent':'Team', 'Team_pts':'Opponent_pts', 'Opponent_pts':'Team_pts'}, inplace=True)
opponent_schedule['Result'] = 'Loss'

# Combine both to get one row per team per game
final_schedule = pd.concat([schedule, opponent_schedule], ignore_index=True)

# Cleaning the index values
final_schedule.reset_index(drop=True, inplace=True)


final_schedule.to_csv(f"nfl_schedule_{current_year}.csv", index=False)
