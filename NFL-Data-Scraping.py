import pandas as pd
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
schedule = schedule[['Winner/tie', 'Loser/tie', 'PtsW', 'PtsL', home_away_col]]

# Renaming the columns
schedule.columns = ['Team', 'Opponent', 'Team_pts', 'Opponent_pts', 'Home/Away']