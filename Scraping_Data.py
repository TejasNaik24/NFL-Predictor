import pandas as pd
import os
import time
from datetime import datetime

folder_name = "data_files"

# Checking if folder exists
if not os.path.exists(folder_name):
    print("data_files folder not found creating one and exporting csv's there...")
    os.makedirs(folder_name)
else:
    print("found data_files folder exporting csv's there...")


def get_schedule(year):
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"

    # Read all tables from the page

    try:
        tables = pd.read_html(url)
    except Exception as e:
        print(f"-> Error getting data for {year}, skipping this year. ({e})")
        return 0

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

    # Build file path for this year's CSV
    file_path = os.path.join(folder_name, f"nfl_schedule_{year}.csv")

    # Export CSV
    final_schedule.to_csv(file_path, index=False)

    games_count = len(final_schedule) // 2
    print(f"-> Success! Got all games for {year} season. Total games scraped: {len(final_schedule) // 2}")

    return games_count


# Figure out the most recent completed NFL season (handles Jan/Feb edge case)
current_year = datetime.now().year
current_month = datetime.now().month
current_season = current_year - 1 if current_month in [1, 2] else current_year

years = list(range(current_season - 10, current_season))
total_games = 0  # initialize before the loop

for i, year in enumerate(years):
    games_this_year = get_schedule(year)
    total_games += games_this_year

    if i < len(years) - 1:
        time.sleep(3)

print(f"Successfully collected a total of {total_games} NFL games")
print(f"All csv files have been saved to '{folder_name}'")