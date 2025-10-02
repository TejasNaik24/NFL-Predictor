import pandas as pd
import os
import sys
import time
from datetime import datetime

title = "\nDATA SCRAPING FOR NFL PREDICTOR"
subtitle = "\nScraps Data Off The Pro-Football-Reference Website"
line = "=" * (len(subtitle) + 10)

print("\n" + line)
print(title)
print(subtitle)
print("\n" + line + "\n")


folder_name = "data_files"

if not os.path.exists(folder_name):
    print("data_files folder not found, creating one and exporting csv's there...")
    os.makedirs(folder_name)
else:
    print("found data_files folder, exporting csv's there...")


def get_schedule(year):
    url = f"https://www.pro-football-reference.com/years/{year}/games.htm"

    try:
        tables = pd.read_html(url)
    except Exception:
        return None

    schedule = tables[0]
    schedule = schedule[schedule['Week'] != 'Week']

    home_away_col = 'Unnamed: 5' if 'Unnamed: 5' in schedule.columns else 'Location'
    schedule = schedule.rename(columns={home_away_col: 'Home/Away'})
    schedule['Home/Away'] = schedule['Home/Away'].apply(lambda x: 'Away' if x == '@' else 'Home')

    box_score_col = 'Unnamed: 7' if 'Unnamed: 7' in schedule.columns else 'BoxScore'
    schedule = schedule.drop(columns=[box_score_col])

    schedule['Result'] = 'Win'

    opponent_schedule = schedule.copy()
    opponent_schedule.rename(columns={
        'Team':'Opponent', 
        'Opponent':'Team', 
        'Team_pts':'Opponent_pts', 
        'Opponent_pts':'Team_pts'
    }, inplace=True)
    opponent_schedule['Result'] = 'Loss'

    final_schedule = pd.concat([schedule, opponent_schedule], ignore_index=True)
    final_schedule.reset_index(drop=True, inplace=True)

    file_path = os.path.join(folder_name, f"nfl_schedule_{year}.csv")
    final_schedule.to_csv(file_path, index=False)

    games_count = len(final_schedule) // 2
    print(f"-> Success! Got all games for {year} season. Total games scraped: {games_count}")

    return games_count

current_year = datetime.now().year
current_month = datetime.now().month
current_season = current_year - 1 if current_month in [1, 2] else current_year

total_games = 0
year = current_season - 1
target_years = 10
collected_years = 0
max_failures = 3
failures = 0

while collected_years < target_years:
    games_this_year = get_schedule(year)

    if games_this_year is not None:
        total_games += games_this_year
        collected_years += 1
        year -= 1
        failures = 0

        if collected_years < target_years:
            time.sleep(3)

    else:
        failures += 1
        print(f"-> Failed to get data for {year}, trying previous year...")
        year -= 1

        if failures >= max_failures:
            print("\n-> Too many consecutive failures. Stopping program.")
            print("-> Please check your code or if the website structure has changed.")
            sys.exit(1)

        if collected_years < target_years:
            time.sleep(2)