import pandas as pd
import os
import sys
import time
from datetime import datetime
import nflreadpy as nfl

title = "\nDATA SCRAPING FOR NFL PREDICTOR"
subtitle = "\nScrapes Data Off NFLVERSE"
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

current_year = datetime.now().year
current_month = datetime.now().month
current_season = current_year - 1 if current_month in [1, 2] else current_year

target_years = 10
collected_years = 0
max_failures = 3
failures = 0
year = current_season - 1
scraping_years = []

while collected_years < target_years:
    try:
        df_year = nfl.load_team_stats(seasons=[year]).to_pandas()
        df_year = df_year.drop(columns=['team'], errors='ignore')

        file_path = os.path.join(folder_name, f"nflverse_stats_{year}.csv")
        df_year.to_csv(file_path, index=False)

        print(f"-> Success! Got all games for {year} season. Total games scraped: {len(df_year)}")

        scraping_years.append(year)
        collected_years += 1
        year -= 1
        failures = 0

        if collected_years < target_years:
            time.sleep(2)

    except Exception as e:
        failures += 1
        print(f"-> Failed to get data for {year}, trying previous year...")
        year -= 1

        if failures >= max_failures:
            print("\n-> Too many consecutive failures. Stopping program.")
            sys.exit(1)

        if collected_years < target_years:
            time.sleep(1)

print(f"\n-> Finished scraping NFLVERSE stats for years: {scraping_years}")
