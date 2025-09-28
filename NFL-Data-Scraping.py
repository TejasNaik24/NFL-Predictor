import pandas as pd
from datetime import datetime

current_year = datetime.now().year

current_year -= 1

url = f"https://www.pro-football-reference.com/years/{current_year}/games.htm"

# Read all tables from the page
tables = pd.read_html(url)

# Main schedule table is usually the first one
schedule = tables[0]

# Drop repeated header rows
schedule = schedule[schedule['Week'] != 'Week']