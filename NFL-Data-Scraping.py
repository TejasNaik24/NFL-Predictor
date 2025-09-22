import pandas as pd
from datetime import datetime

current_year = datetime.now().year

current_year -= 1

url = f"https://www.pro-football-reference.com/years/{current_year}/games.htm"

tables = pd.read_html(url)