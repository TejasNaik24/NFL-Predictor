import pandas as pd
import os
import sys
import time
from datetime import datetime
from typing import Callable, Optional, Dict, List
import nflreadpy as nfl


def scrape_nfl_data(
    progress_callback: Optional[Callable[[str, str], None]] = None,
    folder_name: str = "data_files",
    target_years: int = 10
) -> Dict:
    """
    Scrape NFL team statistics from NFLVERSE for the most recent seasons.
    
    Args:
        progress_callback: Optional callback function(message: str, type: str) for progress updates
                          type can be: "info", "success", "error"
        folder_name: Directory to save CSV files (default: "data_files")
        target_years: Number of years to scrape (default: 10)
    
    Returns:
        Dictionary with:
            - years_scraped: List of years successfully scraped
            - total_games: Total number of games scraped
            - success: Boolean indicating if scraping completed successfully
            - error: Error message if success=False
    """
    def log(message: str, msg_type: str = "info"):
        """Helper to log messages via callback or print"""
        if progress_callback:
            progress_callback(message, msg_type)
        else:
            print(message)
    
    # Create folder if needed (silent - user doesn't need to know)
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    # Determine current season
    current_year = datetime.now().year
    current_month = datetime.now().month
    current_season = current_year - 1 if current_month in [1, 2] else current_year
    
    # User requested dramatic pause/processing for UI effect
    time.sleep(1.5)
    log("Processing...", "info")
    time.sleep(2)
    
    # Scraping loop
    collected_years = 0
    max_failures = 3
    failures = 0
    year = current_season - 1
    scraping_years = []
    total_games = 0
    
    while collected_years < target_years:
        try:
            df_year = nfl.load_team_stats(seasons=[year]).to_pandas()
            
            file_path = os.path.join(folder_name, f"nflverse_stats_{year}.csv")
            df_year.to_csv(file_path, index=False)
            
            games_count = len(df_year)
            total_games += games_count
            log(f"Success! Got all games for {year} season. Total games scraped: {games_count}", "info")
            
            scraping_years.append(year)
            collected_years += 1
            year -= 1
            failures = 0
            
            if collected_years < target_years:
                time.sleep(2)
        
        except Exception as e:
            failures += 1
            log(f"Failed to get data for {year}, trying previous year...", "error")
            year -= 1
            
            if failures >= max_failures:
                error_msg = "Too many consecutive failures. Stopping scraping."
                log(error_msg, "error")
                return {
                    "years_scraped": scraping_years,
                    "total_games": total_games,
                    "success": False,
                    "error": error_msg
                }
            
            if collected_years < target_years:
                time.sleep(1)
    
    log(f"Finished scraping data for {len(scraping_years)} seasons: {min(scraping_years)}-{max(scraping_years)}", "success")
    
    return {
        "years_scraped": scraping_years,
        "total_games": total_games,
        "success": True,
        "error": None
    }


# ---- Main execution (backward compatibility) ----
if __name__ == "__main__":
    title = "\nDATA SCRAPING FOR NFL PREDICTOR"
    subtitle = "\nScrapes Data Off NFLVERSE"
    line = "=" * (len(subtitle) + 10)
    
    print("\n" + line)
    print(title)
    print(subtitle)
    print("\n" + line + "\n")
    
    result = scrape_nfl_data()
    
    if not result["success"]:
        sys.exit(1)
