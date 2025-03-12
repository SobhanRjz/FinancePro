import requests
import snscrape.modules.twitter as sntwitter
import praw
import pandas as pd
from datetime import datetime, timedelta
from textblob import TextBlob
from pytrends.request import TrendReq
import logging
import glob
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


import requests
import time
import json
import os
import re
# Get proxy server information from environment variables
proxy_host = os.environ.get('HTTP_PROXY')  # or 'http_proxy' depending on your environment
proxy_port = os.environ.get('HTTP_PROXY_PORT')  # or 'http_proxy_port' depending on your environment

# If the proxy server requires authentication, also get the username and password from environment variables
proxy_username = os.environ.get('HTTP_PROXY_USERNAME')
proxy_password = os.environ.get('HTTP_PROXY_PASSWORD')

# Build the proxy dictionary
# Configure proxies to route traffic through V2Ray (assuming it's listening on localhost:1080)
proxies = {
    "http": "127.0.0.1:10808",
    "https": "127.0.0.1:10808"
}


class SentimentFeatureExtractor:
    def __init__(self, data_dir='data'):
        Google_proxies = [{
            "http": "http://127.0.0.1:10808",
            "https": "http://127.0.0.1:10808"
        }]
        
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.data_dir = data_dir
        self.Github_api_key = "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"

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resample data to the specified timeframe
        
        Args:
            df (pd.DataFrame): DataFrame to resample
            timeframe (str): Timeframe to resample to (e.g., '1d', '4h', '1h', '5m')
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        # Handle different timeframes for resampling
        if timeframe.lower() == '1d':
            resampled_df = df.resample('D').ffill()
        elif timeframe.lower() == '4h':
            resampled_df = df.resample('4h').ffill()
        elif timeframe.lower() == '1h':
            resampled_df = df.resample('h').ffill()
        elif timeframe.lower() == '5m':
            resampled_df = df.resample('5min').ffill()
        else:
            # Resample to daily frequency and forward fill for other timeframes
            resampled_df = df.resample('D').ffill()
            
        return resampled_df
    def fetch_github_activity(self, repo="bitcoin/bitcoin", start_date="2015-01-01", timeframe="D"):
        """
        Fetch GitHub repository commits and activity metrics for blockchain projects.
        
        Args:
            repo (str): Repository name in format 'owner/repo'
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format (defaults to current date)
            api_key (str): GitHub API key for authentication
            
        Returns:
            pd.DataFrame: DataFrame with daily GitHub activity metrics
        """
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Authentication
        headers = {
            "Authorization": f"token {self.Github_api_key}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        # Date range
        date_range = pd.date_range(start=start_date, end=end_date)
        daily_commit_counts = []
        
        # Fetch commits for each day
        for single_date in date_range:
            day_str = single_date.strftime("%Y-%m-%d")
            
            url = f"https://api.github.com/search/commits?q=repo:{repo}+committer-date:{day_str}..{day_str}"
            
            while True:
                try:
                    response = requests.get(url, headers=headers, proxies=proxies)
                    
                    if response.status_code == 200:
                        commit_data = response.json()
                        commit_count = commit_data.get("total_count", 0)
                        
                        daily_commit_counts.append({
                            "date": day_str, 
                            "commit_count": commit_count
                        })
                        logging.info(f"✅ {day_str}: {commit_count} commits")
                        break  # Successfully got data, exit the loop
                    else:
                        logging.error(f"❌ Error {response.status_code} for {day_str}: {response.text}")
                        time.sleep(5)  # Respect GitHub rate limits
                        continue
                
                except Exception as e:
                    logging.error(f"Error fetching {day_str}: {e}")
                    time.sleep(5)  # Add delay before retry
                    continue
        
        # Convert to DataFrame
        if not daily_commit_counts:
            logging.warning(f"No commits found for {repo} in the specified date range")
            return None
            
        df = pd.DataFrame(daily_commit_counts)
        df['timestamp'] = pd.to_datetime(df['date'])
        df.set_index('timestamp', inplace=True)
        df = self.resample_data(df, timeframe)

        df.ffill().bfill()
        df.drop(columns=['date'], inplace=True)
        
        logging.info(f"Successfully processed {len(df)} days of activity data")
        return df
    def fetch_google_trends_score(self, keyword="Bitcoin"):
        """Fetch Google Trends search interest for the keyword."""
        self.pytrends.build_payload([keyword], timeframe='now 7-d')
        trends = self.pytrends.interest_over_time()
        score = trends[keyword].iloc[-1] if not trends.empty else 0
        logging.info(f"Google Trends Score for {keyword}: {score}")
        return score
        
    def fetch_daily_google_trends(self, keywords, start_date, timeFrameMain):
        """
        Fetch daily Google Trends data by querying 90-day windows (Google's limit for daily data).
        """
        all_results = []
        current_start = start_date
        end_date = datetime.now().strftime('%Y-%m-%d')
        while current_start < end_date:
            try:
                current_end = current_start + timedelta(days=90)
                if current_end > end_date:
                    current_end = end_date

                timeframe = f"{current_start.strftime('%Y-%m-%d')} {current_end.strftime('%Y-%m-%d')}"
                logging.info(f"Fetching Google Trends for timeframe: {timeframe}")

                self.pytrends.build_payload(keywords, timeframe=timeframe, geo='')

                trends = self.pytrends.interest_over_time()
                if not trends.empty:
                    all_results.append(trends)

                current_start = current_end + timedelta(days=1)

                # Avoid rate limits
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error fetching Google Trends for {timeframe}: {e}")
                time.sleep(10)

        # Combine all the data into one DataFrame
        if all_results:
            full_data = pd.concat(all_results)
            full_data = full_data[~full_data.index.duplicated(keep='last')]  # Remove any duplicate dates
            full_data = self.resample_data(full_data, timeFrameMain)
            full_data.ffill().bfill()
            full_data['Trend_Sum'] = full_data.sum(axis=1)
            full_data.drop(columns=keywords, inplace=True)
            full_data.drop(columns=['isPartial'], inplace=True)

            return full_data
        else:
            return pd.DataFrame()

    def process_all_files(self):
        """
        Process all order book files, and fetch daily Google Trends data to match each file's timeframe.
        """
        keywords = ["Bitcoin", "Ethereum", "crypto", "buy Bitcoin", "sell Bitcoin"]

        # Find all order book data files
        file_pattern = os.path.join(self.data_dir, 'OHLCV', "*.pkl.gz")
        file_paths = glob.glob(file_pattern)

        if not file_paths:
            logging.warning(f"No order book files found matching: {file_pattern}")
            return
        # Output directory
        output_dir = os.path.join(self.data_dir)
        os.makedirs(output_dir, exist_ok=True)

        for file_path in file_paths:
            logging.info(f"Processing file: {file_path}")

            file_name = os.path.basename(file_path)
            
            # Extract symbol, timeframe and period from filename
            match = re.search(r'([A-Z]+)_(\w+)_(\w+)', file_name)
            if match:
                symbol, timeframe, period = match.groups()
                logging.info(f"Processing {symbol} data for {timeframe} timeframe over {period}")

            try:
                # Load order book data
                data = pd.DataFrame()
                #data = pd.read_pickle(file_path)
                logging.info(f"Loaded order book data from: {file_path}")
            except Exception as e:
                logging.error(f"Failed to load data from {file_path}: {e}")
                continue

            try:
                # Detect timeframe based on file name and extract start date
                if '5y' in file_name:
                    start_date = datetime.now() - timedelta(days=5 * 365)
                elif '3y' in file_name:
                    start_date = datetime.now() - timedelta(days=3 * 365)
                elif '1y' in file_name:
                    start_date = datetime.now() - timedelta(days=365)
                elif '10y' in file_name:
                    start_date = datetime.now() - timedelta(days=10 * 365)
                else:
                    start_date = datetime.now() - timedelta(days=7)

                end_date = datetime.now()

                # Fetch daily Google Trends data
                Google_trends = self.fetch_daily_google_trends(keywords, start_date, end_date)
                daily_github_activity = self.fetch_github_activity(repo="bitcoin/bitcoin", start_date=start_date, timeframe=timeframe)

                if Google_trends.empty:
                    logging.warning(f"No Google Trends data found for {file_name}")
                    continue

                # If it's intraday (5m, 1h, 4h), map trends data by day
                if any(tf in file_name for tf in ['5m', '1h', '4h']):
                    logging.info(f"Mapping daily Google Trends to intraday data: {file_name}")

                    data['google_trend'] = None  # Add column for trend data

                    # Map each date from order book to the corresponding trend value
                    unique_dates = pd.Series([d.date() for d in data.index]).unique()

                    for date in unique_dates:
                        date_str = date.strftime('%Y-%m-%d')
                        if date_str in Google_trends.index:
                            
                            trend_value = sum(Google_trends.loc[date_str, keyword] for keyword in keywords)
                            data.loc[data.index.date == date, 'google_trend'] = trend_value
                            logging.info(f"Assigned trend {trend_value} to {date_str}")
                        else:
                            logging.warning(f"No trend data found for {date_str}")

                # Create output directories if they don't exist
                os.makedirs(os.path.join(output_dir, 'process_sentiment_features'), exist_ok=True)
            
                logging.info(f"Created output directories for sentiment features")
                # Save trends data
                output_file = os.path.join(output_dir, 'process_sentiment_features', 'Sentiment_Features_' + '_'.join(file_name.split('_')[-3:]))
                trends.to_pickle(output_file)
                logging.info(f"Saved trends data to {output_file}")

                # Optional: save combined order book + trends data if needed
                combined_output_file = os.path.join(output_dir, f"combined_{file_name}")
                data.to_pickle(combined_output_file)
                logging.info(f"Saved combined data to {combined_output_file}")

            except Exception as e:
                logging.error(f"Error processing Google Trends for {file_name}: {e}")


if __name__ == "__main__":
    collector = SentimentFeatureExtractor()

    # Process all order book files and save Google Trends data
    collector.process_all_files()
