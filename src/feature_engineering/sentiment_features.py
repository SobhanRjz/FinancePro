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
# Get proxy server information from environment variables
proxy_host = os.environ.get('HTTP_PROXY')  # or 'http_proxy' depending on your environment
proxy_port = os.environ.get('HTTP_PROXY_PORT')  # or 'http_proxy_port' depending on your environment

# If the proxy server requires authentication, also get the username and password from environment variables
proxy_username = os.environ.get('HTTP_PROXY_USERNAME')
proxy_password = os.environ.get('HTTP_PROXY_PASSWORD')

# Build the proxy dictionary
# Configure proxies to route traffic through V2Ray (assuming it's listening on localhost:1080)
proxies = {
    "http": "socks5h://127.0.0.1:10808",
    "https": "socks5h://127.0.0.1:10808"
}



class SentimentFeatureExtractor:
    def __init__(self, data_dir='data'):
        self.pytrends = TrendReq(hl='en-US', tz=360)
        self.data_dir = data_dir

    def fetch_google_trends_score(self, keyword="Bitcoin"):
        """Fetch Google Trends search interest for the keyword."""
        self.pytrends.build_payload([keyword], timeframe='now 7-d')
        trends = self.pytrends.interest_over_time()
        score = trends[keyword].iloc[-1] if not trends.empty else 0
        logging.info(f"Google Trends Score for {keyword}: {score}")
        return score
        
    def fetch_daily_google_trends(self, keywords, start_date, end_date):
        """
        Fetch daily Google Trends data by querying 90-day windows (Google's limit for daily data).
        """
        all_results = []
        current_start = start_date

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
                time.sleep(10)
            except Exception as e:
                logging.error(f"Error fetching Google Trends for {timeframe}: {e}")
                time.sleep(10)

        # Combine all the data into one DataFrame
        if all_results:
            full_data = pd.concat(all_results)
            full_data = full_data[~full_data.index.duplicated(keep='last')]  # Remove any duplicate dates
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
                trends = self.fetch_daily_google_trends(keywords, start_date, end_date)

                if trends.empty:
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
                        if date_str in trends.index:
                            
                            trend_value = sum(trends.loc[date_str, keyword] for keyword in keywords)
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
