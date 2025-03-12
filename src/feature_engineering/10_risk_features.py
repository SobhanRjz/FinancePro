import os
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
import requests
import time
from playwright.sync_api import sync_playwright
from playwright_stealth import stealth_sync
import cloudscraper
import json

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class RiskFeatureExtractor:
    def __init__(self, data_dir='data/OHLCV'):
        """
        Initialize the RiskFeatureExtractor.
        
        Args:
            data_dir (str): Directory where data files are stored
        """
        self.data_dir = data_dir
        #self.quant_headers = self.GetquantHeaders()
        self.quant_headers = {
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOiI2Njk4MDkiLCJpc3MiOiJDcnlwdG9RdWFudCIsImlhdCI6MTc0MTU1Nzk1NSwiZXhwIjoxNzQxNTYxNTU1fQ.aWVn5wSm2KRXD8j40pMeJrCeAywNEiu3Ey5lzDDebzQ",
            "cache-control": "no-cache",
            "origin": "https://cryptoquant.com",
            "pragma": "no-cache",
            "priority": "u=1, i",
            "referer": "https://cryptoquant.com/",
            "sec-ch-ua": "\"Not(A:Brand\";v=\"99\", \"Microsoft Edge\";v=\"133\", \"Chromium\";v=\"133\"",
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": "\"Windows\"",
            "sec-fetch-dest": "empty",
            "sec-fetch-mode": "cors",
            "sec-fetch-site": "same-site",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0"
        }
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def GetquantHeaders(self):
        """
        Get the headers for the CryptoQuant API request.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Run in the background
            context = browser.new_context()
            page = context.new_page()
            
            # Apply stealth mode to avoid detection
            stealth_sync(page)
            
            # Store the headers from the request
            headers = {}
            
            def capture_headers(request):
                if "api.cryptoquant.com/" in request.url:
                    headers.update(request.headers)
            
            page.on("request", capture_headers)
            
            # Open the website
            page.goto("https://cryptoquant.com/asset/btc/chart/derivatives/long-liquidations", wait_until="networkidle")
            
            # Wait for API requests to complete
            page.wait_for_timeout(5000)
            
            browser.close()
            
        return headers
    
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

    def fetch_leverage_ratio_data(self, from_time: str, period: str):
        """
        Fetch estimated leverage ratio data from CryptoQuant API.
        
        Args:
            from_time (str): Start time for data fetch
            period (str): Time period to fetch data for
            
        Returns:
            pd.DataFrame: DataFrame containing leverage ratio data
        """
        try:
            # API endpoint for estimated leverage ratio
            url = "https://api.cryptoquant.com/live/v3/charts/61a601ac45de34521f1dcc78"
     
            # Parameters
            params = {
                "window": "DAY",
                "from": str(int(pd.Timestamp(from_time).timestamp() * 1000)),  # Convert from_time to milliseconds
                "to": str(int(time.time() * 1000)),  # Current time in milliseconds
                "limit": "70000"
            }
            proxy = {
                "http": "http://127.0.0.1:10808",
                "https": "http://127.0.0.1:10808"
            }
            # Headers for the API request - fixed to remove invalid header names with colons

            
            # Make the request
            response = requests.get(url, params=params, headers=self.quant_headers, proxies=proxy)
            if response.status_code == 200:
                data = response.json()
                
                # Process the data
                if 'result' in data and 'data' in data['result']:
                    df = pd.DataFrame(data['result']['data'])
                    # Rename the first column to datetime if it's not already named 'timestamp'
                    if df.columns[0] != 'timestamp':
                        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                        df.rename(columns={df.columns[1]: 'leverage_ratio'}, inplace=True)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                    if period != '10y' and period != '5y':
                        df = self.resample_data(df, period)
                        return df.ffill().bfill()
                    else:
                        return pd.DataFrame()
                    
                else:
                    logging.error("Unexpected API response structure")
                    return None
            else:
                logging.error(f"API request failed with status code {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error fetching leverage ratio data: {e}")
            return None

    def fetch_liquidation_data(self, from_time: str, period: str):
        """
        Fetch long liquidation data from CryptoQuant API.
        
        Args:
            from_time (str): Start time for data fetch
            period (str): Time period to fetch data for
            
        Returns:
            pd.DataFrame: DataFrame containing liquidation data
        """
        try:
            # API endpoint for long liquidations
            url = "https://api.cryptoquant.com/live/v3/charts/61adc2c66bc0e955292d72b5"
     
            # Parameters
            params = {
                "window": "DAY",
                "from": str(int(pd.Timestamp(from_time).timestamp() * 1000)),  # Convert from_time to milliseconds
                "to": str(int(time.time() * 1000)),  # Current time in milliseconds
                "limit": "70000"
            }
            proxy = {
                "http": "http://127.0.0.1:10808",
                "https": "http://127.0.0.1:10808"
            }

            # Make the request
            response = requests.get(url, params=params, headers=self.quant_headers, proxies=proxy)
            if response.status_code == 200:
                data = response.json()
                
                # Process the data
                if 'result' in data and 'data' in data['result']:
                    df = pd.DataFrame(data['result']['data'])
                    # Rename the first column to datetime if it's not already named 'timestamp'
                    if df.columns[0] != 'timestamp':
                        df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
                        df.rename(columns={df.columns[1]: 'long_liquidation'}, inplace=True)
                        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                        df.set_index('timestamp', inplace=True)
                        
                    if period != '10y' and period != '5y':
                        df = self.resample_data(df, period)
                        return df.ffill().bfill()
                    else:
                        return pd.DataFrame()
                    
                else:
                    logging.error("Unexpected API response structure")
                    return None
            else:
                logging.error(f"API request failed with status code {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error fetching liquidation data: {e}")
            return None
    
    def load_risk_data(self, file_path):
        """
        Load risk data from pickle file.
        
        Args:
            file_path (str): Path to the pickle file
            
        Returns:
            pd.DataFrame: Loaded risk data
        """
        try:
            data = pd.read_pickle(file_path)
            logging.info(f"Successfully loaded risk data from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading risk data: {e}")
            return None
    
    def process_all_files(self):
        """
        Process all risk metric files in the data directory.
        
        Returns:
            pd.DataFrame: Combined risk metrics
        """
        import glob
        import re
        file_pattern = os.path.join(self.data_dir, "*.pkl.gz")
        file_paths = glob.glob(file_pattern)
        
        if not file_paths:
            logging.warning(f"No risk metric files found matching pattern: {file_pattern}")
            return None
        
        dfs = []
        for file_path in file_paths:
            logging.info(f"Processing risk metric file: {file_path}")
            
            # Extract timeframe and period from filename
            file_name = os.path.basename(file_path)
            match = re.search(r'([A-Z]+)_(\w+)_(\w+)', file_name)
            if match:
                symbol, timeframe, period = match.groups()
                logging.info(f"Processing {symbol} data for {timeframe} timeframe over {period}")
                
                # Get period value and unit (e.g. '10y' = 10 years)
                period_value = int(period[:-1])
                period_unit = period[-1].lower()
                
                # Calculate days based on period
                days = 0
                if period_unit == 'd':
                    days = period_value
                elif period_unit == 'w':
                    days = period_value * 7
                elif period_unit == 'm':
                    days = period_value * 30
                elif period_unit == 'y':
                    days = period_value * 365
                
                start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                start_date_str = start_date.strftime('%Y-%m-%d')
            
            df = self.load_risk_data(file_path)
            
            # Fetch leverage ratio data
            liquidation_data = self.fetch_liquidation_data(start_date_str, period=period)
            leverage_ratio = self.fetch_leverage_ratio_data(start_date_str, period=period)



            if df is not None:
                dfs.append(df)
        
        if dfs:
            combined_df = pd.concat(dfs, axis=1)
            return combined_df
        else:
            return None

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the risk feature extractor
    risk_extractor = RiskFeatureExtractor()
    
    # Process all files and get combined risk metrics
    risk_metrics = risk_extractor.process_all_files()
    