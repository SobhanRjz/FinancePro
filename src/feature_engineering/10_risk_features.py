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
        self.quant_headers = self.GetquantHeaders()
        self.quant_headers = {
            "accept": "application/json, text/plain, */*",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9",
            "authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJ1c2VySWQiOiI2Njk4MDkiLCJpc3MiOiJDcnlwdG9RdWFudCIsImlhdCI6MTc0MTk4OTkwOSwiZXhwIjoxNzQxOTkzNTA5fQ.-CUTovmbegw9oXqDgw3bueQqLPUPrTBhHH2yc4a_B3s",
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
        self.timeframe_map = {
            '1d': 'D',
            '4h': '4h', 
            '1h': 'h',
            '5m': '5min',
            '10m': '5min'
        }
    
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)

    def GetquantHeaders(self):
        """
        Get the headers for the CryptoQuant API request.
        """
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=False)  # Run in the background
            context = browser.new_context(viewport={"width": 1920, "height": 1080},
                                          locale='en-US',
                                        user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0')
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
            page.goto("https://cryptoquant.com/asset/btc/chart/derivatives/long-liquidations?exchange=all_exchange&symbol=all_symbol&window=DAY&sma=0&ema=0&priceScale=log&metricScale=linear&chartStyle=column", wait_until="domcontentloaded")
            
            # Wait for API requests to complete
            page.wait_for_timeout(5000)
            
            browser.close()
            
        return headers
    

    def resample_data(self, df: pd.DataFrame, timeframe: str, start_time: str) -> pd.DataFrame:
        """
        Resample data to the specified timeframe
        
        Args:
            df (pd.DataFrame): DataFrame to resample
            timeframe (str): Timeframe to resample to (e.g., '1d', '4h', '1h', '5m')
            
        Returns:
            pd.DataFrame: Resampled DataFrame
        """
        # Get resample frequency from map, default to daily
        freq = self.timeframe_map.get(timeframe.lower(), 'D')
        df = df.bfill().ffill()
        # For other data, use forward fill
        resampled_df = df.resample(freq, origin=df.index[0]).asfreq()
        resampled_df = resampled_df.ffill().bfill()
            
        # Find the closest timestamp to start_time
        start_timestamp = pd.to_datetime(start_time)
        resampled_df = resampled_df[resampled_df.index >= (start_timestamp - pd.Timedelta(minutes=resampled_df.index.freq.n))]
        
        # Adjust index to align with start_time
        if len(resampled_df) > 0:
            time_diff = resampled_df.index[0] - start_timestamp
            if time_diff.total_seconds() >= 0:
                resampled_df.index = resampled_df.index - pd.Timedelta(minutes=time_diff.total_seconds()/60)


        return resampled_df
    

    def fetch_leverage_ratio_data(self, timeframe:str, start_time: str, end_time: str):
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
                "from": str(int(pd.Timestamp(start_time).timestamp() * 1000)),  # Convert from_time to milliseconds
                "to": str(int(pd.Timestamp(end_time).timestamp() * 1000)),  # Current time in milliseconds
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
                        
                    
                        df = self.resample_data(df, timeframe, start_time)
                        # Check if data is available up to the end time
                        current_date = pd.Timestamp(end_time)
                        last_date = df.index.max()

                        if last_date < current_date:
                            logging.info(f"Missing US Dollar Index data from {last_date.date()} to {current_date.date()}. Forward filling last values.")

                            # Generate future dates based on timeframe
                            freq = self.timeframe_map.get(timeframe.lower(), 'D')
                            future_dates = pd.date_range(start=last_date , end=current_date, freq=freq)
                            # Remove first date since it would duplicate the last existing date
                            future_dates = future_dates[1:]
                            # Get last value
                            last_value = df['leverage_ratio'].iloc[-1]

                            # Create DataFrame with last value repeated
                            future_df = pd.DataFrame({
                                'leverage_ratio': [last_value] * len(future_dates)
                            }, index=future_dates)

                            # Append forward filled data
                            df = pd.concat([df, future_df])

                        # Final forward and backward fill for any remaining gaps
                        df = df.ffill().bfill()
                        df.index.name = 'timestamp'
                        return df
                
                    
                else:
                    logging.error("Unexpected API response structure")
                    return None
            else:
                logging.error(f"API request failed with status code {response.status_code}")
                return None
        except Exception as e:
            logging.error(f"Error fetching leverage ratio data: {e}")
            return None

    def fetch_liquidation_data(self, timeframe: str , start_time: str, end_time: str):
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
                "window": "DAY", # you can not get with h or m
                "from": str(int(pd.Timestamp(start_time).timestamp() * 1000)),  # Convert from_time to milliseconds
                "to": str(int(pd.Timestamp(end_time).timestamp() * 1000)),  # Convert end_time to milliseconds
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
                        
                    
                        df = self.resample_data(df, timeframe, start_time)
                        # Check if data is available up to the end time
                        current_date = pd.Timestamp(end_time)
                        last_date = df.index.max()

                        if last_date < current_date:
                            logging.info(f"Missing US Dollar Index data from {last_date.date()} to {current_date.date()}. Forward filling last values.")

                            # Generate future dates based on timeframe
                            freq = self.timeframe_map.get(timeframe.lower(), 'D')
                            future_dates = pd.date_range(start=last_date , end=current_date, freq=freq)
                            # Remove first date since it would duplicate the last existing date
                            future_dates = future_dates[1:]
                            # Get last value
                            last_value = df['long_liquidation'].iloc[-1]

                            # Create DataFrame with last value repeated
                            future_df = pd.DataFrame({
                                'long_liquidation': [last_value] * len(future_dates)
                            }, index=future_dates)

                            # Append forward filled data
                            df = pd.concat([df, future_df])

                        # Final forward and backward fill for any remaining gaps
                        df = df.ffill().bfill()
                        df.index.name = 'timestamp'
                        return df

                    
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

                df = pd.read_pickle(file_path)
                start_time = df.index.min()
                end_time = df.index.max()
                
            
            df = self.load_risk_data(file_path)
            
            # Fetch leverage ratio data
            if period != "5y" and period != '10y':
                liquidation_data = self.fetch_liquidation_data(timeframe, start_time, end_time)
                leverage_ratio = self.fetch_leverage_ratio_data(timeframe, start_time, end_time)


                dataframes = [df for df in [leverage_ratio, liquidation_data] if df is not None]

                # Merge all dataframes if we have any
                if dataframes:
                    merged_data = dataframes[0]
                    for df in dataframes[1:]:
                        merged_data = pd.merge(merged_data, df,
                                             left_index=True, 
                                             right_index=True,
                                             how='outer')
                else:
                    merged_data = pd.DataFrame()
                    

                merged_data = merged_data.ffill().bfill()
                # Create output directory
                output_dir = os.path.join(self.data_dir.split('/')[0], '10_process_risk_feature')
                os.makedirs(output_dir, exist_ok=True)

                # Create output filename with same pattern
                output_file = os.path.join(output_dir, f'risk_feature_{symbol}_{timeframe}_{period}.pkl.gz')
                
                try:
                    merged_data.to_pickle(output_file)
                    logging.info(f"Saved on-chain features to: {output_file}")
                except Exception as e:
                    logging.error(f"Error saving on-chain features: {e}")


                if df is not None:
                    dfs.append(df)

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Initialize the risk feature extractor
    risk_extractor = RiskFeatureExtractor()
    
    # Process all files and get combined risk metrics
    risk_metrics = risk_extractor.process_all_files()
    