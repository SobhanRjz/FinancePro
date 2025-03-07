import pandas as pd
import logging
import os
import glob
import re
from fredapi import Fred
from typing import Optional
import requests
import gzip
import json
from io import BytesIO


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import pyfedwatch.datareader


class MacroFeatureExtractor:
    """Class for extracting macroeconomic features from FRED API"""
    
    def __init__(self, api_key: str = '167da1295b42a6a097932c115926b953 ', data_dir: str = 'data/OHLCV'):
        """
        Initialize MacroFeatureExtractor
        
        Args:
            api_key (str): FRED API key for authentication
            data_dir (str): Directory containing OHLCV data files
        """
        self.api_key = api_key
        self.fred = Fred(api_key=api_key)
        self.data_dir = data_dir
        self.proxy = {
            'http': 'http://127.0.0.1:10808',
            'https': 'http://127.0.0.1:10808'
        }

    def fetch_employment_data(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch employment data from FRED including Non-Farm Payrolls and Unemployment Rate
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with employment data or None if request fails
        """
        try:
            # Fetch Non-Farm Payrolls (PAYEMS) - Total Employment
            nfp = self.fred.get_series('PAYEMS', start_date)
            
            # Fetch Unemployment Rate (UNRATE)
            unemployment = self.fred.get_series('UNRATE', start_date)
            
            # Combine into one DataFrame
            df = pd.DataFrame({
                'Total Employment': nfp,
                'Unemployment Rate': unemployment
            })
            
            df.index.name = 'timestamp'
            
            # Resample to daily frequency and forward fill
            df = df.resample('D').ffill()
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching employment data: {e}")
            return None

    def get_bitcoin_dominance(self, start_date: str, global_market_cap: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Fetch Bitcoin dominance data from CoinCodex API
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with Bitcoin dominance data or None if request fails
        """
        try:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            # Define headers for the API request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Referer": "https://coincheckup.com/",
                "X-Requested-With": "XMLHttpRequest",
                "Accept-Encoding": "br"
            }
            
            # API endpoint for Bitcoin historical data
            url = f"https://coincheckup.com/api/v1/coins/get_coin_history/BTC/{start_date}/{end_date}/1000000"
            
            # Make the request
            response = requests.get(url, proxies=self.proxy, headers=headers)
            response.raise_for_status()
            
            # Parse the JSON response
            data = response.json()
            
            # Extract the Bitcoin data
            btc_data = data.get('BTC', [])
            
            # Create DataFrame from the data
            # Format: [timestamp, price, volume, market_cap]
            df = pd.DataFrame(btc_data, columns=['timestamp', 'price', 'volume', 'btc_market_cap'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df.set_index('timestamp', inplace=True)
            
            # Get global market cap data for the same period
            #global_market_cap = self.get_global_crypto_market_cap(start_date, end_date)
            
            if global_market_cap is not None:
                # Merge the dataframes
                merged_df = pd.merge(df, global_market_cap, left_index=True, right_index=True, how='inner')
                
                # Calculate Bitcoin dominance
                merged_df['btc_dominance'] = (merged_df['btc_market_cap'] / merged_df['Total_market_cap']) * 100
                
                # Select only the dominance column
                dominance_df = merged_df[['btc_dominance']].copy()
                
                # Resample to daily frequency and forward fill
                dominance_df = dominance_df.resample('D').ffill()
                
                return dominance_df
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching Bitcoin dominance data: {e}")
            return None





    def get_global_crypto_market_cap(self, start_date: str, period: str) -> Optional[pd.DataFrame]:
        """
        Fetch global cryptocurrency market capitalization data from CoinCodex API
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with global crypto market cap data or None if request fails
        """
        try:
            # Define headers for the API request
            # Important: Add Browser-like headers to mimic curl/browser request.
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Referer": "https://coincheckup.com/",
                "X-Requested-With": "XMLHttpRequest",
                "Accept-Encoding": "br"  # Explicitly accept Brotli compression
            }

            # Convert start_date to datetime for comparison
            start_date_dt = pd.to_datetime(start_date)
            
            # Set end date to current date
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # API endpoint for global crypto market cap
            url = f"https://coincheckup.com/api/coincodex/get_coin_marketcap/SUM_ALL_COINS/{start_date}/{end_date}/100000"
            
            # Make the request
            response = requests.get(url, proxies=self.proxy, headers=headers)
            response.raise_for_status()  # Raise exception for HTTP errors

                    
            # Parse the JSON response
            data = response.json()
            
            # Extract the market cap data
            market_cap_data = data.get('SUM_ALL_COINS', [])
            
            # Create DataFrame from the data
            # Format: [timestamp, market_cap, volume, other]
            df = pd.DataFrame(market_cap_data, columns=['timestamp', 'Total_market_cap', 'global_volume', 'other'])
            
            df.drop(columns=['other'], inplace=True)
            # Convert Unix timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
            df.index.name = 'timestamp'
            
            # Filter data starting from start_date
            df = df[df.index >= start_date_dt]
            
            # Resample to daily frequency and forward fill
            df = df.resample('D').ffill()
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching global crypto market cap data: {e}")
            return None
    
    def fetch_us_gdp_growth(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch US GDP growth rate data from FRED
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with GDP growth rate data or None if request fails
        """
        try:
            # A191RL1Q225SBEA is the Real GDP Growth Rate series ID (quarterly, percent change from preceding period)
            gdp_growth = self.fred.get_series('A191RL1Q225SBEA', start_date)
            
            # Format as DataFrame
            df = pd.DataFrame(gdp_growth, columns=['gdp_growth_rate'])
            df.index.name = 'timestamp'
            
            # Resample to daily frequency and forward fill
            df = df.resample('D').ffill()
            
            return df
            
        except Exception as e:
            logging.error(f"Error fetching GDP growth rate data: {e}")
            return None
    
    def get_fomc_meetings(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch FOMC meeting dates from Federal Reserve and Fraser Database
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with FOMC meeting dates or None if request fails
        """
        try:
            # Retrieve FOMC meeting data from Fed and Fraser
            df_fed = pyfedwatch.datareader.get_fomc_data_fed(proxy=self.proxy)
            df_fraser = pyfedwatch.datareader.get_fomc_data_fraser(proxy=self.proxy)

            # Combine the two DataFrames, keeping rows from Fed data for duplicates
            result = df_fraser.combine_first(df_fed)
            
            # Convert index to timestamp before filtering
            result.index = pd.to_datetime(result.index)
            
            # Filter for dates after start_date
            result = result[result.index >= pd.Timestamp(start_date)]
            result = self.expand_fomc_dates(result)

            result['timestamp'] = pd.to_datetime(result['timestamp'])
            result.set_index('timestamp', inplace=True)

            return result
            
        except Exception as e:
            logging.error(f"Error fetching FOMC meeting data: {e}")
            return None

    def expand_fomc_dates(self, fomc_data):
        from datetime import datetime, timedelta
        all_days = []

        for index, row in fomc_data.iterrows():
            try:
                year = row['Days'].split(',')[-1].strip()

                # Extract day range (e.g., "April 26-27" or "January 31-February 1")
                month_day_range = row['Days'].split(',')[0].strip()

                # Handle different date formats
                if '-' in month_day_range:
                    parts = month_day_range.split('-')
                    
                    # Clean year if it contains annotations like '(Cancelled)'
                    clean_year = year.split()[0] if '(' in year else year
                    
                    # Case: "January 31-February 1" (month in both parts)
                    if len(parts) == 2 and ' ' in parts[1] and not parts[1].strip()[0].isdigit():
                        month_day_start = parts[0].strip()
                        month_day_end = parts[1].strip()
                        
                        month_start, day_start = month_day_start.split()
                        month_end, day_end = month_day_end.split()
                        
                        start_date = datetime.strptime(f"{month_start} {day_start} {clean_year}", "%B %d %Y")
                        end_date = datetime.strptime(f"{month_end} {day_end} {clean_year}", "%B %d %Y")
                    # Case: "April 26-27" (month only in first part)
                    else:
                        month_day_start, day_end = parts
                        month, day_start = month_day_start.split()
                        start_date = datetime.strptime(f"{month} {day_start} {clean_year}", "%B %d %Y")
                        end_date = datetime.strptime(f"{month} {day_end} {clean_year}", "%B %d %Y")
                else:
                    month = month_day_range.split()[0]
                    day = month_day_range.split()[1]
                    # Handle case where year contains '(Unscheduled)'
                    clean_year = year.split()[0] if '(' in year else year
                    start_date = end_date = datetime.strptime(f"{month} {day} {clean_year}", "%B %d %Y")
                pre_day = start_date - timedelta(days=1)
                post_day = end_date + timedelta(days=1)

                all_days.append({
                    'timestamp': pre_day.date(),
                    'fed_meeting': 0,
                    'pre_meeting': 1,
                    'post_meeting': 0
                })

                all_days.append({
                    'timestamp': start_date.date(),
                    'fed_meeting': 1,
                    'pre_meeting': 0,
                    'post_meeting': 0
                })

                if start_date != end_date:
                    all_days.append({
                        'timestamp': end_date.date(),
                        'fed_meeting': 1,
                        'pre_meeting': 0,
                        'post_meeting': 0
                    })

                all_days.append({
                    'timestamp': post_day.date(),
                    'fed_meeting': 0,
                    'pre_meeting': 0,
                    'post_meeting': 1
                })
            except Exception as e:
                logging.warning(f"Error processing FOMC meeting date: {e} for row: {row}")
                continue

        return pd.DataFrame(all_days) if all_days else pd.DataFrame(columns=['timestamp', 'fed_meeting', 'pre_meeting', 'post_meeting'])

    def get_interest_rates(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch Federal Funds Rate data from FRED
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with interest rate data or None if request fails
        """
        try:
            # FEDFUNDS is the Federal Funds Rate series ID
            interest_rates = self.fred.get_series('DFF', start_date)  # Get daily frequency
            interest_rates = pd.DataFrame(interest_rates, columns=['interest_rate'])
            interest_rates.index.name = 'timestamp'
            # Forward fill missing values since Fed Funds Rate is reported monthly
            interest_rates = interest_rates.resample('D').ffill()
            return interest_rates
        except Exception as e:
            logging.error(f"Error fetching interest rate data: {e}")
            return None

    def get_inflation_rate(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch US inflation rate data from FRED using CPI data
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with inflation rate data or None if request fails
        """
        try:
            # Get CPI data from FRED
            cpi = self.fred.get_series('CPIAUCSL', start_date)
            
            # Convert to DataFrame
            inflation_df = pd.DataFrame(cpi, columns=['CPI'])
            inflation_df.index.name = 'timestamp'
            
            # Calculate year-over-year inflation rate
            inflation_df['inflation_rate'] = inflation_df['CPI'].pct_change(periods=12) * 100
            
            # Resample to daily frequency and forward fill
            inflation_df = inflation_df.resample('D').ffill()
            
            return inflation_df
            
        except Exception as e:
            logging.error(f"Error fetching inflation rate data: {e}")
            return None
        

    def process_all_files(self):
        """Process all OHLCV files and extract macro features"""
        
        # Get all OHLCV data files
        file_pattern = os.path.join(self.data_dir, "*.pkl.gz")
        file_paths = glob.glob(file_pattern)
        
        if not file_paths:
            logging.warning(f"No files found matching pattern: {file_pattern}")
            return None

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            
            # Extract timeframe and period from filename
            match = re.search(r'([A-Z]+)_(\w+)_(\w+)', file_name)
            if match:
                symbol, timeframe, period = match.groups()
                logging.info(f"Processing {symbol} data for {timeframe} timeframe over {period}")
                
                # Get period value and unit (e.g. '10y' = 10 years)
                period_value = int(period[:-1])
                period_unit = period[-1].lower()
                
                # Convert period to days
                if period_unit == 'y':
                    days = period_value * 365
                elif period_unit == 'm':
                    days = period_value * 30
                elif period_unit == 'w':
                    days = period_value * 7
                elif period_unit == 'd':
                    days = period_value
                else:
                    raise ValueError(f"Invalid period unit: {period_unit}")
                
                start_date = pd.Timestamp.now() - pd.Timedelta(days=days)
                start_date_str = start_date.strftime('%Y-%m-%d')

                # Fetch interest rate data

                crypto_market_cap = self.get_global_crypto_market_cap(start_date_str, period)
                bitcoin_dominance = self.get_bitcoin_dominance(start_date_str, crypto_market_cap)
                employment_data = self.fetch_employment_data(start_date_str)
                gdp_growth = self.fetch_us_gdp_growth(start_date_str)
                meetings = self.get_fomc_meetings(start_date_str)
                interest_rates = self.get_interest_rates(start_date_str)
                inflation_rate = self.get_inflation_rate(start_date_str)

                # Merge interest rate and inflation rate data
                df = pd.merge(interest_rates, inflation_rate, left_index=True, right_index=True, how='left')
                
                if interest_rates is None:
                    logging.warning("No interest rate data available - skipping")
                    continue

                # Create output directory
                output_dir = os.path.join(self.data_dir.split('/')[0], 'process_macro_features')
                os.makedirs(output_dir, exist_ok=True)

                # Create output filename with same pattern
                output_file = os.path.join(output_dir, f'macro_features_{symbol}_{timeframe}_{period}.pkl.gz')

                try:
                    interest_rates.to_pickle(output_file)
                    logging.info(f"Saved macro features to: {output_file}")
                except Exception as e:
                    logging.error(f"Error saving macro features: {e}")
            else:
                logging.warning(f"Could not parse pattern from filename: {file_name}")

if __name__ == "__main__":
    # Initialize extractor and process files
    extractor = MacroFeatureExtractor()
    extractor.process_all_files()
