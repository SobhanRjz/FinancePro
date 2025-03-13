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
from datetime import datetime
from fake_useragent import UserAgent

from statsmodels.tsa.arima.model import ARIMA
import logging
import pandas as pd
from typing import Optional
import time 

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
        self.timeframe_map = {
            '1d': 'D',
            '4h': '4h', 
            '1h': 'h',
            '5m': '5min'
        }
    

    def resample_data(self, df: pd.DataFrame, timeframe: str, start_time : str, ismeeting: bool = False) -> pd.DataFrame:
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

        if ismeeting:
            # For FOMC meeting data, use vectorized operations instead of loops
            # Get the initial timestamp's time component to preserve in resampling

            resampled_df = df.resample(freq).asfreq()

                
            daily_df = df.resample('D').asfreq()
            
            # Create a mapping from dates to daily index values
            date_to_idx = {idx.date(): idx for idx in daily_df.index}
            
            # Create a mapping dictionary for faster lookups
            date_mapping = {}
            for idx in resampled_df.index:
                if idx.date() in date_to_idx:
                    date_mapping[idx] = date_to_idx[idx.date()]
            
            # Apply the mapping to all columns at once
            for col in resampled_df.columns:
                for target_idx, source_idx in date_mapping.items():
                    resampled_df.loc[target_idx, col] = daily_df.loc[source_idx, col]
            
            # Fill remaining NaN values with 0
            resampled_df = resampled_df.fillna(0)

        else:
            # For other data, use forward fill
            resampled_df = df.resample(freq).asfreq()
            resampled_df = resampled_df.ffill().bfill()
            
        # Find the closest timestamp to start_time
        start_timestamp = pd.to_datetime(start_time)
        resampled_df = resampled_df[resampled_df.index >= (start_timestamp - pd.Timedelta(minutes=resampled_df.index.freq.n))]
        
        # Adjust index to align with start_time
        if len(resampled_df) > 0:
            time_diff = resampled_df.index[0] - start_timestamp
            resampled_df.index = resampled_df.index - pd.Timedelta(minutes=time_diff.total_seconds()/60)


        return resampled_df
    

    def get_us_dollar_index(self, start_date: str = '20151101', end_data_str: str = '20151101',  timeframe: str = '1d') -> Optional[pd.DataFrame]:
        """
        Fetch US Dollar Index (DXY) data using Playwright in 3-month chunks
        
        Args:
            start_date (str): Start date in YYYYMMDD format (default: '20151101')
            end_date (str): End date in YYYYMMDD format (default: current date)
            timeframe (str): Timeframe to resample to (e.g., '1d', '4h', '1h', '5m')
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with US Dollar Index data or None if request fails
        """
        try:
            
            # Convert dates to timestamps
            start_ts = pd.Timestamp(start_date)
            end_ts = pd.Timestamp(end_data_str)
            
            # Initialize empty list to store DataFrames
            dfs = []
            
            # Split into 3-month chunks
            current_start = start_ts
            while current_start < end_ts:
                # Calculate chunk end date (3 months from start)
                chunk_end = min(current_start + pd.DateOffset(months=3), end_ts)
                
                # Convert to unix timestamps
                chunk_start_ts = int(current_start.timestamp())
                chunk_end_ts = int(chunk_end.timestamp())
                
                # Use Playwright to fetch data for this chunk
                from playwright.sync_api import sync_playwright
                
                with sync_playwright() as p:
                    browser = p.chromium.launch(headless=True)
                    page = browser.new_page()
                    
                    page.set_extra_http_headers({
                        'User-Agent': UserAgent().random,
                        'Accept': 'application/json, text/javascript, */*; q=0.01',
                        'Accept-Language': 'en-US,en;q=0.9',
                        'Referer': 'https://www.investing.com',
                        'Origin': 'https://www.investing.com'
                    })
                    
                    if timeframe == '1d':
                        resolution = 'D'
                    elif timeframe == '4h':
                        resolution = '240'
                    elif timeframe == '1h':
                        resolution = '60'
                    elif timeframe == '5m':
                        resolution = '50'
                        
                    url = f"https://tvc4.investing.com/ea89217fb58266dde61d40b25e07c0d0/1741461861/1/1/8/history?symbol=942611&resolution={resolution}&from={chunk_start_ts}&to={chunk_end_ts}"
                    
                    page.goto(url)
                    content = page.content()
                    
                    import re
                    json_match = re.search(r'({.*})', content)
                    if json_match:
                        data = json.loads(json_match.group(1))
                    else:
                        data_text = page.inner_text('pre') or page.inner_text('body')
                        data = json.loads(data_text)
                    
                    browser.close()
                
                if data and 't' in data and len(data['t']) > 0:
                    # Create DataFrame for this chunk
                    chunk_df = pd.DataFrame({
                        'timestamp': pd.to_datetime(data['t'], unit='s'),
                        'close': data['c'],
                    })
                    
                    chunk_df = chunk_df.set_index('timestamp')
                    chunk_df['us_dollar_index'] = chunk_df['close']
                    dfs.append(chunk_df)
                    
                    logging.info(f"Successfully fetched chunk from {current_start.date()} to {chunk_end.date()}")
                else:
                    logging.warning(f"No data returned for chunk {current_start.date()} to {chunk_end.date()}")
                
                # Move to next chunk
                current_start = chunk_end
                
                # Add small delay between requests
                time.sleep(1)
            
            if not dfs:
                logging.warning("No US Dollar Index data returned from any chunk")
                return None
                
            # Combine all chunks
            df = pd.concat(dfs)
            
            # Remove duplicates that may exist at chunk boundaries
            df = df[~df.index.duplicated(keep='first')]
            
            # Resample to the specified timeframe
            df = self.resample_data(df, timeframe, start_date)
            # Check if data is available up to the end time
            current_date = pd.Timestamp(end_data_str)
            last_date = df.index.max()

            if last_date < current_date:
                logging.info(f"Missing US Dollar Index data from {last_date.date()} to {current_date.date()}. Forward filling last values.")

                # Generate future dates based on timeframe
                freq = self.timeframe_map.get(timeframe.lower(), 'D')
                future_dates = pd.date_range(start=last_date , end=current_date, freq=freq)
                # Remove first date since it would duplicate the last existing date
                future_dates = future_dates[1:]
                # Get last value
                last_value = df['us_dollar_index'].iloc[-1]

                # Create DataFrame with last value repeated
                future_df = pd.DataFrame({
                    'us_dollar_index': [last_value] * len(future_dates)
                }, index=future_dates)

                # Append forward filled data
                df = pd.concat([df, future_df])

            # Final forward and backward fill for any remaining gaps
            df = df.ffill().bfill()
            logging.info(f"Successfully fetched complete US Dollar Index data with {len(df)} records")
            return df[['us_dollar_index']]
            
        except Exception as e:
            logging.error(f"Error fetching US Dollar Index data: {e}")
            return None
    def fetch_employment_data(self, start_date: str, end_data_str: str , timeframe: str) -> Optional[pd.DataFrame]:
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
            df = self.resample_data(df, timeframe, start_date)
            # Drop data before start_date
            df = df[df.index >= pd.Timestamp(start_date)]

            # Check if data is available up to the current date
            current_date = pd.Timestamp(end_data_str)
            last_date = df.index.max()

            if last_date < current_date:
                logging.info(f"Missing data from {last_date.date()} to {current_date.date()}. Forward filling last values.")
                
                freq = self.timeframe_map.get(timeframe.lower(), 'D')
                # Generate future dates
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=current_date, freq=freq)
                
                # Get last values
                last_values = df.iloc[-1]
                
                # Create DataFrame with future dates filled with last values
                future_df = pd.DataFrame(index=future_dates, data={col: last_values[col] for col in df.columns})
                
                # Concatenate with original data
                df = pd.concat([df, future_df])
            
            df.index = pd.to_datetime(df.index)
            df.index.name = 'timestamp'
            return df
            
        except Exception as e:
            logging.error(f"Error fetching employment data: {e}")
            return None

    def get_bitcoin_dominance(self, start_date: str, end_data_str: str , global_market_cap: pd.DataFrame, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch Bitcoin dominance data from CoinCodex API
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            timeframe (str): Timeframe of the data (e.g., '1d', '4h', etc.)
            global_market_cap (pd.DataFrame): DataFrame containing global market cap data
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with Bitcoin dominance data or None if request fails
        """
        import pandas as pd
        import requests
        import time

        try:
            end_date = end_data_str
            # Define headers for the API request
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                "Accept": "application/json",
                "Referer": "https://coincheckup.com/",
                "X-Requested-With": "XMLHttpRequest",
                "Accept-Encoding": "br"
            }
            
            # Parse start_date to get year
            start_year = pd.to_datetime(start_date).year
            current_year = pd.Timestamp.now().year
            
            # Initialize empty DataFrame to store all data
            all_btc_data = pd.DataFrame()
            
            # Fetch data in 3-month intervals
            start_date_dt = pd.to_datetime(start_date)
            end_date_dt = pd.to_datetime(end_data_str)
            
            # Generate 3-month intervals
            current_start = start_date_dt
            while current_start <= end_date_dt:
                # Calculate end of current 3-month period
                current_end = min(
                    current_start + pd.DateOffset(months=3),
                    end_date_dt + pd.DateOffset(days = 1)
                )
                
                # Format dates for API
                period_start = current_start.strftime('%Y-%m-%d')
                period_end = current_end.strftime('%Y-%m-%d')
                
                # API endpoint for Bitcoin historical data for 3-month period
                url = f"https://coincheckup.com/api/v1/coins/get_coin_history/BTC/{period_start}/{period_end}/1000000"
                
                logging.info(f"Fetching Bitcoin data for period {period_start} to {period_end}")
                
                try:
                    # Make the request with timeout and retry logic
                    for attempt in range(3):  # Try up to 3 times
                        try:
                            response = requests.get(url, proxies=self.proxy, headers=headers, timeout=30)
                            response.raise_for_status()
                            break
                        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                            if attempt == 2:  # Last attempt
                                logging.warning(f"Failed to fetch Bitcoin data for period {period_start} to {period_end}: {e}")
                                raise
                            logging.warning(f"Attempt {attempt + 1} failed, retrying in 2 seconds...")
                            time.sleep(10)  # Wait before retrying
                    
                    # Parse the JSON response
                    data = response.json()
                    
                    # Extract the Bitcoin data
                    btc_data = data.get('BTC', [])
                    
                    if btc_data:
                        # Create DataFrame from the data
                        # Format: [timestamp, price, volume, market_cap]
                        period_df = pd.DataFrame(btc_data, columns=['timestamp', 'price', 'volume', 'btc_market_cap'])
                        
                        # Convert timestamp to datetime
                        period_df['timestamp'] = pd.to_datetime(period_df['timestamp'], unit='s')
                        period_df.set_index('timestamp', inplace=True)
                        
                        # Append to the main DataFrame
                        all_btc_data = pd.concat([all_btc_data, period_df])
                        logging.info(f"Successfully processed data for period {period_start} to {period_end}")
                
                except Exception as e:
                    logging.error(f"Error processing Bitcoin data for period {period_start} to {period_end}: {e}")
                    # Continue with the next period instead of failing completely
                
                # Move to next 3-month period
                current_start = current_end + pd.DateOffset(days=1)
                
            # If we haven't reached end_date_dt, make one final request for remaining period
            if current_start <= end_date_dt:
                period_start = current_start.strftime('%Y-%m-%d')
                period_end = end_date_dt.strftime('%Y-%m-%d')
                
                url = f"https://coincheckup.com/api/v1/coins/get_coin_history/BTC/{period_start}/{period_end}/1000000"
                
                try:
                    response = requests.get(url, proxies=self.proxy, headers=headers, timeout=30)
                    response.raise_for_status()
                    data = response.json()
                    btc_data = data.get('BTC', [])
                    
                    if btc_data:
                        period_df = pd.DataFrame(btc_data, columns=['timestamp', 'price', 'volume', 'btc_market_cap'])
                        period_df['timestamp'] = pd.to_datetime(period_df['timestamp'], unit='s')
                        period_df.set_index('timestamp', inplace=True)
                        all_btc_data = pd.concat([all_btc_data, period_df])
                        
                except Exception as e:
                    logging.error(f"Error processing final Bitcoin data period: {e}")
            
            # Sort by timestamp
            if not all_btc_data.empty:
                # Sort by timestamp and remove duplicate indices
                all_btc_data = all_btc_data.sort_index()
                all_btc_data = all_btc_data[~all_btc_data.index.duplicated(keep='first')]
                all_btc_data = self.resample_data(all_btc_data, timeframe, start_date)
                

                
                if global_market_cap is not None:
                    # Merge the dataframes
                    merged_df = pd.merge(all_btc_data, global_market_cap, left_index=True, right_index=True, how='inner')
                    
                    # Calculate Bitcoin dominance
                    merged_df['btc_dominance'] = (merged_df['btc_market_cap'] / merged_df['Total_market_cap']) * 100
                    merged_df = merged_df[merged_df.index <= pd.Timestamp(end_data_str)]

                    # Select only the dominance column
                    dominance_df = merged_df[['btc_dominance']].copy()
                    
                    # Ensure timestamp is the index
                    if dominance_df.index.name != 'timestamp':
                        dominance_df.index.name = 'timestamp'
                    
                    return dominance_df
            
            return None
        except Exception as e:
            logging.error(f"Error fetching Bitcoin dominance data: {e}")
            return None

    def get_global_crypto_market_cap(self, start_date: str, end_data_str: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch global cryptocurrency market capitalization data from CoinCodex API
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            timeframe (str): Timeframe of the data (e.g., 'day', '1h', etc.)
            
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
            end_date_dt = pd.to_datetime(end_data_str)
            
            # Initialize empty DataFrame to store all data
            all_market_cap_data = pd.DataFrame()
            
            # Split into 3-month periods
            current_start = start_date_dt
            while current_start < end_date_dt:
                # Calculate period end (3 months from start)
                period_end = min(current_start + pd.DateOffset(months=3), end_date_dt)
                
                # Format dates for API
                period_start = current_start.strftime('%Y-%m-%d')
                period_end_str = period_end.strftime('%Y-%m-%d')
                
                # API endpoint for global crypto market cap
                url = f"https://coincheckup.com/api/coincodex/get_coin_marketcap/SUM_ALL_COINS/{period_start}/{period_end_str}/100000"
                
                try:
                    # Make the request
                    response = requests.get(url, proxies=self.proxy, headers=headers)
                    response.raise_for_status()
                    
                    # Parse the JSON response
                    data = response.json()
                    market_cap_data = data.get('SUM_ALL_COINS', [])
                    
                    if market_cap_data:
                        # Create DataFrame from the period data
                        period_df = pd.DataFrame(market_cap_data, columns=['timestamp', 'Total_market_cap', 'global_volume', 'other'])
                        period_df.drop(columns=['other'], inplace=True)
                        period_df['timestamp'] = pd.to_datetime(period_df['timestamp'], unit='s')
                        period_df.set_index('timestamp', inplace=True)
                        all_market_cap_data = pd.concat([all_market_cap_data, period_df])
                        
                except Exception as e:
                    logging.error(f"Error processing period {period_start} to {period_end_str}: {e}")
                
                # Move to next period
                current_start = period_end

            if not all_market_cap_data.empty:
                # Remove duplicates and sort
                all_market_cap_data = all_market_cap_data.sort_index()
                all_market_cap_data = all_market_cap_data[~all_market_cap_data.index.duplicated(keep='first')]
                
                # Filter data starting from start_date
                all_market_cap_data = all_market_cap_data[all_market_cap_data.index >= start_date_dt]
                
                # Resample data according to timeframe
                df = self.resample_data(all_market_cap_data, timeframe, start_date)
                
                current_date = pd.Timestamp(end_data_str)
                # Drop data after end time
                last_date = df.index.max()
                df = df[df.index <= current_date]

                if last_date < current_date:
                    logging.info(f"Missing market cap data from {last_date.date()} to {current_date.date()}. Forward filling last values.")
                    
                    # Generate future dates based on timeframe
                    freq = self.timeframe_map.get(timeframe.lower(), 'D')
                    future_dates = pd.date_range(start=last_date, end=current_date, freq=freq)
                    future_dates = future_dates[1:]  # Remove first date to avoid duplication
                    
                    # Get last values
                    last_market_cap = df['Total_market_cap'].iloc[-1]
                    last_volume = df['global_volume'].iloc[-1]
                    
                    # Create DataFrame with last values repeated
                    future_df = pd.DataFrame({
                        'Total_market_cap': [last_market_cap] * len(future_dates),
                        'global_volume': [last_volume] * len(future_dates)
                    }, index=future_dates)
                    
                    # Append forward filled data
                    df = pd.concat([df, future_df])

                df.index = pd.to_datetime(df.index)
                df.index.name = 'timestamp'
                return df
            
            return None
            
        except Exception as e:
            logging.error(f"Error fetching global crypto market cap data: {e}")
            return None
    
    def fetch_us_gdp_growth(self, start_date: str, end_data_str: str, timeframe: str) -> Optional[pd.DataFrame]:
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
            
            df = self.resample_data(df, timeframe, start_date)
            
            # Drop data before start_date
            df = df[df.index >= pd.Timestamp(start_date)]

            # Check if data is available up to the current date
            current_date = pd.Timestamp(end_data_str)
            last_date = df.index.max()

            if last_date < current_date:
                logging.info(f"Missing data from {last_date.date()} to {current_date.date()}. Forward filling last values.")
                
                # Generate future dates
                freq = self.timeframe_map.get(timeframe.lower(), 'D')
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=current_date, freq=freq)
                
                # Get last value
                last_gdp = df['gdp_growth_rate'].iloc[-1]
                
                # Create DataFrame with future dates filled with last value
                future_df = pd.DataFrame(index=future_dates, data={'gdp_growth_rate': last_gdp})
                
                # Concatenate with original data
                df = pd.concat([df, future_df])

            df.index = pd.to_datetime(df.index)
            df.index.name = 'timestamp'
            return df
            
        except Exception as e:
            logging.error(f"Error fetching GDP growth rate data: {e}")
            return None
    
    def get_fomc_meetings(self, start_date: str, end_data_str: str, timeframe: str) -> Optional[pd.DataFrame]:
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
            # Check for duplicate timestamps before setting index
            if result['timestamp'].duplicated().any():
                # Drop duplicates or keep the first occurrence
                result = result.drop_duplicates(subset=['timestamp'], keep='first')
                logging.warning("Dropped duplicate FOMC meeting dates")
            
            # Now set the index safely
            result.set_index('timestamp', inplace=True)
            result = self.resample_data(result, timeframe, start_date, ismeeting=True)
            # Add zeros for dates from start_date to the first meeting date
            if not result.empty:
                # Convert start_date to timestamp
                start_timestamp = pd.Timestamp(start_date)
                
                # Get the first meeting date in the result
                first_meeting_date = result.index.min()
                
                # Only add zeros if the first meeting date is after the start date
                if first_meeting_date > start_timestamp:
                    # Create a date range from start_date to the day before first meeting
                    date_range = pd.date_range(start=start_timestamp, end=first_meeting_date - pd.Timedelta(days=1))
                    
                    # Create a DataFrame with zeros for all columns in the date range
                    zeros_df = pd.DataFrame(0, index=date_range, columns=result.columns)
                    
                    # Concatenate with the original result
                    result = pd.concat([zeros_df, result])
                    
                    logging.info(f"Added {len(date_range)} days with zero values before first FOMC meeting")


            # Ensure the index is named 'timestamp'
            result.index.name = 'timestamp'
            
            # Sort the index to ensure chronological order
            result.sort_index(inplace=True)
            
            logging.info(f"Processed FOMC meeting data with {len(result)} records")
            result = result[result.index <= pd.Timestamp(end_data_str)]
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

    def get_interest_rates(self, start_date: str, end_data_str: str, timeframe: str) -> Optional[pd.DataFrame]:
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

            interest_rates = self.resample_data(interest_rates, timeframe, start_date)
            # Check if data is available up to the current date
            current_date = pd.Timestamp(end_data_str)
            last_date = interest_rates.index.max()
            interest_rates = interest_rates[interest_rates.index <= current_date]

            if last_date < current_date:
                logging.info(f"Missing data from {last_date.date()} to {current_date.date()}. Forward filling last values.")
                
                # Generate future dates
                freq = self.timeframe_map.get(timeframe.lower(), 'D')
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=current_date, freq=freq)
                
                # Get last value
                last_rate = interest_rates['interest_rate'].iloc[-1]
                
                # Create DataFrame with future dates filled with last value
                future_df = pd.DataFrame(index=future_dates, data={'interest_rate': last_rate})
                
                # Concatenate with original data
                interest_rates = pd.concat([interest_rates, future_df])
            
            interest_rates.index = pd.to_datetime(interest_rates.index)
            interest_rates.index.name = 'timestamp'
            return interest_rates
        except Exception as e:
            logging.error(f"Error fetching interest rate data: {e}")
            return None



    def get_inflation_rate(self, start_date: str, end_data_str: str , timeframe: str) -> Optional[pd.DataFrame]:
        """
        Fetch US inflation rate data from FRED using CPI data and handle missing future values with ARIMA forecasting.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            timeframe (str): The timeframe for resampling data (e.g., 'D' for daily)
        
        Returns:
            Optional[pd.DataFrame]: DataFrame with inflation rate data or None if request fails
        """
        try:
            # Get CPI data from FRED
            cpi = self.fred.get_series('CPIAUCSL', start_date, end_date= end_data_str)
            
            # Convert to DataFrame
            inflation_df = pd.DataFrame(cpi, columns=['CPI'])
            inflation_df.index.name = 'timestamp'

            # Calculate Inflation Rate (MoM % Change)
            inflation_df['inflation_rate_mom'] = inflation_df['CPI'].pct_change() * 100

            # Calculate Inflation Rate (YoY % Change)
            inflation_df['inflation_rate_yoy'] = inflation_df['CPI'].pct_change(12) * 100

            # Resample to daily frequency and forward fill
            inflation_df = self.resample_data(inflation_df, timeframe, start_date)
            inflation_df = inflation_df.ffill().bfill()  # Fill missing values
            # Drop data before start_date
            inflation_df = inflation_df[inflation_df.index >= pd.Timestamp(start_date)]

            # Check if data is available up to the current date
            current_date = pd.Timestamp(end_data_str)
            last_date = inflation_df.index.max()

            if last_date < current_date:
                logging.info(f"Missing data from {last_date.date()} to {current_date.date()}. Forward filling last values.")

                # Generate future dates
                freq = self.timeframe_map.get(timeframe.lower(), 'D')
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), end=current_date, freq=freq)

                # Get last values
                last_cpi = inflation_df['CPI'].iloc[-1]
                last_mom = inflation_df['inflation_rate_mom'].iloc[-1] 
                last_yoy = inflation_df['inflation_rate_yoy'].iloc[-1]

                # Create DataFrame with last values repeated
                future_df = pd.DataFrame({
                    'CPI': [last_cpi] * len(future_dates),
                    'inflation_rate_mom': [last_mom] * len(future_dates),
                    'inflation_rate_yoy': [last_yoy] * len(future_dates)
                }, index=future_dates)

                # Append forward filled data
                inflation_df = pd.concat([inflation_df, future_df])

            # Final fill for any remaining missing values
            inflation_df = inflation_df.ffill().bfill()

            # Log success message
            logging.info("Successfully retrieved and filled inflation data.")
            inflation_df.index = pd.to_datetime(inflation_df.index)
            inflation_df.index.name = 'timestamp'

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
                
                # Read the data file to get first timestamp
                try:
                    df = pd.read_pickle(file_path)
                    start_date_str = df.index.min()
                    end_data_str = df.index.max()
                    logging.info(f"Using start date from data: {start_date_str}")
                except Exception as e:
                    logging.error(f"Error reading file {file_path}: {e}")
                    continue

                # Fetch all macro features
                crypto_market_cap = self.get_global_crypto_market_cap(start_date_str, end_data_str, timeframe)
                bitcoin_dominance = self.get_bitcoin_dominance(start_date_str, end_data_str, crypto_market_cap, timeframe)
                gdp_growth = self.fetch_us_gdp_growth(start_date_str, end_data_str, timeframe)
                inflation_rate = self.get_inflation_rate(start_date_str, end_data_str, timeframe)
                interest_rates = self.get_interest_rates(start_date_str, end_data_str, timeframe)
                employment_data = self.fetch_employment_data(start_date_str, end_data_str, timeframe)
                us_dollar_index = self.get_us_dollar_index(start_date_str, end_data_str, timeframe=timeframe)
                meetings = self.get_fomc_meetings(start_date_str, end_data_str, timeframe)

                # Initialize with first dataframe
                df = interest_rates
                
                # Merge all dataframes
                dataframes = [us_dollar_index, inflation_rate, employment_data, gdp_growth, meetings, crypto_market_cap, bitcoin_dominance]
                for data in dataframes:
                    if data is not None:
                        df = pd.merge(df, data, left_index=True, right_index=True, how='left')

                # Fill missing values with zero for FOMC meeting related columns
                meeting_columns = ['fed_meeting', 'post_meeting', 'pre_meeting']
                meeting_columns = [col for col in meeting_columns if col in df.columns]
                if meeting_columns:
                    df[meeting_columns] = df[meeting_columns].fillna(0)
                
                if timeframe == '5m':
                    # Drop inflation_rate_yoy column for 5m timeframe
                    if 'inflation_rate_yoy' in df.columns:
                        df.drop('inflation_rate_yoy', axis=1, inplace=True)
                        logging.info("Dropped inflation_rate_yoy column for 5m timeframe")
                # Log the columns that were filled
                logging.info(f"Filled missing values with zeros in FOMC meeting columns")

                if interest_rates is None:
                    logging.warning("No interest rate data available - skipping")
                    continue

                # Create output directory
                output_dir = os.path.join(self.data_dir.split('/')[0], '5_process_macro_features')
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
