import requests
import pandas as pd
import logging
from typing import Optional
import os
import glob
import re
import time
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OnChainFeatureExtractor:
    """Class for extracting on-chain metrics from Santiment API"""
    
    def __init__(self, api_key: str = 's3thksqg5r2n5p7p_qf62ve274f4q7u2r', data_dir: str = 'data/OHLCV'):
        """
        Initialize OnChainFeatureExtractor
        
        Args:
            api_key (str): Santiment API key for authentication
            data_dir (str): Directory containing OHLCV data files
        """
        self.duneApiKey = '1Oq4Pq2gNV9aRIskDiaqggKsEe7x5p61'
        self.api_key = api_key
        self.base_url = "https://api.santiment.net/graphql"
        self.proxy = {
            "http": "http://127.0.0.1:10808",
            "https": "http://127.0.0.1:10808"
        }
        self.data_dir = data_dir
                # Constants
        self.BASE_URL = "https://api.blockchair.com/bitcoin/transactions"
        self.USER_AGENT = {
            "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
            "accept-encoding": "gzip, deflate, br, zstd",
            "accept-language": "en-US,en;q=0.9", 
            "cache-control": "no-cache",
            "connection": "keep-alive",
            "host": "api.blockchair.com",
            "pragma": "no-cache",
            "sec-ch-ua": '"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
            "sec-fetch-dest": "document",
            "sec-fetch-mode": "navigate", 
            "sec-fetch-site": "none",
            "sec-fetch-user": "?1",
            "upgrade-insecure-requests": "1",
            "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0"
        }

        # Whale threshold - adjust as needed
        self.BTC_THRESHOLD = 500  # Transactions moving >= 500 BTC
        self.USD_THRESHOLD = 1_000_000  # Transactions worth >= $1 million USD

    def fetch_long_term_holder_supply(self, start_date: str) -> Optional[pd.DataFrame]:
        """
        Fetch Long Term Holder supply data from Bitcoin Magazine Pro
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with LTH supply data or None if request fails
        """
        try:
            url = "https://www.bitcoinmagazinepro.com/django_plotly_dash/app/lth_supply/_dash-update-component"
            
            headers = {
                "accept": "application/json",
                "accept-encoding": "gzip, deflate, br, zstd", 
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "no-cache",
                "content-type": "application/json",
                "origin": "https://www.bitcoinmagazinepro.com",
                "pragma": "no-cache",
                "referer": "https://www.bitcoinmagazinepro.com/charts/long-term-holder-supply/",
                "sec-ch-ua": '"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"',
                "sec-ch-ua-mobile": "?0",
                "sec-ch-ua-platform": '"Windows"',
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0"
            }
            
            payload = {
                "output": "chart.figure",
                "outputs": {
                    "id": "chart",
                    "property": "figure"
                },
                "inputs": [
                    {
                        "id": "url",
                        "property": "pathname", 
                        "value": "/charts/long-term-holder-supply/"
                    },
                    {
                        "id": "period",
                        "property": "value",
                        "value": "all"
                    },
                    {
                        "id": "display",
                        "property": "children",
                        "value": "xxl 1912px"
                    }
                ],
                "changedPropIds": ["url.pathname", "display.children"]
            }

            proxy = {
                "http": "http://127.0.0.1:10808",
                "https": "http://127.0.0.1:10808"
            }
            response = requests.post(url, headers=headers, json=payload, proxies=proxy)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract x and y data from response
            x_data = [x for x in data["response"]["chart"]["figure"]["data"][1]["x"] if pd.Timestamp(x) < pd.Timestamp.now() - pd.Timedelta(days=2)]
            y_data = data["response"]["chart"]["figure"]["data"][1]["y"]
            
            # Create DataFrame with long term holder supply data
            df = pd.DataFrame({
                'timestamp': pd.to_datetime(x_data, format='ISO8601'),
                'long_term_holder_supply': y_data
            })
            df = df.set_index('timestamp')
            # Filter by start date if provided
            if start_date:
                df = df[df.index >= pd.to_datetime(start_date)]
            logging.info(f"Successfully fetched {len(df)} LTH supply records")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching LTH supply data: {e}")
            return None
        
    def fetch_exchange_reserve(self, start_date):
        """
        Fetch Bitcoin exchange reserve data from CryptoQuant API
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with exchange reserve data
        """
        try:
            # Convert start_date to timestamp in milliseconds
            start_ts = int(pd.Timestamp(start_date).timestamp() * 1000)
            end_ts = int(pd.Timestamp.now().timestamp() * 1000)
            
            url = f"https://api.cryptoquant.com/live/v3/charts/61a5fb0c45de34521f1dcaad?window=DAY&from={start_ts}&to={end_ts}&limit=70000"
            
            headers = {
                "accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7",
                "accept-encoding": "gzip, deflate, br, zstd",
                "accept-language": "en-US,en;q=0.9",
                "cache-control": "no-cache",
                "pragma": "no-cache",
                "sec-ch-ua": '"Not(A:Brand";v="99", "Microsoft Edge";v="133", "Chromium";v="133"',
                "sec-ch-ua-mobile": "?0", 
                "sec-ch-ua-platform": '"Windows"',
                "sec-fetch-dest": "document",
                "sec-fetch-mode": "navigate",
                "sec-fetch-site": "none",
                "sec-fetch-user": "?1",
                "upgrade-insecure-requests": "1",
                "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/133.0.0.0 Safari/537.36 Edg/133.0.0.0"
            }
            response = requests.get(url, headers=headers, proxies=self.proxy)
            response.raise_for_status()
            
            data = response.json()['result']['data']
            # Convert to DataFrame with column names
            df = pd.DataFrame(data, columns=['timestamp', 'exchange_reserve'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df = df.set_index('timestamp')
            
            logging.info(f"Successfully fetched {len(df)} exchange reserve records")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching exchange reserve data: {e}")
            return None
    def fetch_daily_avg_fees(self, start_date, end_date):
        while True:
            try:
                params = {
                    "a": "date,avg(fee_usd)", 
                    "s": "date(desc)",
                    "q": f"time({start_date}..{end_date})"
                }
                response = requests.get(self.BASE_URL, params=params, headers=self.USER_AGENT, proxies=self.proxy)
                
                if response.status_code != 200:
                    logging.error(f"Failed to fetch data: {response.status_code}")
                    time.sleep(10)
                    continue

                data = response.json().get('data', [])
                df = pd.DataFrame(data, columns=['date', 'avg(fee_usd)'])
                df['timestamp'] = pd.to_datetime(df['date'])
                df = df.drop('date', axis=1)
                df.set_index('timestamp', inplace=True)
                return df
            except Exception as e:
                logging.error(f"Error fetching fee data: {e}")
                time.sleep(10)
                continue



    def fetch_transactions_for_date_range(self, start_date, end_date, offset=0):

        while True:
            try:
                """Fetch transactions for a given date range."""
                query = f"time({start_date}..{end_date}),output_total({self.BTC_THRESHOLD * 100000000}..)"
                params = {
                    "q": query,
                    "a": "date,count(),sum(output_total),sum(output_total_usd)"
                }
                
                response = requests.get("https://api.blockchair.com/bitcoin/transactions",
                                     headers=self.USER_AGENT,
                                     params=params,
                                     proxies=self.proxy)
                if response.status_code != 200:
                    logging.error(f"Failed to fetch {start_date}: {response.status_code}")
                    time.sleep(10)
                    continue
                data = response.json()
                break
            except Exception as e:
                logging.error(f"Error fetching outputs for {start_date}: {e}")
                time.sleep(10)
        return data

    def process_transactions(self, data, date):
        """Filter transactions to find whale transactions and return them."""
        whale_transactions = []
        shouldBreak = False

        context = data.get('data', {})
        total_rows = context.get('count()', 0)


        shouldBreak = total_rows == 0
        return total_rows, date

    def fetch_whales_from_date(self, start_date):
        """
        Fetch whale transactions from start_date to now.
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            pd.DataFrame: DataFrame with whale transaction data indexed by timestamp
        """
        from datetime import datetime, timedelta
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        today = datetime.utcnow()

        all_whale_transactions = []

        daily_whale_data = {}
        current = start

        data = self.fetch_transactions_for_date_range(current.date(), today.date(), 1)
        if not data:
            return None 
        
        for row in data["data"]:
            daily_whale_data[row["date"]] = {
                'whale_tx_count': row["count()"],
                'whale_btc_volume': row["sum(output_total)"] / 1e8,
                'whale_usd_volume': row["sum(output_total_usd)"]
            }

        current = current + timedelta(days=1)

        # Convert the daily data dictionary to DataFrame
        df = pd.DataFrame.from_dict(daily_whale_data, orient='index')
        df.index = pd.to_datetime(df.index)
        df.index.name = 'timestamp'
        df = df.sort_index()  # Sort by date

        return df
    def get_active_addresses(self, start_date: str = "2016-02-01") -> Optional[pd.DataFrame]:
        """
        Fetch daily active addresses data from Santiment API
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            
        Returns:
            Optional[pd.DataFrame]: DataFrame with active addresses data indexed by timestamp,
                                  or None if request fails
        """
        query = {
            "query": """
            {
                getMetric(metric: "active_addresses_24h") {
                    timeseriesData(
                        slug: "bitcoin"
                        from: "%sT00:00:00Z"
                        to: "utc_now"
                        interval: "1d"
                    ) {
                        datetime
                        value
                    }
                }
            }
            """ % start_date
        }

        try:
            headers = {"Authorization": f"Apikey {self.api_key}"}
            response = requests.post(
                self.base_url, 
                json=query, 
                headers=headers, 
                proxies=self.proxy
            )
            response.raise_for_status()
            
            data = response.json()
            timeseries_data = data['data']['getMetric']['timeseriesData']
            
            # Convert to DataFrame and format
            df = pd.DataFrame(timeseries_data)
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.rename(columns={
                'datetime': 'timestamp',
                'value': 'active_addresses_24h'
            })
            df = df.set_index('timestamp')
            
            logging.info(f"Successfully fetched {len(df)} active addresses records")
            return df
            
        except Exception as e:
            logging.error(f"Error fetching active addresses data: {e}")
            return None

    def process_all_files(self):
        """
        Process all OHLCV files and save on-chain features
        """
        # Get all OHLCV files
        file_pattern = os.path.join(self.data_dir, "*.pkl.gz")
        file_paths = glob.glob(file_pattern)

        if not file_paths:
            logging.warning(f"No OHLCV files found matching pattern: {file_pattern}")
            return None

        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            
            # Extract timeframe and period from filename
            match = re.search(r'([A-Z]+)_(\w+)_(\w+)', file_name)
            if match:
                symbol, timeframe, period = match.groups()
                logging.info(f"Processing {symbol} data for {timeframe} timeframe over {period}")
                
                # Get on-chain data
                # Get start date from period (e.g. '10y' = 10 years)
                period_value = int(period[:-1])
                period_unit = period[-1].lower()
                
                # Convert period to days using datetime
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

                # Format date string once for reuse
                start_date_str = start_date.strftime('%Y-%m-%d')
                end_date_str = pd.Timestamp.now().strftime('%Y-%m-%d')

                # Fetch all required data
                long_term_holder_supply = self.fetch_long_term_holder_supply(start_date=start_date_str)
                exchange_reserve = self.fetch_exchange_reserve(start_date=start_date_str)
                whale_supply_distribution = self.fetch_whales_from_date(start_date=start_date_str)
                fees = self.fetch_daily_avg_fees(start_date=start_date_str, end_date=end_date_str)
                onchain_data = self.get_active_addresses(start_date=start_date_str)

                # Only proceed with merging if we have the core onchain data
                if onchain_data is None:
                    logging.warning("No onchain data available - skipping merges")
                    return None

                # Convert all dataframes to timezone-naive for consistent merging
                dataframes = [df for df in [long_term_holder_supply, whale_supply_distribution, onchain_data, fees] if df is not None]
                
                # Only append exchange reserve data if it exists and has data within our time range
                if exchange_reserve is not None and not exchange_reserve.empty:
                    earliest_reserve = exchange_reserve.index.min()
                    if (pd.Timestamp.now() - earliest_reserve).days >= period_value * 365:
                        dataframes.append(exchange_reserve)
                for df in dataframes:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)

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
                    
                    logging.info("Successfully merged whale supply and active addresses data")
                if onchain_data is None:
                    continue

                merged_data = merged_data.ffill().bfill()
                # Create output directory
                output_dir = os.path.join(self.data_dir.split('/')[0], 'process_onchain_features')
                os.makedirs(output_dir, exist_ok=True)

                # Create output filename with same pattern
                output_file = os.path.join(output_dir, f'onchain_features_{symbol}_{timeframe}_{period}.pkl.gz')

                try:
                    onchain_data.to_pickle(output_file)
                    logging.info(f"Saved on-chain features to: {output_file}")
                except Exception as e:
                    logging.error(f"Error saving on-chain features: {e}")
            else:
                logging.warning(f"Could not parse pattern from filename: {file_name}")


if __name__ == "__main__":
    # Initialize extractor and process files
    extractor = OnChainFeatureExtractor()
    extractor.process_all_files()
