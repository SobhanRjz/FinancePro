import os
import logging
import pandas as pd
import requests
import time
from typing import Dict, List, Optional
from datetime import datetime

class OtherFeatureExtractor:
    def __init__(self, data_dir='data/11_other_features'):
        """Initialize feature extractor for commodities and market indices."""
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

        self.proxy = {
            'http': 'http://127.0.0.1:10808',
            'https': 'http://127.0.0.1:10808'
        }

        self.timeframe_map = {
            '1d': 'D',
            '4h': '4H',
            '1h': 'H', 
            '5m': '5min'
        }

        self.headers = {
            'authority': 'p.fxempire.com',
            'method': 'GET',
            'scheme': 'https',
            'accept': '*/*',
            'accept-encoding': 'gzip, deflate, br, zstd',
            'accept-language': 'en-US,en;q=0.9',
            'cache-control': 'no-cache',
            'origin': 'https://www.fxempire.com',
            'pragma': 'no-cache',
            'priority': 'u=1, i',
            'referer': 'https://www.fxempire.com/',
            'sec-ch-ua': '"Chromium";v="134", "Not:A-Brand";v="24", "Microsoft Edge";v="134"',
            'sec-ch-ua-mobile': '?0',
            'sec-ch-ua-platform': '"Windows"',
            'sec-fetch-dest': 'empty',
            'sec-fetch-mode': 'cors',
            'sec-fetch-site': 'same-site',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/134.0.0.0 Safari/537.36 Edg/134.0.0.0'
        }

        self.granularity_map = {
            "4h": "H4",
            "1h": "H1",
            "5m": "M1", 
            "D": "D"
        }
        self.period_precision_map_Nasdaq = {
                "1d": {"period": "1", "precision": "day"},
                "4h": {"period": "4", "precision": "hour"},
                "1h": {"period": "1", "precision": "hour"},
                "5m": {"period": "1", "precision": "minute"}
        }
        self.period_precision_map_SP500 = {
                "1d": {"period": "24", "precision": "Hours"},
                "4h": {"period": "4", "precision": "Hours"},
                "1h": {"period": "1", "precision": "Hours"},
                "5m": {"period": "1", "precision": "Minutes"}
            }
            
            

        self.chunk_sizes = {
            "1h": 17,    # 17 days of hourly data
            "4h": 70,    # 70 days of 4-hour data
            "5m": 1,     # 1 day of 5-minute data
            "D": 420     # 420 days of daily data
        }

    def _make_api_request(self, url: str, chunk_start: str, chunk_end: str) -> Optional[Dict]:
        """Make API request with retries."""
        instrument = url.split('/')[-2]  # Extract instrument name from URL
        max_retries = 5
        for retry in range(max_retries):
            try:
                response = requests.get(url, headers=self.headers, proxies=self.proxy)
                if response.status_code == 200:
                    logging.info(f"Successfully fetched {instrument} data for period {chunk_start} to {chunk_end}")
                    return response.json()
                
                logging.error(f"Failed to fetch {instrument} data: {response.status_code}, attempt {retry + 1} of {max_retries}")
            except requests.exceptions.RequestException as e:
                logging.error(f"Request error for {instrument}: {str(e)}, attempt {retry + 1} of {max_retries}")
            except ValueError as e:
                logging.error(f"JSON decode error for {instrument}: {str(e)}, attempt {retry + 1} of {max_retries}")
            except Exception as e:
                logging.error(f"Unexpected error for {instrument}: {str(e)}, attempt {retry + 1} of {max_retries}")
            
            time.sleep(1)
            
        logging.error(f"Failed to fetch {instrument} data after {max_retries} retries")
        return None

    def resample_data(self, df: pd.DataFrame, timeframe: str, start_time: str) -> pd.DataFrame:
        """Resample data to specified timeframe."""
        freq = self.timeframe_map.get(timeframe.lower(), 'D')
        df = df.bfill().ffill()
        resampled_df = df.resample(freq, origin=df.index[0]).asfreq()
        resampled_df = resampled_df.ffill().bfill()
            
        start_timestamp = pd.to_datetime(start_time)
        resampled_df = resampled_df[resampled_df.index >= (start_timestamp - pd.Timedelta(minutes=resampled_df.index.freq.n))]
        
        if len(resampled_df) > 0:
            time_diff = resampled_df.index[0] - start_timestamp
            resampled_df.index = resampled_df.index - pd.Timedelta(minutes=time_diff.total_seconds()/60)

        return resampled_df

    def _process_data(self, data_list: List[Dict], start_date: str, end_date: str, 
                     timeframe: str, price_column: str) -> Optional[pd.DataFrame]:
        """Process collected data into DataFrame."""
        if not data_list:
            logging.error(f"No {price_column} data collected")
            return None

        data = pd.DataFrame(data_list)
        data.set_index('timestamp', inplace=True)
        data.sort_index(inplace=True)
        
        # Remove duplicate indices before processing
        data = data[~data.index.duplicated(keep='first')]
        
        data.index = pd.to_datetime(data.index).tz_localize(None)

        start_dt = pd.to_datetime(start_date).tz_localize(None)
        end_dt = pd.to_datetime(end_date).tz_localize(None)

        data = data[(data.index >= start_dt) & (data.index <= end_dt)]
        data = self.resample_data(data, timeframe, start_date)

        if data.index.max() < pd.to_datetime(end_date):
            last_row = data.iloc[-1]
            additional_dates = pd.date_range(
                start=data.index.max() + data.index.freq,
                end=pd.to_datetime(end_date),
                freq=data.index.freq
            )
            additional_df = pd.DataFrame(index=additional_dates, columns=data.columns)
            additional_df.loc[:] = last_row.values
            data = pd.concat([data, additional_df[~additional_df.index.isin(data.index)]])
            data.sort_index(inplace=True)

        data = data.ffill().bfill()
        data.index.name = 'timestamp'
        return data

    def _fetch_market_data(self, instrument: str, timeframe: str, start_date: str, end_date: str,
                          price_column: str, url_template: str, period_precision_map: dict) -> Optional[pd.DataFrame]:
        """Generic method to fetch market data."""
        if end_date is None:
            end_date = pd.Timestamp.now().strftime('%Y-%m-%d')

        try:
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            data_list = []
            current_start = start_dt

            while current_start <= end_dt:
                days_to_add = self.chunk_sizes.get(timeframe, self.chunk_sizes["D"])
                chunk_end = min(current_start + pd.DateOffset(days=days_to_add), end_dt)
                

                chunk_start_str = current_start.strftime('%Y-%m-%d')
                chunk_end_str = chunk_end.strftime('%Y-%m-%d')

                # Check if period is in url_template before formatting
                if "{period}" in url_template:
                    if timeframe == '5m' and instrument == 'SPX.IND_CBOM':
                        chunk_start_str = current_start.strftime('%Y-%m-%d %H:%M').replace("-", "/").replace(' ', '%20')
                        chunk_end_str = chunk_end.strftime('%Y-%m-%d %H:%M').replace("-", "/").replace(' ', '%20')

                    url = url_template.format(
                        instrument=instrument,
                        granularity=period_precision_map.get(timeframe, "D")["period"],
                        period=period_precision_map.get(timeframe, "D")["precision"],
                        start=chunk_start_str,
                        end=chunk_end_str
                    )
                else:
                    if timeframe == '5m' and instrument == 'XAG/USD':
                        chunk_start_str = str(int(current_start.timestamp()))
                        chunk_end_str = str(int(chunk_end.timestamp()))

                    url = url_template.format(
                        instrument=instrument,
                        granularity=period_precision_map.get(timeframe, "D"),
                        start=chunk_start_str,
                        end=chunk_end_str
                    )
                
                json_data = self._make_api_request(url, chunk_start_str, chunk_end_str)
                
                # Process data based on instrument type
                if instrument in ['NDAQ', 'NVDA', 'AMD']:
                    if isinstance(json_data, list):
                        for bar in json_data:
                            date_str = f"{bar['StartDate']} {bar['StartTime']}"
                            time = pd.to_datetime(date_str)
                            if time is not None:
                                close_price = float(bar['Close'])
                                if time.time() != pd.to_datetime(end_date).time() and timeframe == '1d':
                                    time = time.replace(hour=0, minute=0, second=0, microsecond=0)
                                data_list.append({
                                    'timestamp': time,
                                    price_column: close_price
                                })
                elif instrument == 'SPX.IND_CBOM':
                    if isinstance(json_data, dict) and 'ChartBars' in json_data:
                        for bar in json_data['ChartBars']:
                            date_str = f"{bar['StartDate']} {bar['StartTime']}"
                            time = pd.to_datetime(date_str)
                            if time is not None:
                                close_price = float(bar['Close'])
                                data_list.append({
                                    'timestamp': time,
                                    price_column: close_price
                                })
                elif instrument in ['XAU/USD', 'XAG/USD']:
                    if isinstance(json_data, dict) and 'candles' in json_data:
                        for item in json_data['candles']:
                            time_str = item.get('time')
                            if time_str and 'mid' in item:
                                date = pd.to_datetime(time_str)
                                close = float(item['mid'].get('c'))
                                if date and close is not None:
                                    if date.time() != pd.to_datetime(end_date).time() and timeframe == '1d':
                                        date = date.replace(hour=0, minute=0, second=0, microsecond=0)
                                    data_list.append({
                                        'timestamp': date,
                                        price_column: close
                                    })
                elif instrument == 'BCO/USD':
                    if isinstance(json_data, dict) and 'candles' in json_data:
                        for candle in json_data['candles']:
                            time = pd.to_datetime(candle.get('time'))
                            if 'mid' in candle and time is not None:
                                close_price = float(candle['mid'].get('c'))
                                if time.time() !=  pd.to_datetime(end_date).time()  and timeframe == '1d':
                                    time = time.replace(hour=0, minute=0, second=0, microsecond=0)
                                data_list.append({
                                    'timestamp': time,
                                    price_column: close_price
                                })
                
                current_start = chunk_end + pd.DateOffset(days=1)

            return self._process_data(data_list, start_date, end_date, timeframe, price_column)

        except Exception as e:
            logging.error(f"Error fetching {instrument} data: {e}")
            return None

    def fetch_Gold_data(self, timeframe: str, start_date: str = '2010-01-01', end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch gold price data."""
        return self._fetch_market_data(
            instrument='XAU/USD',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            price_column='gold_price',
            url_template="https://p.fxempire.com/oanda/candles/latest?instrument={instrument}&granularity={granularity}&from={start}&to={end}",
            period_precision_map=self.granularity_map
        )

    def fetch_Silver_data(self, timeframe: str, start_date: str = '2010-01-01', end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch silver price data."""
        return self._fetch_market_data(
            instrument='XAG/USD',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            price_column='silver_price',
            url_template="https://p.fxempire.com/oanda/candles/latest?instrument={instrument}&granularity={granularity}&from={start}&to={end}",
            period_precision_map=self.granularity_map
        )

    def fetch_Oil_data(self, timeframe: str, start_date: str = '2010-01-01', end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch oil price data."""
        return self._fetch_market_data(
            instrument='BCO/USD',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            price_column='oil_price',
            url_template="https://p.fxempire.com/oanda/candles/latest?instrument={instrument}&granularity={granularity}&from={start}&to={end}",
            period_precision_map=self.granularity_map
        )

    def fetch_SP500_data(self, timeframe: str, start_date: str = '2010-01-01', end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch S&P 500 data."""
        return self._fetch_market_data(
            instrument='SPX.IND_CBOM',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            price_column='S&P 500',
            url_template="https://p.fxempire.com/globalindices/xglobalindices.json/GetChartBars?IdentifierType=Symbol&period={granularity}&Precision={period}&StartTime={start}&EndTime={end}&Identifier={instrument}&AdjustmentMethod=PriceReturn&PriceType=Mid&IncludeExtended=true",
            period_precision_map=self.period_precision_map_SP500
        )
    
    def fetch_Nvidia_data(self, timeframe: str, start_date: str = '2010-01-01', end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch Nvidia data."""
        return self._fetch_market_data(
            instrument='NVDA',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            price_column='Nvidia GPU',
            url_template="https://www.fxempire.com/api/v1/en/charts/lightchart/candles/{instrument}/range/{granularity}/{period}/{start}/{end}?include-extended=false",
            period_precision_map=self.period_precision_map_Nasdaq
        )
    def fetch_AMD_data(self, timeframe: str, start_date: str = '2010-01-01', end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch AMD data."""
        return self._fetch_market_data(
            instrument='AMD',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            price_column='AMD GPU',
            url_template="https://www.fxempire.com/api/v1/en/charts/lightchart/candles/{instrument}/range/{granularity}/{period}/{start}/{end}?include-extended=false",
            period_precision_map=self.period_precision_map_Nasdaq
        )

    def fetch_Nasdaq_data(self, timeframe: str, start_date: str = '2010-01-01', end_date: str = None) -> Optional[pd.DataFrame]:
        """Fetch Nasdaq data."""
        return self._fetch_market_data(
            instrument='NDAQ',
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            price_column='Nasdaq',
            url_template="https://www.fxempire.com/api/v1/en/charts/lightchart/candles/{instrument}/range/{granularity}/{period}/{start}/{end}?include-extended=false",
            period_precision_map=self.period_precision_map_Nasdaq
        )

    def process_all_features(self):
        """Process and save all market features."""
        import glob
        import re

        file_pattern = os.path.join('data', 'OHLCV', '*.pkl.gz')
        file_paths = glob.glob(file_pattern)
        
        if not file_paths:
            logging.warning("No OHLCV files found to determine time range")
            return

        List_of_files = ['data/11_other_features/Gold_Oil_Features_gold_BTCUSDT_5m_1y.pkl.gz',
                         'data/11_other_features/Gold_Oil_Features_nasdaq_BTCUSDT_5m_1y.pkl.gz',
                         'data/11_other_features/Gold_Oil_Features_nvidia_BTCUSDT_5m_1y.pkl.gz',
                         'data/11_other_features/Gold_Oil_Features_oil_BTCUSDT_5m_1y.pkl.gz',
                         'data/11_other_features/Gold_Oil_Features_sp500_BTCUSDT_5m_1y.pkl.gz',
                         'data/11_other_features/Gold_Oil_Features_amd_BTCUSDT_5m_1y.pkl.gz',
                         ]
        for file_path in file_paths[3:]:
            file_name = os.path.basename(file_path)
            match = re.search(r'([A-Z]+)_(\w+)_(\w+)', file_name)
            
            if not match:
                continue
                
            symbol, timeframe, period = match.groups()
            logging.info(f"Processing {symbol} data for {timeframe} timeframe over {period}")

            df = pd.read_pickle(file_path)
            start_date = df.index.min()
            end_date = df.index.max()

            # Fetch all market data
            # Fetch individual market data
            #amd_data = self.fetch_AMD_data(timeframe, start_date, end_date)
            #sp500_data = self.fetch_SP500_data(timeframe, start_date, end_date)
            #nvidia_data = self.fetch_Nvidia_data(timeframe, start_date, end_date)
            #nasdaq_data = self.fetch_Nasdaq_data(timeframe, start_date, end_date)
            #oil_data = self.fetch_Oil_data(timeframe, start_date, end_date)
            #gold_data = self.fetch_Gold_data(timeframe, start_date, end_date)
            #silver_data = self.fetch_Silver_data(timeframe, start_date, end_date)
            market_data = []
            for file in List_of_files:
                df = pd.read_pickle(file)
                start_date = df.index.min()
                end_date = df.index.max()

                # Fetch all market data
                # Fetch individual market data
                # Combine into list and filter out None values
                market_data.append(df)
                
            valid_data = [data for data in market_data if data is not None]
            
            if not valid_data:
                logging.error("No data available to merge")
                continue

            merged_data = valid_data[0]
            for data in valid_data[1:]:
                merged_data = merged_data.join(data, how='outer')

            merged_data = merged_data.ffill().bfill()

            # Save processed data
            output_dir = os.path.join(self.data_dir, '11_process_other_features')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f'Gold_Oil_Features_{file_name}')
            merged_data.to_pickle(output_file)
            logging.info(f"Saved processed features to {output_file}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    feature_extractor = OtherFeatureExtractor()
    feature_extractor.process_all_features()
