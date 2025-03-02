import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from models.cryptoModel import Crypto

class CryptoDataFetcher:
    """Class for fetching cryptocurrency price data using CryptoCompare's API"""
    
    def __init__(self):
        self.endpoint_day = "https://min-api.cryptocompare.com/data/v2/histoday"
        self.endpoint_hour = "https://min-api.cryptocompare.com/data/v2/histohour" 
        self.endpoint_minute = "https://min-api.cryptocompare.com/data/v2/histominute"
        
        self.timeframe_mapping = {
            '1m': (1/1440, self.endpoint_minute),  # 1 minute
            '5m': (5/1440, self.endpoint_minute),  # 5 minutes
            '15m': (15/1440, self.endpoint_minute),  # 15 minutes
            '30m': (30/1440, self.endpoint_minute),  # 30 minutes
            '1h': (1/24, self.endpoint_hour),    # 1 hour
            '2h': (2/24, self.endpoint_hour),    # 2 hours
            '4h': (4/24, self.endpoint_hour),    # 4 hours
            '1d': (1, self.endpoint_day),       # 1 day
            '1w': (7, self.endpoint_day),       # 1 week
            '1M': (30, self.endpoint_day)       # 1 month (approximate)
        }

    def get_crypto_data(self, crypto_name='BTC', timeframe='1d', period='1y'):
        """
        Fetch cryptocurrency price data for a specific timeframe and period using CryptoCompare's free API
        
        Args:
            crypto_name (str): Cryptocurrency symbol (e.g. 'BTC', 'ETH', 'XRP')
            timeframe (str): Time interval ('1m', '5m', '15m', '30m', '1h', '2h', '4h', '1d', '1w', '1M')
            period (str): Time period (e.g. '3y' for 3 years, '2w' for 2 weeks, '5d' for 5 days)
            
        Returns:
            list[dict]: Cryptocurrency price data including timestamp, open, high, low, close prices
        """
        import requests
        from datetime import datetime, timedelta
        from time import sleep

        days, endpoint = self.timeframe_mapping.get(timeframe, (1, self.endpoint_day))
        
        # Parse period string to get number and unit
        period_value = int(period[:-1])
        period_unit = period[-1].lower()
        
        # Calculate from_time based on period
        now = datetime.now()
        if period_unit == 'y':
            from_time = now - timedelta(days=period_value*365)
        elif period_unit == 'm':
            from_time = now - timedelta(days=period_value*30)
        elif period_unit == 'w':
            from_time = now - timedelta(weeks=period_value)
        elif period_unit == 'd':
            from_time = now - timedelta(days=period_value)
        else:
            raise ValueError("Invalid period unit. Use 'y' for years, 'm' for months, 'w' for weeks, 'd' for days")
        
        formatted_data = []
        current_time = now
        
        while current_time > from_time:
            # Parameters for the API request
            params = {
                'fsym': crypto_name,
                'tsym': 'USD',
                'toTs': int(current_time.timestamp()),
                'limit': 2000  # API limit
            }
            
            try:
                # Make API request
                response = requests.get(endpoint, params=params)
                response.raise_for_status()
                
                # Parse response data
                data = response.json()['Data']['Data']
                
                # Format the data
                for candle in reversed(data):
                    timestamp = candle['time']
                    candle_time = datetime.fromtimestamp(timestamp)
                    formatted_time = candle_time.strftime('%Y-%m-%d %H:%M:%S')
                    
                    # Only include data within the specified time range
                    if candle_time < from_time:
                        continue
                    if candle_time > now:
                        continue
                        
                    formatted_data.insert(0, {
                        'cryptoname': crypto_name,
                        'timeframe': timeframe,
                        'period': period,
                        'timestamp': formatted_time,
                        'open': float(candle['open']),
                        'high': float(candle['high']),
                        'low': float(candle['low']),
                        'close': float(candle['close']),
                        'volume': float(candle['volumefrom'])
                    })
                
                # Update current_time to fetch next batch
                if data:
                    current_time = datetime.fromtimestamp(data[0]['time'])
                else:
                    break
                    
            except requests.exceptions.RequestException as e:
                print(f"Error fetching {crypto_name} data: {e}")
                return None
                
            # Add small delay to avoid hitting rate limits
            sleep(0.25)
        
        # Convert formatted_data to Crypto dataclass
        crypto_data = [Crypto(**item) for item in formatted_data]
        return crypto_data

    def save_to_pickle(self, crypto_data: list[Crypto], filename: str = None) -> None:
        """Save cryptocurrency data to a compressed pickle file.
        
        Args:
            crypto_data: List of Crypto objects to save
            filename: Optional custom filename, defaults to crypto_timeframe_period.pkl.gz
        """
        if not crypto_data:
            print("No crypto data to save")
            return
            
        # Generate default filename if none provided
        if filename is None:
            sample = crypto_data[0]
            filename = f"{sample.cryptoname}_{sample.timeframe}_{sample.period}.pkl.gz"
            
        try:
            import gzip
            import pickle
            
            # Save compressed pickle file
            with gzip.open(filename, 'wb') as f:
                pickle.dump(crypto_data, f)
                
            print(f"Successfully saved {len(crypto_data)} records to {filename}")
            
        except Exception as e:
            print(f"Error saving pickle file: {e}")
    def save_crypto_data(self, crypto_data: list[Crypto]) -> None:
        """Save cryptocurrency data to the database"""
        from DataBase.PostgreSQL import PostgreSQLDatabase
        db = PostgreSQLDatabase()
        db.save_crypto_data(crypto_data)

    def load_from_pickle(self, filename: str) -> list[Crypto]:
        """Load cryptocurrency data from a compressed pickle file.
        
        Args:
            filename: Name of the pickle file to load
            
        Returns:
            list[Crypto]: List of Crypto objects loaded from the file
        """
        try:
            import gzip
            import pickle
            
            # Load compressed pickle file
            with gzip.open(filename, 'rb') as f:
                crypto_data = pickle.load(f)
                
            print(f"Successfully loaded {len(crypto_data)} records from {filename}")
            return crypto_data
            
        except Exception as e:
            print(f"Error loading pickle file: {e}")
            return None

# Example usage:
crypto_fetcher = CryptoDataFetcher()
crypto_data = crypto_fetcher.get_crypto_data(crypto_name='BTC', timeframe='1d', period='5y')  # Get Bitcoin hourly data for last 3 years

crypto_fetcher.save_crypto_data(crypto_data)
crypto_fetcher.save_to_pickle(crypto_data)
crypto_data = crypto_fetcher.load_from_pickle('BTC_1d_5y.pkl.gz')
crypto_fetcher.save_crypto_data(crypto_data)
# crypto_fetcher.get_crypto_data(crypto_name='ETH', timeframe='1h', period='2w')  # Get Ethereum hourly data for last 2 weeks
# crypto_fetcher.get_crypto_data(crypto_name='XRP', timeframe='15m', period='2d') # Get Ripple 15-min data for last 2 days