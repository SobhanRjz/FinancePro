import sys
import logging
from pathlib import Path
from datetime import datetime, timedelta
from time import sleep
import ccxt
import httpx
from playwright.sync_api import sync_playwright
from bs4 import BeautifulSoup
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.cryptoModel import Crypto

import json
import gzip
import pickle

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class CoreFeatureExtractor:
    """Class for fetching cryptocurrency price data using Binance API through CCXT"""

    def __init__(self):
        """
        Initialize with a pre-configured ccxt exchange instance.
        This allows you to inject custom headers, URLs, etc.
        """
        self.timeframe_mapping = {
            '1m': '1m', '3m': '3m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '1d': '1d', '1w': '1w', '1M': '1M'
        }
        logger.info("CryptoDataFetcher initialized")

    def fetch_currency_data(self, symbol='BTCUSDT', timeframe='3m', period='1d'):
        """Fetch cryptocurrency candle data using Playwright
        
        Args:
            symbol (str): Trading pair symbol (default: 'BTCUSDT')
            timeframe (str): Candle timeframe (default: '3m')
            period (str): Time period to fetch (default: '1d', format: '1d', '1w', '1y', etc.)
            
        Returns:
            list[Crypto]: List of Crypto objects with the fetched data
        """
        logger.info(f"Fetching {symbol} data for {period} with {timeframe} timeframe")
        
        # Convert period to number of candles
        period_value = int(period[:-1])
        period_unit = period[-1].lower()
        
        # Calculate number of candles based on period and timeframe
        timeframe_minutes = self._calculate_timeframe_minutes(timeframe)
        total_candles = self._calculate_total_candles(period_value, period_unit, timeframe_minutes)
        
        # Extract crypto name from symbol (assuming format like BTCUSDT)
        crypto_name = symbol.replace('USDT', '')
        
        # Binance API limit per request
        limit = 1000
        
        # Calculate end time (now)
        end_time = int(datetime.now().timestamp() * 1000)
        
        formatted_data = []
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            
            try:
                formatted_data = self._fetch_historical_data(
                    page, symbol, timeframe, period, crypto_name, 
                    total_candles, end_time, limit
                )
            except Exception as e:
                logger.error(f"Error fetching data from Binance API: {e}")
                return []
            finally:
                browser.close()
            
            # Sort data by timestamp (oldest to newest)
            formatted_data.sort(key=lambda x: datetime.strptime(x['timestamp'], '%Y-%m-%d %H:%M:%S'))
            
            logger.info(f"Successfully fetched {len(formatted_data)} candles for {symbol}")
            return [Crypto(**item) for item in formatted_data]

    def _fetch_historical_data(self, page, symbol, timeframe, period, crypto_name, 
                                    total_candles, end_time, limit):
        """Helper method to fetch historical data with pagination"""
        formatted_data = []
        remaining_candles = total_candles
        current_end_time = end_time
        price_volume_sum = 0
        volume_sum = 0
        while remaining_candles > 0:
            # Calculate how many candles to fetch in this request
            batch_size = min(remaining_candles, limit)
            
            # Construct URL with endTime parameter
            url = f"https://api.binance.com/api/v3/klines?symbol={symbol}&interval={timeframe}&limit={batch_size}&endTime={current_end_time}"
            
            page.goto(url)
            
            # Extract JSON response as plain text from the page body
            content = page.evaluate("() => document.body.innerText")
            
            try:
                data = json.loads(content)
                
                if not data:  # No more data available
                    break
                    
                # Process the candles
                for candle in data:
                    timestamp = candle[0] / 1000
                    candle_time = datetime.fromtimestamp(timestamp)

                    # Calculate typical price
                    typical_price = (float(candle[2]) + float(candle[3]) + float(candle[4])) / 3

                    # Update cumulative sums
                    price_volume_sum += typical_price * float(candle[5])
                    volume_sum += float(candle[5])

                    # Calculate VWAP (up to this candle)
                    vwap = price_volume_sum / volume_sum if volume_sum > 0 else None
                    
                    formatted_data.append({
                        'cryptoname': crypto_name,
                        'timeframe': timeframe,
                        'period': period,
                        'timestamp': candle_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'open': float(candle[1]),
                        'high': float(candle[2]),
                        'low': float(candle[3]),
                        'close': float(candle[4]),
                        'volume': float(candle[5]),
                        'num_trades': int(candle[8]),
                        'vwap': vwap # Volume Weighted Average Price
                    })
                
                # Update parameters for next request
                remaining_candles -= len(data)
                
                if len(data) > 0:
                    # Set the end time to the timestamp of the oldest candle minus 1ms
                    current_end_time = data[0][0] - 1
                else:
                    break
                    
                # Add a small delay to avoid rate limiting
                sleep(0.5)
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.debug(f"Raw content: {content[:100]}...")  # Print first 100 chars for debugging
                break
        
        return formatted_data

    def _calculate_timeframe_minutes(self, timeframe):
        """Calculate minutes in the given timeframe"""
        if timeframe[-1] == 'm':
            return int(timeframe[:-1])
        elif timeframe[-1] == 'h':
            return int(timeframe[:-1]) * 60
        elif timeframe[-1] == 'd':
            return int(timeframe[:-1]) * 1440
        elif timeframe[-1] == 'w':
            return int(timeframe[:-1]) * 10080
        else:  # For months
            return int(timeframe[:-1]) * 43200

    def _calculate_total_candles(self, period_value, period_unit, timeframe_minutes):
        """Calculate total candles based on period and timeframe"""
        if period_unit == 'y':
            return period_value * 525600 // timeframe_minutes  # minutes in a year
        elif period_unit == 'm':
            return period_value * 43200 // timeframe_minutes   # minutes in a month (30 days)
        elif period_unit == 'w':
            return period_value * 10080 // timeframe_minutes   # minutes in a week
        elif period_unit == 'd':
            return period_value * 1440 // timeframe_minutes    # minutes in a day
        else:
            raise ValueError("Invalid period unit. Use 'y', 'm', 'w', 'd'.")

    def get_crypto_data(self, crypto_name='BTC', timeframe='1d', period='1y'):
        """
        Fetch cryptocurrency price data using CCXT.
        
        Args:
            crypto_name (str): Cryptocurrency name (default: 'BTC')
            timeframe (str): Candle timeframe (default: '1d')
            period (str): Time period to fetch (default: '1y')
            
        Returns:
            list[Crypto]: List of Crypto objects with the fetched data
        """
        logger.info(f"Fetching {crypto_name} data for {period} with {timeframe} timeframe using CCXT")
        
        period_value = int(period[:-1])
        period_unit = period[-1].lower()

        now = datetime.now()
        from_time = self._calculate_from_time(now, period_value, period_unit)

        formatted_data = []
        symbol = f"{crypto_name}/USDT"
        tf = self.timeframe_mapping[timeframe]

        try:
            ohlcv = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=tf,
                since=int(from_time.timestamp() * 1000),
                limit=1000
            )

            for candle in ohlcv:
                timestamp = candle[0] / 1000
                candle_time = datetime.fromtimestamp(timestamp)
                formatted_data.append({
                    'cryptoname': crypto_name,
                    'timeframe': timeframe,
                    'period': period,
                    'timestamp': candle_time.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': float(candle[1]),
                    'high': float(candle[2]),
                    'low': float(candle[3]),
                    'close': float(candle[4]),
                    'volume': float(candle[5])
                })

            sleep(0.25)  # Rate limiting
            logger.info(f"Successfully fetched {len(formatted_data)} candles for {crypto_name}")

        except Exception as e:
            logger.error(f"Error fetching {crypto_name} data: {e}")
            return []  # Return empty list instead of None for consistency

        return [Crypto(**item) for item in formatted_data]

    def _calculate_from_time(self, now, period_value, period_unit):
        """Calculate the start time based on period"""
        if period_unit == 'y':
            return now - timedelta(days=period_value * 365)
        elif period_unit == 'm':
            return now - timedelta(days=period_value * 30)
        elif period_unit == 'w':
            return now - timedelta(weeks=period_value)
        elif period_unit == 'd':
            return now - timedelta(days=period_value)
        else:
            raise ValueError("Invalid period unit. Use 'y', 'm', 'w', 'd'.")

    def save_to_pickle(self, crypto_data: list[Crypto], filename: str = None) -> None:
        """Save crypto data to a compressed pickle file"""
        if not crypto_data:
            logger.warning("No crypto data to save")
            return

        if filename is None:
            if not crypto_data:
                logger.warning("Cannot generate filename: empty crypto data")
                return
            sample = crypto_data[0]
            filename = f"{sample.cryptoname}_{sample.timeframe}_{sample.period}.pkl.gz"

        # Ensure directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        try:
            with gzip.open(filename, 'wb') as f:
                pickle.dump(crypto_data, f)
            logger.info(f"Saved {len(crypto_data)} records to {filename}")
        except Exception as e:
            logger.error(f"Error saving to pickle file {filename}: {e}")

    def save_to_database(self, crypto_data: list[Crypto]) -> None:
        """Save crypto data to PostgreSQL database"""
        if not crypto_data:
            logger.warning("No crypto data to save to database")
            return
            
        try:
            from dataBase.PostgreSQL import PostgreSQLDatabase
            db = PostgreSQLDatabase()
            db.save_crypto_data(crypto_data)
            logger.info(f"Saved {len(crypto_data)} records to database")
        except Exception as e:
            logger.error(f"Error saving to database: {e}")

    def load_from_pickle(self, filename: str) -> list[Crypto]:
        """Load crypto data from a compressed pickle file"""
        try:
            with gzip.open(filename, 'rb') as f:
                crypto_data = pickle.load(f)
            logger.info(f"Loaded {len(crypto_data)} records from {filename}")
            return crypto_data
        except Exception as e:
            logger.error(f"Error loading pickle file {filename}: {e}")
            return []  # Return empty list instead of None for consistency

    def process_all_files(self, symbol='BTCUSDT', timeframe='5m', period='1d'):
        """Run the data collection process for the specified symbol and timeframe"""
        logger.info(f"Starting data collection for {symbol} with {timeframe} timeframe for {period}")
        
        try:
            # Fetch data using Playwright
            currency_data = self.fetch_currency_data(symbol=symbol, timeframe=timeframe, period=period)
            
            if currency_data:
                # Save data to database and pickle file
                self.save_to_database(currency_data)
                self.save_to_pickle(currency_data, f'data/{symbol}_{timeframe}_{period}.pkl.gz')
                logger.info(f"Data collection completed for {symbol}")
            else:
                logger.warning(f"No data fetched for {symbol}")
        except Exception as e:
            logger.error(f"Error in data collection for {symbol}: {e}")
        
        
if __name__ == "__main__":
    import ccxt
    from models.cryptoModel import Crypto
    
    try:
        # Initialize exchange and data fetcher
        crypto_fetcher = CoreFeatureExtractor()
        
        # Define timeframes to collect
        timeframes = [ '5m', '1h', '4h', '1d']
        symbol = 'BTCUSDT'
        
        # Collect data for each timeframe with recommended periods
        for timeframe in timeframes:
            period = Crypto.get_recommended_period(timeframe)
            logger.info(f"Collecting {symbol} data for {timeframe} timeframe (period: {period})")
            crypto_fetcher.process_all_files(
                symbol=symbol,
                timeframe=timeframe,
                period=period
            )
            
        logger.info("Data collection completed successfully for all timeframes")
    except Exception as e:
        logger.critical(f"Fatal error in data collection: {e}", exc_info=True)
