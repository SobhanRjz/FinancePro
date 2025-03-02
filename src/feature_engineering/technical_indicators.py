import numpy as np
import pandas as pd
import logging
import gzip
import pickle
from pathlib import Path
from typing import List, Dict, Union, Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.cryptoModel import Crypto

# Configure logger
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

class TechnicalIndicatorExtractor:
    """Class for calculating technical indicators from cryptocurrency price data"""
    
    def __init__(self, data_dir='data/OHLCV'):
        """Initialize the technical indicators calculator"""
        self.data_dir = data_dir
        logger.info("Technical Indicators calculator initialized")
    
    def load_from_pickle(self, filename: str) -> List[Crypto]:
        """Load cryptocurrency data from a compressed pickle file"""
        try:
            with gzip.open(filename, 'rb') as f:
                crypto_data = pickle.load(f)
            logger.info(f"Loaded {len(crypto_data)} records from {filename}")
            return crypto_data
        except Exception as e:
            logger.error(f"Error loading pickle file {filename}: {e}")
            return []
    
    def save_to_pickle(self, data: pd.DataFrame, filename: str) -> None:
        """Save processed data to a compressed pickle file"""
        try:
            # Ensure directory exists
            Path(filename).parent.mkdir(parents=True, exist_ok=True)
            
            with gzip.open(filename, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved processed data to {filename}")
        except Exception as e:
            logger.error(f"Error saving to pickle file {filename}: {e}")
    
    def crypto_to_dataframe(self, crypto_data: List[Crypto]) -> pd.DataFrame:
        """Convert list of Crypto objects to pandas DataFrame"""
        if not crypto_data:
            logger.warning("Empty crypto data provided")
            return pd.DataFrame()
        
        # Extract relevant fields from Crypto objects
        data = [{
            'timestamp': c.timestamp,
            'open': float(c.open) if c.open else None,
            'high': float(c.high) if c.high else None,
            'low': float(c.low) if c.low else None,
            'close': float(c.close) if c.close else None,
            'volume': float(c.volume) if c.volume else None,
            'cryptoname': c.cryptoname,
            'timeframe': c.timeframe,
            'period': c.period,
            'num_trades': c.num_trades,
            'vwap': c.vwap
        } for c in crypto_data]
        
        # Create DataFrame
        df = pd.DataFrame(data)
        
        # Set timestamp as index
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
        
        return df
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators for the given DataFrame"""
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot calculate indicators")
            return df
        
        logger.info("Calculating technical indicators...")
        
        # Make a copy to avoid modifying the original
        result_copy= df.copy()
        result_df = pd.DataFrame()

        # Simple Moving Averages
        result_df['sma_20'] = self._calculate_sma(result_copy, 20)
        result_df['sma_50'] = self._calculate_sma(result_copy, 50)
        result_df['sma_200'] = self._calculate_sma(result_copy, 200)
        
        # Exponential Moving Averages
        result_df['ema_20'] = self._calculate_ema(result_copy, 20)
        result_df['ema_50'] = self._calculate_ema(result_copy, 50)
        result_df['ema_200'] = self._calculate_ema(result_copy, 200)
        
        # MACD
        macd_df = self._calculate_macd(result_copy)
        result_df['macd'] = macd_df['macd']
        result_df['macd_signal'] = macd_df['signal']
        result_df['macd_histogram'] = macd_df['histogram']
        
        # RSI
        result_df['rsi_14'] = self._calculate_rsi(result_copy, 14)
        
        # Bollinger Bands
        bollinger = self._calculate_bollinger_bands(result_copy, 20, 2)
        result_df['bollinger_upper'] = bollinger['upper']
        result_df['bollinger_middle'] = bollinger['middle']
        result_df['bollinger_lower'] = bollinger['lower']
        
        # Average True Range
        result_df['atr_14'] = self._calculate_atr(result_copy, 14)
        
        # Stochastic Oscillator
        stoch = self._calculate_stochastic(result_copy, 14, 3)
        result_df['stochastic_k'] = stoch['k']
        result_df['stochastic_d'] = stoch['d']
        
        # Average Directional Index
        adx_data = self._calculate_adx(result_copy, 14)
        result_df['adx'] = adx_data['adx']
        result_df['plus_di'] = adx_data['plus_di']
        result_df['minus_di'] = adx_data['minus_di']
        
        # Commodity Channel Index
        result_df['cci_14'] = self._calculate_cci(result_copy, 14)
        
        logger.info("Technical indicators calculation completed")
        return result_df
    
    def _calculate_sma(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return df['close'].rolling(window=window, min_periods=1).mean()
    
    def _calculate_ema(self, df: pd.DataFrame, window: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return df['close'].ewm(span=window, adjust=False).mean()
    
    def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        """Calculate MACD, Signal Line, and Histogram"""
        ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
        ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_histogram = macd - macd_signal
        
        return pd.DataFrame({
            'macd': macd,
            'signal': macd_signal,
            'histogram': macd_histogram
        })
    
    def _calculate_rsi(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = df['close'].diff()
        
        # Make two series: one for gains and one for losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calculate average gain and loss
        avg_gain = gain.rolling(window=window, min_periods=1).mean()
        avg_loss = loss.rolling(window=window, min_periods=1).mean()
        
        # Calculate RS
        rs = avg_gain / avg_loss.replace(0, np.finfo(float).eps)  # Avoid division by zero
        
        # Calculate RSI
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_bollinger_bands(self, df: pd.DataFrame, window: int = 20, num_std: float = 2) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        middle_band = df['close'].rolling(window=window, min_periods=1).mean()
        std_dev = df['close'].rolling(window=window, min_periods=1).std()
        
        upper_band = middle_band + (std_dev * num_std)
        lower_band = middle_band - (std_dev * num_std)
        
        return {
            'upper': upper_band,
            'middle': middle_band,
            'lower': lower_band
        }
    
    def _calculate_atr(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate ATR
        atr = tr.rolling(window=window, min_periods=1).mean()
        
        return atr
    
    def _calculate_stochastic(self, df: pd.DataFrame, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        # Calculate %K
        lowest_low = df['low'].rolling(window=k_window, min_periods=1).min()
        highest_high = df['high'].rolling(window=k_window, min_periods=1).max()
        
        k = 100 * ((df['close'] - lowest_low) / (highest_high - lowest_low + np.finfo(float).eps))
        
        # Calculate %D (3-day SMA of %K)
        d = k.rolling(window=d_window, min_periods=1).mean()
        
        return {
            'k': k,
            'd': d
        }
    
    def _calculate_adx(self, df: pd.DataFrame, window: int = 14) -> Dict[str, pd.Series]:
        """Calculate Average Directional Index (ADX) with Wilder's smoothing"""
        high = df['high']
        low = df['low']
        close = df['close']

        # Calculate directional movement
        up_move = high.diff()
        down_move = low.shift() - low

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

        # True Range (TR)
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Initialize smoothed values
        smoothed_plus_dm = np.zeros(len(df))
        smoothed_minus_dm = np.zeros(len(df))
        smoothed_tr = np.zeros(len(df))

        # Initial values - sum over first 'window' periods
        smoothed_plus_dm[window - 1] = plus_dm[:window].sum()
        smoothed_minus_dm[window - 1] = minus_dm[:window].sum()
        smoothed_tr[window - 1] = tr[:window].sum()

        # Wilder’s smoothing applied from 'window' onwards
        for i in range(window, len(df)):
            smoothed_plus_dm[i] = (smoothed_plus_dm[i-1] * (window - 1) + plus_dm[i]) / window
            smoothed_minus_dm[i] = (smoothed_minus_dm[i-1] * (window - 1) + minus_dm[i]) / window
            smoothed_tr[i] = (smoothed_tr[i-1] * (window - 1) + tr[i]) / window

        # Calculate DI
        plus_di = 100 * (smoothed_plus_dm / (smoothed_tr + np.finfo(float).eps))
        minus_di = 100 * (smoothed_minus_dm / (smoothed_tr + np.finfo(float).eps))

        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + np.finfo(float).eps)

        adx = np.zeros(len(df))
        adx[window-1] = dx[:window].mean()

        # Smooth ADX (Wilder's method)
        for i in range(window, len(df)):
            adx[i] = (adx[i-1] * (window - 1) + dx[i]) / window

        return {
            'adx': pd.Series(adx, index=df.index),
            'plus_di': pd.Series(plus_di, index=df.index),
            'minus_di': pd.Series(minus_di, index=df.index)
        }
    
    def _calculate_cci(self, df: pd.DataFrame, window: int = 14) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        
        # Calculate moving average of typical price
        ma_tp = typical_price.rolling(window=window, min_periods=1).mean()
        
        # Calculate mean deviation
        mean_deviation = abs(typical_price - ma_tp).rolling(window=window, min_periods=1).mean()
        
        # Calculate CCI
        cci = (typical_price - ma_tp) / (0.015 * mean_deviation.replace(0, np.finfo(float).eps))
        
        return cci
    
    def process_crypto_data(self, input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Process cryptocurrency data by calculating technical indicators
        
        Args:
            input_file: Path to the input pickle file
            output_file: Optional path to save the processed data
            
        Returns:
            pd.DataFrame: DataFrame with calculated technical indicators
        """
        # Load data
        crypto_data = self.load_from_pickle(input_file)
        
        if not crypto_data:
            logger.error(f"No data loaded from {input_file}")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = self.crypto_to_dataframe(crypto_data)
        
        # Calculate indicators
        result_df = self.calculate_indicators(df)
        result_df = result_df.ffill().bfill()

        # Save processed data if output file is specified
        if output_file:
            self.save_to_pickle(result_df, output_file)
        
        return result_df

    def process_all_files(self):
        """Process all cryptocurrency data files in the specified directory
        
        Args:
            data_dir: Directory containing input pickle files
            output_dir: Directory to save processed data files
        """
        output_dir = 'data/process_technical_indicators'
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all pickle files in the data directory
        input_files = list(Path(self.data_dir).glob('*.pkl.gz'))
        
        if not input_files:
            logger.warning(f"No pickle files found in {self.data_dir}")
            return
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Process each file
        for input_file in input_files:
            try:
                # Generate output filename
                output_file = Path(output_dir) / f"processed_{input_file.name}"
                
                logger.info(f"Processing {input_file}")
                self.process_crypto_data(str(input_file), str(output_file))
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")

# Example usage
if __name__ == "__main__":
    indicator_calculator = TechnicalIndicatorExtractor()

    # Process all files in the data directory
    indicator_calculator.process_all_crypto_files()
