import pandas as pd
import numpy as np
import os
import glob
import re
import logging
import ta
from typing import Optional
import sys
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sys.path.append(str(Path(__file__).parent.parent.parent))


class DerivedFeatureExtractor:
    """Class for extracting derived features from OHLCV data"""
    
    def __init__(self, data_dir: str = 'data/OHLCV'):
        """
        Initialize DerivedFeatureExtractor
        
        Args:
            data_dir (str): Directory containing OHLCV data files
        """
        self.data_dir = data_dir
        self.timeframe_map = {
            '1d': 'D',
            '4h': '4h', 
            '1h': 'h',
            '5m': '5min'
        }
    
    
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
            resampled_df.index = resampled_df.index - pd.Timedelta(minutes=time_diff.total_seconds()/60)


        return resampled_df
    
    def calculate_momentum_score(self, df : pd.DataFrame, timeframe : str,  start_time:str):
        """
        Calculate momentum score based on RSI, ROC, and EMA indicators
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            
        Returns:
            pd.DataFrame: DataFrame with momentum indicators and score
        """
        dfTemp = pd.DataFrame()
        dfTemp['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        dfTemp['roc'] = ta.momentum.ROCIndicator(df['close']).roc()
        dfTemp['ema'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
        dfTemp['ema_slope'] = dfTemp['ema'].diff()

        # Normalize indicators to [0, 1] range
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())

        dfTemp['rsi_score'] = normalize(dfTemp['rsi'])
        dfTemp['roc_score'] = normalize(dfTemp['roc'])
        dfTemp['ema_slope_score'] = normalize(dfTemp['ema_slope'])

        # Combine into momentum score (simple average)
        dfTemp['momentum_score'] = dfTemp[['rsi_score', 'roc_score', 'ema_slope_score']].mean(axis=1)
                # Fill initial NaN/0 values with first non-zero value
        first_valid = dfTemp['momentum_score'][dfTemp['momentum_score'] > 0].iloc[0]
        dfTemp['momentum_score'] = dfTemp['momentum_score'].replace(0, first_valid)

        # Ensure we have a DatetimeIndex before resampling
        dfTemp = dfTemp.copy()
        dfTemp['timestamp'] = df.index
        
        # Check if index is already a DatetimeIndex, if not convert it
        if not isinstance(dfTemp.index, pd.DatetimeIndex):
            dfTemp = dfTemp.set_index('timestamp')
            # Ensure the index is DatetimeIndex
            if not isinstance(dfTemp.index, pd.DatetimeIndex):
                dfTemp.index = pd.to_datetime(dfTemp.index)
        
        # Now resample with proper DatetimeIndex
        dfTemp = self.resample_data(dfTemp, timeframe, start_time)
        dfTemp = dfTemp.ffill().bfill()
        
        dfTemp = dfTemp[['momentum_score']]

        return dfTemp
    
    def calculate_mean_reversion_signal(self, df: pd.DataFrame, timeframe: str,  start_time:str):
        """
        Calculate mean reversion signals based on Bollinger Bands and Z-Score
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            timeframe (str): Timeframe to resample to
            
        Returns:
            pd.DataFrame: DataFrame with mean reversion indicators and signals
        """
        dfTemp = pd.DataFrame()
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
        dfTemp['bb_upper'] = bb.bollinger_hband()
        dfTemp['bb_lower'] = bb.bollinger_lband()
        dfTemp['bb_mean'] = bb.bollinger_mavg()

        # Z-Score (distance from mean in standard deviations)
        dfTemp['price_mean'] = df['close'].rolling(20).mean()
        dfTemp['price_std'] = df['close'].rolling(20).std()
        dfTemp['z_score'] = (df['close'] - dfTemp['price_mean']) / dfTemp['price_std']

        # Mean Reversion Signal
        dfTemp['mean_reversion_signal'] = 0
        dfTemp.loc[df['close'] > dfTemp['bb_upper'], 'mean_reversion_signal'] = -1  # Price far above = short signal
        dfTemp.loc[df['close'] < dfTemp['bb_lower'], 'mean_reversion_signal'] = 1   # Price far below = long signal

        # Optional: Add Z-Score-based signals (extreme deviations)
        dfTemp.loc[dfTemp['z_score'] > 2, 'mean_reversion_signal'] = -1
        dfTemp.loc[dfTemp['z_score'] < -2, 'mean_reversion_signal'] = 1
        
        # Ensure we have a DatetimeIndex before resampling
        dfTemp = dfTemp.copy()
        dfTemp['timestamp'] = df.index
        
        # Check if index is already a DatetimeIndex, if not convert it
        if not isinstance(dfTemp.index, pd.DatetimeIndex):
            dfTemp = dfTemp.set_index('timestamp')
            # Ensure the index is DatetimeIndex
            if not isinstance(dfTemp.index, pd.DatetimeIndex):
                dfTemp.index = pd.to_datetime(dfTemp.index)
        
        # Now resample with proper DatetimeIndex
        dfTemp = self.resample_data(dfTemp, timeframe,  start_time)
        dfTemp = dfTemp.ffill().bfill()
        
        return dfTemp[['bb_upper', 'bb_lower', 'z_score', 'mean_reversion_signal']]
        
    def calculate_trend_strength(self, df: pd.DataFrame, timeframe: str,  start_time:str):
        """
        Calculate trend strength based on ADX, EMA slope, and ROC indicators
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            timeframe (str): Timeframe to resample to
            
        Returns:
            pd.DataFrame: DataFrame with trend strength indicators and score
        """
        dfTemp = pd.DataFrame()
        
        # ADX - Average Directional Index
        adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
        dfTemp['adx'] = adx_indicator.adx()
        # Fill initial NaN/0 values with first non-zero value
        first_valid = dfTemp['adx'][dfTemp['adx'] > 0].iloc[0]
        dfTemp['adx'] = dfTemp['adx'].replace(0, first_valid)

        # EMA and its slope (trend slope)
        dfTemp['ema'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
        dfTemp['ema_slope'] = dfTemp['ema'].diff()

        # Rate of Change (ROC) - measures price speed (momentum)
        dfTemp['roc'] = ta.momentum.ROCIndicator(df['close']).roc()

        # Normalize indicators to [0, 1] range
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())

        # Normalize all indicators
        dfTemp['adx_score'] = normalize(dfTemp['adx'])
        dfTemp['ema_slope_score'] = normalize(dfTemp['ema_slope'])
        dfTemp['roc_score'] = normalize(dfTemp['roc'])

        # Combine into trend strength score (average)
        dfTemp['trend_strength'] = dfTemp[['adx_score', 'ema_slope_score', 'roc_score']].mean(axis=1)

        # Ensure we have a DatetimeIndex before resampling
        dfTemp = dfTemp.copy()
        dfTemp['timestamp'] = df.index
        
        # Check if index is already a DatetimeIndex, if not convert it
        if not isinstance(dfTemp.index, pd.DatetimeIndex):
            dfTemp = dfTemp.set_index('timestamp')
            # Ensure the index is DatetimeIndex
            if not isinstance(dfTemp.index, pd.DatetimeIndex):
                dfTemp.index = pd.to_datetime(dfTemp.index)
        
        # Now resample with proper DatetimeIndex
        dfTemp = self.resample_data(dfTemp, timeframe,  start_time)
        dfTemp = dfTemp.ffill().bfill()

        # Extract only the trend_strength column
        dfTemp = dfTemp[['trend_strength']]

        return dfTemp
    
    def calculate_trend_continuation_probability(self, df: pd.DataFrame, timeframe: str,  start_time:str):
        """
        Calculate trend continuation probability based on ADX, EMA slope, MACD histogram, and price-EMA relationship
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            timeframe (str): Timeframe to resample to
            
        Returns:
            pd.DataFrame: DataFrame with trend continuation probability indicators
        """
        dfTemp = pd.DataFrame()
        
        # ADX (trend strength)
        adx_indicator = ta.trend.ADXIndicator(high=df['high'], low=df['low'], close=df['close'])
        dfTemp['adx'] = adx_indicator.adx()

        # EMA and Slope
        dfTemp['ema'] = ta.trend.EMAIndicator(df['close'], window=14).ema_indicator()
        dfTemp['ema_slope'] = dfTemp['ema'].diff()

        # MACD Histogram (trend momentum)
        macd = ta.trend.MACD(df['close'])
        dfTemp['macd_histogram'] = macd.macd_diff()

        # Is price above EMA? (1 = trend intact, 0 = trend weakening)
        dfTemp['above_ema'] = (df['close'] > dfTemp['ema']).astype(float)

        # Normalize indicators to 0-1 range
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())
            
        dfTemp['adx_score'] = normalize(dfTemp['adx'])
        dfTemp['ema_slope_score'] = normalize(dfTemp['ema_slope'])
        dfTemp['macd_histogram_score'] = normalize(dfTemp['macd_histogram'])

        # Composite trend continuation probability
        dfTemp['trend_continuation_probability'] = (
            (dfTemp['adx_score'] +
             dfTemp['ema_slope_score'] +
             dfTemp['macd_histogram_score'] +
             dfTemp['above_ema']) / 4
        )
        
        # Ensure we have a DatetimeIndex before resampling
        dfTemp = dfTemp.copy()
        dfTemp['timestamp'] = df.index
        
        # Check if index is already a DatetimeIndex, if not convert it
        if not isinstance(dfTemp.index, pd.DatetimeIndex):
            dfTemp = dfTemp.set_index('timestamp')
            # Ensure the index is DatetimeIndex
            if not isinstance(dfTemp.index, pd.DatetimeIndex):
                dfTemp.index = pd.to_datetime(dfTemp.index)
        
        # Now resample with proper DatetimeIndex
        dfTemp = self.resample_data(dfTemp, timeframe, start_time)
        dfTemp = dfTemp.ffill().bfill()

        return dfTemp[['ema_slope', 'macd_histogram', 'above_ema', 'trend_continuation_probability']]
    
    def calculate_volatility_cluster_indicator(self, df: pd.DataFrame, timeframe: str, start_time:str, window=20):
        """
        Calculate volatility clustering indicators based on standard deviation, ATR, and Bollinger Band width
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            timeframe (str): Timeframe to resample to
            window (int): Window size for calculations
            
        Returns:
            pd.DataFrame: DataFrame with volatility cluster indicators
        """
        dfTemp = pd.DataFrame()
        
        # Calculate returns
        dfTemp['price'] = df['close']
        dfTemp['returns'] = df['close'].pct_change()
        
        # Rolling standard deviation of returns (volatility measure)
        dfTemp['volatility'] = dfTemp['returns'].rolling(window).std()

        # ATR (captures true volatility movements)
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=window)
        dfTemp['atr'] = atr.average_true_range()

        # Bollinger Band Width (relative measure of volatility expansion/contraction)
        bb = ta.volatility.BollingerBands(df['close'], window=window, window_dev=2)
        dfTemp['bb_width'] = bb.bollinger_wband()

        # Normalize to a 0-1 scale for aggregation
        def normalize(series):
            return (series - series.min()) / (series.max() - series.min())
            
        dfTemp['volatility_score'] = normalize(dfTemp['volatility'])
        dfTemp['atr_score'] = normalize(dfTemp['atr'])
        dfTemp['bb_width_score'] = normalize(dfTemp['bb_width'])

        # Final Volatility Cluster Indicator (average of all measures)
        dfTemp['volatility_cluster_indicator'] = (dfTemp['volatility_score'] + dfTemp['atr_score'] + dfTemp['bb_width_score']) / 3

        # Ensure we have a DatetimeIndex before resampling
        dfTemp = dfTemp.copy()
        dfTemp['timestamp'] = df.index
        
        # Check if index is already a DatetimeIndex, if not convert it
        if not isinstance(dfTemp.index, pd.DatetimeIndex):
            dfTemp = dfTemp.set_index('timestamp')
            # Ensure the index is DatetimeIndex
            if not isinstance(dfTemp.index, pd.DatetimeIndex):
                dfTemp.index = pd.to_datetime(dfTemp.index)
        
        # Now resample with proper DatetimeIndex
        dfTemp = self.resample_data(dfTemp, timeframe,  start_time)
        dfTemp = dfTemp.ffill().bfill()

        return dfTemp[['bb_width', 'volatility_cluster_indicator']]
    
    def classify_market_regime(self, df: pd.DataFrame, timeframe: str, start_time:str):
        """
        Classify market regime as Bull, Bear, or Sideways based on technical indicators
        
        Args:
            df (pd.DataFrame): DataFrame with price data
            timeframe (str): Timeframe to resample to
            
        Returns:
            pd.DataFrame: DataFrame with market regime classification
        """
        dfTemp = pd.DataFrame()
        
        # Price data
        dfTemp['price'] = df['close']
        
        # Long-Term EMA
        dfTemp['ema_200'] = ta.trend.EMAIndicator(dfTemp['price'], window=200).ema_indicator()
        
        # Slope of EMA (Trend Strength)
        dfTemp['ema_slope'] = dfTemp['ema_200'].diff()
        
        # RSI (Momentum Indicator)
        dfTemp['rsi'] = ta.momentum.RSIIndicator(dfTemp['price'], window=14).rsi()

        # MACD Histogram (Momentum Confirmation)
        macd = ta.trend.MACD(dfTemp['price'])
        dfTemp['macd_histogram'] = macd.macd_diff()

        # ATR-Based Volatility Detection
        atr = ta.volatility.AverageTrueRange(high=df['high'], low=df['low'], close=df['close'])
        dfTemp['atr'] = atr.average_true_range()

        # Market Regime Classification
        dfTemp['market_regime'] = 'Sideways'  # Default

        dfTemp.loc[(dfTemp['price'] > dfTemp['ema_200']) & (dfTemp['ema_slope'] > 0) & (dfTemp['rsi'] > 55) & (dfTemp['macd_histogram'] > 0), 'market_regime'] = 'Bull'
        
        dfTemp.loc[(dfTemp['price'] < dfTemp['ema_200']) & (dfTemp['ema_slope'] < 0) & (dfTemp['rsi'] < 45) & (dfTemp['macd_histogram'] < 0), 'market_regime'] = 'Bear'
        
        dfTemp.loc[(dfTemp['atr'] < dfTemp['atr'].rolling(50).mean()), 'market_regime'] = 'Sideways'  # Low volatility = sideways
        
        # Ensure we have a DatetimeIndex before resampling
        dfTemp = dfTemp.copy()
        dfTemp['timestamp'] = df.index
        
        # Check if index is already a DatetimeIndex, if not convert it
        if not isinstance(dfTemp.index, pd.DatetimeIndex):
            dfTemp = dfTemp.set_index('timestamp')
            # Ensure the index is DatetimeIndex
            if not isinstance(dfTemp.index, pd.DatetimeIndex):
                dfTemp.index = pd.to_datetime(dfTemp.index)
        
        # Now resample with proper DatetimeIndex
        dfTemp = self.resample_data(dfTemp, timeframe, start_time)
        dfTemp = dfTemp.ffill().bfill()
        
        return dfTemp[['market_regime']]
    
    def process_all_files(self):
        """Process all OHLCV files and extract derived features"""
        
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
                
                try:
                    # Load the OHLCV data
                    df = pd.read_pickle(file_path)
                    
                    # Ensure df is a DataFrame
                    if not isinstance(df, pd.DataFrame):
                        logging.warning(f"Data from {file_path} is not a DataFrame. Converting...")
                        try:
                            df = pd.DataFrame(df)
                        except Exception as e:
                            logging.error(f"Failed to convert data to DataFrame: {e}")
                            continue
                    
                    # Check if DataFrame is empty
                    if df.empty:
                        logging.warning(f"DataFrame from {file_path} is empty. Skipping...")
                        continue
                        
                    # Log DataFrame info for debugging
                    logging.info(f"DataFrame shape: {df.shape}")
                    logging.info(f"DataFrame columns: {df.columns.tolist()}")

                    #df.index.name = 'timestamp'
                    df = df.reset_index().set_index('timestamp')
                    start_date_str = df.index.min()
                    end_data_str = df.index.max()
                    # Calculate momentum score
                    market_regime_df = self.classify_market_regime(df, timeframe, start_date_str)
                    volatility_cluster_df = self.calculate_volatility_cluster_indicator(df, timeframe, start_date_str)
                    trend_continuation_df = self.calculate_trend_continuation_probability(df, timeframe, start_date_str)
                    mean_reversion_df = self.calculate_mean_reversion_signal(df, timeframe, start_date_str)
                    trend_strength_df = self.calculate_trend_strength(df, timeframe, start_date_str)
                    momentum_df = self.calculate_momentum_score(df, timeframe, start_date_str)
                    
                    # Merge all derived features
                    derived_features_df = pd.concat([market_regime_df, volatility_cluster_df, trend_continuation_df, mean_reversion_df, trend_strength_df, momentum_df], axis=1)
                    
                    # Create output directory
                    output_dir = os.path.join(self.data_dir.split('/')[0], '7_process_derived_features')
                    os.makedirs(output_dir, exist_ok=True)
                    
                    # Create output filename with same pattern
                    output_file = os.path.join(output_dir, f'derived_features_{symbol}_{timeframe}_{period}.pkl.gz')
                    
                    # Save the derived features
                    derived_features_df.to_pickle(output_file)
                    logging.info(f"Saved derived features to: {output_file}")
                    
                except Exception as e:
                    logging.error(f"Error processing file {file_name}: {e}")
            else:
                logging.warning(f"Could not parse pattern from filename: {file_name}")

if __name__ == "__main__":
    # Initialize extractor and process files
    extractor = DerivedFeatureExtractor()
    extractor.process_all_files()
