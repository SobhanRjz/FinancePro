import pandas as pd
import numpy as np
import gzip
import pickle
import logging
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime
import holidays
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.cryptoModel import Crypto

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TimeFeatureExtractor:
    """Class for calculating time-based features from cryptocurrency data"""
    
    def __init__(self):
        # Initialize holidays for major markets
        self.us_holidays = holidays.US()
        self.eu_holidays = holidays.EuropeanCentralBank()
        
    def load_data(self, filename: str) -> pd.DataFrame:
        """Load processed data from a compressed pickle file"""
        try:
            with gzip.open(filename, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded data from {filename}")
            return data
        except Exception as e:
            logger.error(f"Error loading pickle file {filename}: {e}")
            return pd.DataFrame()
    
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
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate time-based features for the given DataFrame"""
        if df.empty:
            logger.warning("Empty DataFrame provided, cannot calculate time features")
            return df
        
        logger.info("Calculating time features...")
        
        # Make a copy to avoid modifying the original
        result_df = pd.DataFrame()
        
        # Reset index to access timestamp as a column
        if isinstance(df.index, pd.DatetimeIndex):
            result_df = df.reset_index()
        
        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
        # Day of week (0 = Monday, 6 = Sunday)
        result_df['day_of_week'] = df['timestamp'].dt.dayofweek
        
        # Hour of day (0-23)
        result_df['hour_of_day'] = df['timestamp'].dt.hour
        
        # Is weekend (Saturday or Sunday)
        result_df['is_weekend'] = result_df['day_of_week'].isin([5, 6]).astype(int)
        
        # Is holiday (US or EU)
        result_df['is_holiday'] = df['timestamp'].dt.date.apply(
            lambda x: x in self.us_holidays or x in self.eu_holidays
        ).astype(int)
        
        # Market session
        # Asia: 22:00-07:00 UTC
        # Europe: 07:00-16:00 UTC
        # US: 13:00-22:00 UTC
        self.market_sessions = [
        ('Asia', (0, 7)),
        ('EU', (7, 13)),
        ('US', (13, 22)),
        ('Asia', (22, 24))  # Asia overlaps the night
        ]

        def get_market_session(hour):
            for session, (start, end) in self.market_sessions:
                if start <= hour < end:
                    return session
            return 'Unknown'
        
        result_df['market_session'] = result_df['hour_of_day'].apply(get_market_session)
        
        # Convert market_session to categorical for better memory usage
        result_df['market_session'] = pd.Categorical(result_df['market_session'], 
                                                    categories=['Asia', 'EU', 'US'])
        
        # Set timestamp back as index if it was the index before
        if isinstance(df.index, pd.DatetimeIndex):
            result_df = result_df.set_index('timestamp')
        
        logger.info("Time features calculation completed")
        return result_df
    
    def process_crypto_data(self, input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Process cryptocurrency data by calculating time features
        
        Args:
            input_file: Path to the input pickle file with statistical features
            output_file: Optional path to save the processed data
            
        Returns:
            pd.DataFrame: DataFrame with calculated time features
        """
        # Load data
        df = self.load_data(input_file)
        
        df = pd.DataFrame(df)
        if df.empty:
            logger.error(f"No data loaded from {input_file}")
            return pd.DataFrame()
        
        # Calculate time features
        result_df = self.calculate_time_features(df)
        
        result_df = result_df.ffill().bfill()
        # Save processed data if output file is specified
        if output_file:
            self.save_to_pickle(result_df, output_file)
        
        return result_df

    def process_all_files(self, data_dir: str = 'data/OHLCV', 
                                output_dir: str = 'data/process_time_features'):
        """Process all cryptocurrency data files in the specified directory
        
        Args:
            data_dir: Directory containing input pickle files with statistical features
            output_dir: Directory to save processed data files
        """
        # Create output directory if it doesn't exist
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Get all pickle files in the data directory
        input_files = list(Path(data_dir).glob('*.pkl.gz'))
        
        if not input_files:
            logger.warning(f"No pickle files found in {data_dir}")
            return
        
        logger.info(f"Found {len(input_files)} files to process")
        
        # Process each file
        for input_file in input_files:
            try:
                # Generate output filename
                output_file = Path(output_dir) / f"time_{input_file.name}"
                
                logger.info(f"Processing {input_file}")
                self.process_crypto_data(str(input_file), str(output_file))
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")
                

# Example usage
if __name__ == "__main__":
    time_feature_calculator = TimeFeatureExtractor()

    # Process all files in the data directory
    time_feature_calculator.process_all_files()
