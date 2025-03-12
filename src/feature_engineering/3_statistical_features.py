import numpy as np
import pandas as pd
from pathlib import Path
import logging
from typing import Optional, Dict
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StatisticalFeatureExtractor:
    """Class to calculate statistical features from cryptocurrency OHLCV data"""
    
    def __init__(self, data_dir: str = 'data/OHLCV'):
        """Initialize the feature extractor
        
        Args:
            data_dir: Directory containing input data files
        """
        self.data_dir = data_dir

    def load_from_pickle(self, filepath: str) -> Dict:
        """Load data from a compressed pickle file"""
        try:
            return pd.read_pickle(filepath)
        except Exception as e:
            logger.error(f"Error loading file {filepath}: {e}")
            return {}

    def save_to_pickle(self, df: pd.DataFrame, filepath: str):
        """Save DataFrame to a compressed pickle file"""
        try:
            df.to_pickle(filepath)
            logger.info(f"Saved processed data to {filepath}")
        except Exception as e:
            logger.error(f"Error saving to {filepath}: {e}")

    def crypto_to_dataframe(self, crypto_data: Dict) -> pd.DataFrame:
        """Convert cryptocurrency data dictionary to DataFrame"""
        return pd.DataFrame(crypto_data)

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate statistical features
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with calculated features
        """
        dfTemp = pd.DataFrame()

        # Simple returns
        dfTemp['returns'] = df['close'].pct_change()
        
        # Log returns
        dfTemp['log_returns'] = np.log(df['close']).diff()
        
        # Price slope (using 14-day window)
        window = 14
        x = np.arange(window)
        dfTemp['price_slope'] = df['close'].rolling(window=window).apply(
            lambda y: np.polyfit(x, y, 1)[0] if len(y) == window else np.nan
        )
        
        # Price acceleration (second derivative)
        dfTemp['price_acceleration'] = dfTemp['returns'].diff()
        
        # Z-score (20-day window)
        window = 20
        dfTemp['z_score'] = (df['close'] - df['close'].rolling(window=window).mean()) / \
                        df['close'].rolling(window=window).std()
        
        # 30-day volatility
        dfTemp['volatility_30'] = dfTemp['returns'].rolling(window=30).std()
        
        # Maximum drawdown
        rolling_max = df['close'].expanding().max()
        drawdown = (df['close'] - rolling_max) / rolling_max
        dfTemp['drawdown_max'] = drawdown.expanding().min()
        dfTemp['timestamp'] = df.index
        dfTemp.set_index('timestamp', inplace=True)
        
        return dfTemp

    def process_crypto_data(self, input_file: str, output_file: Optional[str] = None) -> pd.DataFrame:
        """Process cryptocurrency data by calculating statistical features
        
        Args:
            input_file: Path to the input pickle file
            output_file: Optional path to save the processed data
            
        Returns:
            pd.DataFrame: DataFrame with calculated features
        """
        # Load data
        crypto_data = self.load_from_pickle(input_file)

        # Convert to DataFrame
        #df = self.crypto_to_dataframe(crypto_data)
        
        # Calculate features
        result_df = self.calculate_features(crypto_data)
        result_df = result_df.ffill().bfill()

        # Save processed data if output file is specified
        if output_file:
            self.save_to_pickle(result_df, output_file)
        
        return result_df

    def process_all_files(self):
        """Process all cryptocurrency data files in the specified directory"""
        output_dir = 'data/3_process_statistical_features'
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
                output_file = Path(output_dir) / f"process_statistical_features_{input_file.name}"
                
                logger.info(f"Processing {input_file}")
                self.process_crypto_data(str(input_file), str(output_file))
                
            except Exception as e:
                logger.error(f"Error processing {input_file}: {e}")

# Example usage
if __name__ == "__main__":
    feature_extractor = StatisticalFeatureExtractor()
    feature_extractor.process_all_files()
