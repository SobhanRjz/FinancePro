import os
import pandas as pd
import numpy as np
from bidask import edge, edge_rolling, edge_expanding
import logging
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class OrderBookFeatureExtractor:
    def __init__(self, data_dir='data/OHLCV'):
        """
        Initialize the OrderBookFeatureExtractor.
        
        Args:
            data_dir (str): Directory where data files are stored
        """
        self.data_dir = data_dir
        
    def load_order_book_data(self, file_path):
        """
        Load order book data from pickle file.
        
        Args:
            file_path (str): Path to the pickle file
            
        Returns:
            pd.DataFrame: Loaded order book data
        """
        try:
            data = pd.read_pickle(file_path)
            logging.info(f"Successfully loaded order book data from {file_path}")
            return data
        except Exception as e:
            logging.error(f"Error loading order book data: {e}")
            return None
    
    def calculate_bid_ask_spread(self, data):
        """
        Calculate the bid-ask spread using the bidask library.
        
        Args:
            data (pd.DataFrame): Order book data with OHLC columns
            
        Returns:
            pd.Series: Bid-ask spread values
        """
        try:
            # Check if data has required OHLC columns
            required_cols = ['open', 'high', 'low', 'close']
            if all(col.lower() in map(str.lower, data.columns) for col in required_cols):
                # Calculate spread using edge function
                spread = edge(data['open'], data['high'], data['low'], data['close'])
                
                # Calculate rolling spread (30-period window)
                rolling_spread = edge_rolling(data, window=30)
                
                # Calculate expanding spread
                expanding_spread = edge_expanding(data, min_periods=10)
                
                result = pd.DataFrame({
                    #'spread': spread,
                    'rolling_spread_30': rolling_spread,
                    'expanding_spread': expanding_spread
                })
                
                logging.info("Successfully calculated bid-ask spread metrics")
                return result
            elif 'bid' in data.columns and 'ask' in data.columns:
                # If we have direct bid/ask data
                spread = data['ask'] - data['bid']
                spread_pct = spread / ((data['ask'] + data['bid']) / 2)
                
                logging.info("Calculated spread from direct bid/ask data")
                return pd.DataFrame({'spread': spread_pct})
            else:
                logging.error("Data does not contain required OHLC or bid/ask columns")
                return None
        except Exception as e:
            logging.error(f"Error calculating bid-ask spread: {e}")
            return None
    
    def process_all_files(self, time_period=None):
        """
        Extract order book features for the specified time period.
        
        Args:
            time_period (str, optional): Time period to extract features for
            
        Returns:
            pd.DataFrame: DataFrame containing extracted features
        """
        import glob
        # Get all order book data files
        import os
        
        if time_period:
            file_pattern = os.path.join(self.data_dir, f"{time_period}*.pkl.gz")
        else:
            file_pattern = os.path.join(self.data_dir, "*.pkl.gz")
        
        file_paths = glob.glob(file_pattern)
        
        if not file_paths:
            logging.warning(f"No order book files found matching pattern: {file_pattern}")
            return None
        

        for file_path in file_paths:
            logging.info(f"Processing order book file: {file_path}")
            
            # Load the data
            datacopy = self.load_order_book_data(file_path)
            datacopy = pd.DataFrame(datacopy)
            if datacopy is None:
                logging.warning(f"Skipping file {file_path} due to loading error")
                continue
            data = pd.DataFrame()
            # Calculate bid-ask spread
            spread_features = self.calculate_bid_ask_spread(datacopy)
            data["rolling_spread_30"] = spread_features["rolling_spread_30"]
            data["expanding_spread"] = spread_features["expanding_spread"]
            data["timestamp"] = datacopy["timestamp"]
            data = data.set_index("timestamp")
            
            data = data.ffill().bfill()
            if spread_features is None:
                logging.warning(f"Skipping file {file_path} due to feature calculation error")
                continue
                
            # Add file identifier
            file_name = os.path.basename(file_path)
            # Create output directory if it doesn't exist
            output_dir = os.path.join(self.data_dir, 'process_order_book_features')
            os.makedirs(output_dir, exist_ok=True)
            
            # Create output file path with same name but in the new directory
            output_file = os.path.join(self.data_dir.split('/')[0], 'process_order_book_features', 'order_book_features_' + '_'.join(file_name.split('_')[-3:]))
            
            # Save the processed data
            try:
                data.to_pickle(output_file)
                logging.info(f"Saved order book features to: {output_file}")
            except Exception as e:
                logging.error(f"Error saving order book features: {e}")
        

# Example usage
if __name__ == "__main__":
    extractor = OrderBookFeatureExtractor()
    features = extractor.process_all_files()
    if features is not None:
        print(features.head())
