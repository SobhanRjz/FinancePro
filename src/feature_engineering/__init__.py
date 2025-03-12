import os
import logging
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

# Import all feature extractors
module = __import__("feature_engineering.1_core_features", fromlist=["CoreFeatureExtractor"])
CoreFeatureExtractor = getattr(module, "CoreFeatureExtractor")

module = __import__("feature_engineering.2_technical_indicators", fromlist=["TechnicalIndicatorExtractor"])
TechnicalIndicatorExtractor = getattr(module, "TechnicalIndicatorExtractor")

module = __import__("feature_engineering.3_statistical_features", fromlist=["StatisticalFeatureExtractor"])
StatisticalFeatureExtractor = getattr(module, "StatisticalFeatureExtractor")

module = __import__("feature_engineering.4_time_features", fromlist=["TimeFeatureExtractor"])
TimeFeatureExtractor = getattr(module, "TimeFeatureExtractor")

module = __import__("feature_engineering.7_derived_features", fromlist=["DerivedFeatureExtractor"])
DerivedFeatureExtractor = getattr(module, "DerivedFeatureExtractor")

module = __import__("feature_engineering.9_on_chain_features", fromlist=["OnChainFeatureExtractor"])
OnChainFeatureExtractor = getattr(module, "OnChainFeatureExtractor")

module = __import__("feature_engineering.10_risk_features", fromlist=["RiskFeatureExtractor"])
RiskFeatureExtractor = getattr(module, "RiskFeatureExtractor")



# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger("FeatureEngineering")

class FeatureEngineeringOrchestrator:
    def __init__(self, data_dir='data'):
        """
        Initialize the orchestrator that runs all feature engineering processes.
        
        Args:
            data_dir (str): Base directory where data files are stored
        """
        self.data_dir = data_dir
        self.extractors = {
            #'core': CoreFeatureExtractor(os.path.join(data_dir, 'OHLCV')),
            #'order_book': OrderBookFeatureExtractor(os.path.join(data_dir, 'OHLCV')),
            'technical': TechnicalIndicatorExtractor(os.path.join(data_dir, 'OHLCV')),
            #'sentiment': SentimentFeatureExtractor(os.path.join(data_dir, 'OHLCV')),
            'time': TimeFeatureExtractor()
        }
        
    def run_all_extractors(self, time_period=None):
        """
        Run all feature extractors and collect their outputs.
        
        Args:
            time_period (str, optional): Time period to extract features for
            
        Returns:
            dict: Dictionary containing results from each extractor
        """
        results = {}
        
        for name, extractor in self.extractors.items():
            logger.info(f"Running {name} feature extractor...")
            try:
                if hasattr(extractor, 'process_all_files'):
                    results[name] = extractor.process_all_files()
                    logger.info(f"Successfully extracted {name} features")
                else:
                    logger.warning(f"Extractor {name} does not have extract_features method")
            except Exception as e:
                logger.error(f"Error running {name} feature extractor: {e}", exc_info=True)
                results[name] = None
                
        return results
    
    def merge_features(self, features_dict):
        """
        Merge features from different extractors into a single dataset.
        
        Args:
            features_dict (dict): Dictionary containing results from each extractor
            
        Returns:
            pd.DataFrame: Merged features dataframe
        """
        # Filter out None values
        valid_features = {k: v for k, v in features_dict.items() if v is not None}
        
        if not valid_features:
            logger.warning("No valid features to merge")
            return None
        
        # Start with the first dataframe
        first_key = list(valid_features.keys())[0]
        merged_df = valid_features[first_key].copy()
        
        # Merge the rest
        for name, df in list(valid_features.items())[1:]:
            try:
                # Assuming all dataframes have the same index or can be joined on index
                merged_df = merged_df.join(df, how='outer')
                logger.info(f"Merged {name} features")
            except Exception as e:
                logger.error(f"Error merging {name} features: {e}")
        
        return merged_df
    
    def run_pipeline(self, time_period=None, save_output=True):
        """
        Run the complete feature engineering pipeline.
        
        Args:
            time_period (str, optional): Time period to extract features for
            save_output (bool): Whether to save the output to disk
            
        Returns:
            pd.DataFrame: Complete feature set
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Run all extractors
        features = self.run_all_extractors(time_period)
        
        return features

# Export the orchestrator class
__all__ = ['FeatureEngineeringOrchestrator']

if __name__ == "__main__":
    import argparse
    import glob
    import re
    import pickle
    import gzip

    parser = argparse.ArgumentParser(description='Run feature engineering pipeline')
    parser.add_argument('--time_period', type=str, help='Time period to extract features for', default=None)
    parser.add_argument('--data_dir', type=str, help='Base data directory', default='data')
    args = parser.parse_args()
    
    # orchestrator = FeatureEngineeringOrchestrator(data_dir=args.data_dir)
    # features = orchestrator.run_pipeline(time_period=args.time_period)
    
    # Merge features
    # Merge features from different directories
    logger.info("Merging features from different directories...")
    
    # Define directories to search for feature files
    # Sort directories but ensure OHLCV comes first
    feature_dirs = sorted([d for d in os.listdir(args.data_dir) if os.path.isdir(os.path.join(args.data_dir, d))])
    if 'OHLCV' in feature_dirs:
        feature_dirs.remove('OHLCV')
        feature_dirs.insert(0, 'OHLCV')
    # Remove merged_features directory if it exists
    if 'merged_features' in feature_dirs:
        feature_dirs.remove('merged_features')
        logger.info("Removed merged_features directory from processing list")
    
    # Dictionary to store dataframes by symbol/timeframe pattern
    grouped_files = {}
    
    # Iterate through each feature directory
    for feature_dir in feature_dirs:
        dir_path = os.path.join(args.data_dir, feature_dir)
        if not os.path.exists(dir_path):
            logger.warning(f"Directory {dir_path} does not exist, skipping")
            continue
            
        # Find all pickle files in the directory
        file_pattern = os.path.join(dir_path, "*.pkl.gz")
        file_paths = glob.glob(file_pattern)
        
        if not file_paths:
            logger.warning(f"No pickle files found in {dir_path}")
            continue
            
        logger.info(f"Found {len(file_paths)} files in {feature_dir}")
        
        # Process each file
        for file_path in file_paths:
            file_name = os.path.basename(file_path)
            
            # Extract symbol, timeframe and period from filename
            # Look for patterns like BTCUSDT_1d_10y
            match = re.search(r'([A-Z]+)_(\w+)_(\w+)', file_name)
            if match:
                symbol, timeframe, period = match.groups()
                key = f"{symbol}_{timeframe}_{period}"
                
                if key not in grouped_files:
                    grouped_files[key] = []
                
                grouped_files[key].append((feature_dir, file_path))
                logger.info(f"Added {file_name} to group {key}")
            else:
                logger.warning(f"Could not parse pattern from filename: {file_name}")
    
    # Merge files with the same key
    merged_features = {}
    for key, file_list in grouped_files.items():
        logger.info(f"Merging {len(file_list)} files for {key}")
        
        # Start with an empty dataframe
        merged_df = None
        
        for feature_dir, file_path in file_list:
            try:
                # Load the dataframe
                with gzip.open(file_path, 'rb') as f:
                    df = pickle.load(f)
                
                # Convert to DataFrame if it's not already
                if not isinstance(df, pd.DataFrame):
                    df = pd.DataFrame(df).set_index('timestamp')
                
                if df.empty:
                    logger.warning(f"Empty dataframe from {file_path}, skipping")
                    continue


                if 'Sentiment_Features' in file_path:

                    # Sum numeric columns and assign to googletrend for matching dates
                    if merged_df is not None:
                        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
                        if len(numeric_cols) > 0:
                            numeric_sums = df[numeric_cols].sum(axis=1)
                            merged_dates = pd.to_datetime(merged_df.index).date
                            for idx, sum_val in numeric_sums.items():
                                date_match = merged_dates == pd.to_datetime(idx).date()
                                if date_match.any():
                                    for idx in merged_df.index[date_match]:
                                        merged_df.at[idx, 'googletrend'] = sum_val

                # Initialize merged_df if this is the first valid dataframe
                # Ensure index is Timestamp type
                if df.index.dtype != 'datetime64[ns]':
                    df.index = pd.to_datetime(df.index)
                if merged_df is None:
                    merged_df = df.copy()
                    logger.info(f"Initialized merged dataframe with {len(df.columns)} columns from {feature_dir}")
                else:

                    if 'Sentiment_Features' not in file_path:
                        # Merge with existing dataframe
                        new_cols = [col for col in df.columns if col not in merged_df.columns]
                        if new_cols:
                            merged_df = merged_df.join(df[new_cols], how='outer')
                        logger.info(f"Merged {len(df.columns)} columns from {feature_dir}")
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
        
        if merged_df is not None:
            merged_features[key] = merged_df
            
            # Save the merged features
           
            output_dir = os.path.join(args.data_dir, 'merged_features')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, f"merged_{key}.pkl.gz")
            try:
                with gzip.open(output_file, 'wb') as f:
                    pickle.dump(merged_df, f)
                logger.info(f"Saved merged features for {key} to {output_file}")
            except Exception as e:
                logger.error(f"Error saving merged features for {key}: {e}")
    
    # Use the first merged dataframe as the result, or None if no merges were successful
    features = next(iter(merged_features.values())) if merged_features else None
    logger.info(f"Completed feature merging process with {len(merged_features)} merged datasets")
    if features is not None:
        print(f"Generated {len(features.columns)} features for {len(features)} samples")
        print("\nFeature columns:")
        for i, col in enumerate(features.columns, 1):
            print(f"{i}. {col}")