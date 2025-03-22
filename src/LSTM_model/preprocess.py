import pandas as pd
import numpy as np
import os
import gzip
import pickle
from sklearn.preprocessing import StandardScaler, RobustScaler
from typing import List, Tuple, Dict, Union, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class BitcoinTradingPreprocessor:
    def __init__(
        self, 
        data_path: str = 'data/merged_features/merged_BTCUSDT_4h_5y.pkl.gz', 
        target_col: str = 'close', 
        window_size: int = 72,  # 4h * 72 = 12 days of data (recommended 48-96 for 4h timeframe)
        future_offset: int = 1,  # Predict next 1 steps (8 hours ahead for 4h data)
        threshold: float = 0.005,  # 0.5% price movement threshold
        feature_selection: Optional[List[str]] = None,
        scaler_type: str = 'robust',
        timeframe: str = '4h'
    ):
        """
        Advanced preprocessor for Bitcoin trading data with LSTM models.
        
        Recommended settings by timeframe:
        - 10y daily: window_size=30-60, future_offset=1-3
        - 5y 4h: window_size=48-96, future_offset=1-2
        - 3y 1h: window_size=72-120, future_offset=1-2
        - 1y 5min: window_size=288-720, future_offset=1-6

        Args:
            data_path: Path to the compressed pickle file containing feature data
            target_col: Column name for the price target (typically 'close')
            window_size: Number of past timesteps to include in each sequence
            future_offset: How many timesteps ahead to predict
            threshold: Percentage change threshold to classify as buy/sell
            feature_selection: Optional list of specific features to use
            scaler_type: Type of scaler to use ('standard' or 'robust')
            timeframe: Data timeframe ('1d', '4h', '1h', '5m', etc.)
        """
        self.data_path = data_path
        self.target_col = target_col
        self.window_size = window_size
        self.future_offset = future_offset
        self.threshold = threshold
        self.feature_selection = feature_selection
        self.timeframe = timeframe
        self.scaler_type = scaler_type
        
        # Initialize scalers dictionary to store a scaler for each feature
        self.scalers = {}
        
        # Load and prepare the dataframe
        self.df = self._load_data()
        self.features = self._select_features()
        
        logger.info(f"Initialized preprocessor with {len(self.features)} features, "
                   f"window_size={window_size}, future_offset={future_offset}, "
                   f"threshold={threshold}")

    def _load_data(self) -> pd.DataFrame:
        """
        Load and validate the dataset from the compressed pickle file.
        
        Returns:
            DataFrame with the loaded data
        """
        try:
            logger.info(f"Loading data from {self.data_path}")
            with gzip.open(self.data_path, 'rb') as f:
                df = pickle.load(f)
            
            # Ensure DataFrame format and datetime index
            if not isinstance(df, pd.DataFrame):
                df = pd.DataFrame(df)
            
            # Ensure index is datetime
            if df.index.dtype != 'datetime64[ns]':
                df.index = pd.to_datetime(df.index)
            
            # Sort by time index
            df = df.sort_index()
            
            # Basic validation
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            
            # Ensure target column exists
            if self.target_col not in df.columns:
                available_cols = [col for col in df.columns if 'close' in col.lower()]
                if available_cols:
                    self.target_col = available_cols[0]
                    logger.warning(f"Target column '{self.target_col}' not found. Using '{self.target_col}' instead.")
                else:
                    raise ValueError(f"Target column '{self.target_col}' not found in dataset")
            
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def _select_features(self) -> List[str]:
        """
        Select features based on configuration or use all available features.
        
        Returns:
            List of feature column names
        """
        if self.feature_selection:
            # Use only specified features that exist in the dataframe
            features = [col for col in self.feature_selection if col in self.df.columns]
            if len(features) < len(self.feature_selection):
                logger.warning(f"Some requested features not found in dataset")
        else:
            # Use all columns except the target
            features = [col for col in self.df.columns if col != self.target_col]
        
        # Remove any problematic columns (all NaN, constant values, etc.)
        features_to_remove = []
        for col in features:
            if self.df[col].isna().all() or self.df[col].nunique() <= 1:
                features_to_remove.append(col)
        
        features = [col for col in features if col not in features_to_remove]
        
        if features_to_remove:
            logger.warning(f"Removed {len(features_to_remove)} problematic features")
        
        return features

    def clean_data(self) -> None:
        """
        Clean the dataset by handling missing values and outliers.
        """
        # Store original shape for logging
        original_shape = self.df.shape
        
        # Forward fill missing values (appropriate for time series)
        self.df = self.df.fillna(method='ffill')
        
        # Backward fill any remaining NaNs at the beginning
        self.df = self.df.fillna(method='bfill')
        
        # If any NaNs still remain, fill with column median
        if self.df.isna().any().any():
            for col in self.df.columns:
                if self.df[col].isna().any():
                    self.df[col] = self.df[col].fillna(self.df[col].median())
        
        # Drop any rows that still have NaN values
        self.df = self.df.dropna()
        
        logger.info(f"Data cleaning: {original_shape} -> {self.df.shape}")

    def normalize_features(self) -> None:
        """
        Normalize features using the specified scaler type.
        Stores a separate scaler for each feature for better handling of different scales.
        """
        for feature in self.features:
            try:
                # Create appropriate scaler based on configuration
                # Check if feature contains string values
                if pd.api.types.is_object_dtype(self.df[feature]) or pd.api.types.is_string_dtype(self.df[feature]):
                    # For categorical features like 'US', 'ASIA', 'EU'
                    from sklearn.preprocessing import OneHotEncoder
                    encoder = OneHotEncoder(sparse_output=False)
                    values = self.df[feature].values.reshape(-1, 1)
                    encoded_values = encoder.fit_transform(values)
                    
                    # Replace the original column with encoded values
                    # For simplicity, we'll use the first encoded column
                    # In a production system, you might want to add all columns
                    self.df[feature] = encoded_values[:, 0]
                    
                    # Store the encoder for potential inverse transformation
                    self.scalers[feature] = encoder
                else:
                    # For numerical features, use standard scalers
                    if self.scaler_type.lower() == 'robust':
                        scaler = RobustScaler()  # More resistant to outliers
                    else:
                        scaler = StandardScaler()
                    
                    # Reshape for sklearn compatibility
                    values = self.df[feature].values.reshape(-1, 1)
                    
                    # Fit and transform
                    self.df[feature] = scaler.fit_transform(values)
                    
                    # Store the scaler for potential inverse transformation
                    self.scalers[feature] = scaler
            except Exception as e:
                logger.error(f"Failed to normalize feature '{feature}': {str(e)}")
                continue
        
        logger.info(f"Normalized {len(self.features)} features using {self.scaler_type} scaling")

    def generate_sequences_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate windowed sequences and corresponding trading signals.
        
        Returns:
            Tuple containing:
                - X: 3D array of shape (samples, window_size, features)
                - y: 1D array of labels (1=Buy, 0=Hold, -1=Sell)
        """
        X, y = [], []
        
        # Calculate price changes for labeling
        price_series = self.df[self.target_col]
        
        # Pre-calculate future returns for efficiency
        future_returns = (price_series.shift(-self.future_offset) - price_series) / price_series
        
        valid_indices = range(self.window_size, len(self.df) - self.future_offset)
        total_samples = len(valid_indices)
        
        logger.info(f"Generating sequences from {total_samples} valid samples")
        
        # Pre-extract feature data for faster access
        feature_data = self.df[self.features].values
        
        # Vectorize label generation
        labels = np.zeros(len(valid_indices), dtype=np.int8)
        X = np.zeros((len(valid_indices), self.window_size, len(self.features)))
        
        for idx, i in enumerate(valid_indices):
            # Extract window using direct numpy indexing
            start_idx = i - self.window_size
            X[idx] = feature_data[start_idx:i]
            
            # Get pre-calculated return
            price_return = future_returns.iloc[i]
            
            # Apply threshold logic for classification
            if price_return > self.threshold:
                labels[idx] = 2  # Buy signal (class index 2)
            elif price_return < -self.threshold:
                labels[idx] = 0  # Sell signal (class index 0)
            else:
                labels[idx] = 1  # Hold signal (class index 1)
        
        # Replace list operations with pre-allocated arrays
        X_array, y_array = X, labels
        
        # Log class distribution
        unique, counts = np.unique(y_array, return_counts=True)
        distribution = dict(zip(unique, counts))
        logger.info(f"Class distribution: {distribution}")
        
        return X_array, y_array

    def train_val_test_split(self, X: np.ndarray, y: np.ndarray, 
                            val_size: float = 0.15, 
                            test_size: float = 0.15) -> Dict[str, np.ndarray]:
        """
        Split the data into training, validation and test sets, respecting time order.
        
        Args:
            X: Feature sequences
            y: Target labels
            val_size: Proportion for validation
            test_size: Proportion for testing
            
        Returns:
            Dictionary containing X_train, y_train, X_val, y_val, X_test, y_test
        """
        n_samples = len(X)
        
        # Calculate split indices
        test_idx = int(n_samples * (1 - test_size))
        val_idx = int(n_samples * (1 - test_size - val_size))
        
        # Split the data
        X_train, y_train = X[:val_idx], y[:val_idx]
        X_val, y_val = X[val_idx:test_idx], y[val_idx:test_idx]
        X_test, y_test = X[test_idx:], y[test_idx:]
        
        logger.info(f"Data split: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        # Store feature names for reference
        feature_names = self.features
        
        return {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val, 'y_val': y_val,
            'X_test': X_test, 'y_test': y_test,
            'feature_names': feature_names
        }

    def preprocess(self) -> Dict[str, np.ndarray]:
        """
        Execute the full preprocessing pipeline.
        
        Returns:
            Dictionary with train/val/test splits ready for model training
        """
        logger.info("Starting preprocessing pipeline")
        
        # Execute preprocessing steps
        self.clean_data()
        self.normalize_features()
        X, y = self.generate_sequences_and_labels()
        
        # Split the data
        data_splits = self.train_val_test_split(X, y)
        
        logger.info("Preprocessing complete")
        return data_splits

if __name__ == "__main__":
    preprocessor = BitcoinTradingPreprocessor()
    data_splits = preprocessor.preprocess()
    print(data_splits)

