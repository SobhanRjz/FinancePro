import pandas as pd
import numpy as np
import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional
from sklearn.preprocessing import RobustScaler, StandardScaler
import pickle
import gzip

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class XGBTimeSeriesPreprocessor:
    """
    Preprocessor for XGBoost time series prediction.
    Handles data loading, cleaning, feature engineering, and sequence creation.
    """
    
    def __init__(
        self,
        data_path: str,
        target_col: str = 'close',
        window_size: int = 96,
        future_steps: int = 4,
        threshold: float = 0.001,
        feature_selection: Optional[List[str]] = None,
        scaler_type: str = 'robust',
        timeframe: str = '4h'
    ):
        """
        Initialize the preprocessor.
        
        Args:
            data_path: Path to the data file
            target_col: Target column to predict
            window_size: Number of time steps to include in each window
            future_steps: Number of future steps to predict
            threshold: Threshold for classification
            feature_selection: List of features to use (None for all)
            scaler_type: Type of scaler to use ('robust' or 'standard')
            timeframe: Timeframe of the data
        """
        self.data_path = data_path
        self.target_col = target_col
        self.window_size = window_size
        self.future_steps = future_steps
        self.threshold = threshold
        self.feature_selection = feature_selection
        self.scaler_type = scaler_type
        self.timeframe = timeframe
        
        # Load data
        self.df = self._load_data()
        
        # Set features
        if feature_selection is not None:
            self.features = [f for f in feature_selection if f in self.df.columns]
            logger.info(f"Using selected features: {self.features}")
        else:
            # Exclude the target column and any date/time columns
            exclude_cols = [target_col, 'date', 'timestamp', 'time', 'datetime']
            self.features = [col for col in self.df.columns if col not in exclude_cols]
            logger.info(f"Using all available features: {len(self.features)} features")
        
        # Ensure we have both the target column and features
        columns_to_keep = self.features.copy()
        if self.target_col not in columns_to_keep:
            columns_to_keep.append(self.target_col)
        
        # Select only the required columns
        self.df = self.df[columns_to_keep]
        logger.info(f"Selected {len(columns_to_keep)} columns: target '{self.target_col}' and {len(self.features)} features")
        # Initialize scalers dictionary
        self.scalers = {}
        
        logger.info(f"Initialized preprocessor with {len(self.features)} features, "
                   f"window_size={window_size}, future_steps={future_steps}, threshold={threshold}")
    
    def _load_data(self) -> pd.DataFrame:
        """Load data from file."""
        try:
            logger.info(f"Loading data from {self.data_path}")
            
            if self.data_path.endswith('.pkl.gz'):
                with gzip.open(self.data_path, 'rb') as f:
                    df = pickle.load(f)
            elif self.data_path.endswith('.pkl'):
                df = pd.read_pickle(self.data_path)
            elif self.data_path.endswith('.csv.gz'):
                df = pd.read_csv(self.data_path, compression='gzip')
            elif self.data_path.endswith('.csv'):
                df = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
            
            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self) -> None:
        """Clean the data by handling missing values and outliers."""
        # Record initial shape
        initial_shape = self.df.shape
        
        # Forward fill missing values
        self.df = self.df.ffill()
        
        # Backward fill any remaining missing values
        self.df = self.df.bfill()
        
        # Log the results
        logger.info(f"Data cleaning: {initial_shape} -> {self.df.shape}")
    
    def add_technical_indicators(self) -> None:
        """Add technical indicators as features."""
        # Only proceed if we have the necessary price columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in self.df.columns for col in required_cols):
            logger.warning("Not all required columns for technical indicators are available")
            return
        
        # Add Moving Averages
        for period in [5, 10, 20, 50, 100]:
            self.df[f'sma_{period}'] = self.df['close'].rolling(window=period).mean()
            self.df[f'ema_{period}'] = self.df['close'].ewm(span=period, adjust=False).mean()
        
        # Add MACD
        self.df['macd'] = self.df['close'].ewm(span=12, adjust=False).mean() - \
                          self.df['close'].ewm(span=26, adjust=False).mean()
        self.df['macd_signal'] = self.df['macd'].ewm(span=9, adjust=False).mean()
        self.df['macd_hist'] = self.df['macd'] - self.df['macd_signal']
        
        # Add RSI
        delta = self.df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.df['rsi'] = 100 - (100 / (1 + rs))
        
        # Add Bollinger Bands
        for period in [20]:
            self.df[f'bb_middle_{period}'] = self.df['close'].rolling(window=period).mean()
            std = self.df['close'].rolling(window=period).std()
            self.df[f'bb_upper_{period}'] = self.df[f'bb_middle_{period}'] + 2 * std
            self.df[f'bb_lower_{period}'] = self.df[f'bb_middle_{period}'] - 2 * std
        
        # Add ATR (Average True Range)
        high_low = self.df['high'] - self.df['low']
        high_close = (self.df['high'] - self.df['close'].shift()).abs()
        low_close = (self.df['low'] - self.df['close'].shift()).abs()
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        self.df['atr_14'] = true_range.rolling(14).mean()
        
        # Add lag features
        for lag in [1, 2, 3, 5, 10]:
            self.df[f'close_lag_{lag}'] = self.df['close'].shift(lag)
            self.df[f'return_lag_{lag}'] = self.df['close'].pct_change(lag)
        
        # Add rolling statistics
        for window in [5, 10, 20]:
            self.df[f'volatility_{window}'] = self.df['close'].pct_change().rolling(window).std()
            self.df[f'return_{window}'] = self.df['close'].pct_change(window)
            self.df[f'volume_ma_{window}'] = self.df['volume'].rolling(window).mean()
        
        # Drop NaN values created by indicators
        self.df = self.df.dropna()
        
        # Update features list
        self.features = [col for col in self.df.columns if col != self.target_col]
        
        logger.info(f"Added technical indicators. New shape: {self.df.shape}")
    
    def normalize_features(self) -> None:
        """Normalize features using the specified scaler type."""
        for feature in self.features:
            try:
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
    
    def create_xgb_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create features for XGBoost by flattening the time windows.
        
        Returns:
            X: Feature matrix with shape (n_samples, window_size * n_features)
            y: Target matrix with shape (n_samples, future_steps)
        """
        logger.info(f"Creating XGBoost features from {len(self.df) - self.window_size - self.future_steps} valid samples")
        
        # Initialize arrays
        X = []
        y = []
        
        # Generate sequences
        for i in range(len(self.df) - self.window_size - self.future_steps):
            # Extract window
            window = self.df.iloc[i:i+self.window_size][self.features].values
            
            # Flatten the window for XGBoost (from 2D to 1D)
            X.append(window.flatten())
            
            # Extract future targets
            future_targets = []
            for step in range(1, self.future_steps + 1):
                current_price = self.df.iloc[i+self.window_size-1][self.target_col]
                future_price = self.df.iloc[i+self.window_size+step-1][self.target_col]
                
                # Calculate percent change
                pct_change = (future_price - current_price) / current_price
                
                # Convert to class based on threshold
                if pct_change > self.threshold:
                    target_class = 2  # Price increased
                elif pct_change < -self.threshold:
                    target_class = 0  # Price decreased
                else:
                    target_class = 1  # Price stable
                
                future_targets.append(target_class)
            
            y.append(future_targets)
        
        # Convert to numpy arrays
        X = np.array(X)
        y = np.array(y)
        
        # Log shapes and class distribution
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Log class distribution for each prediction step
        for step in range(self.future_steps):
            classes, counts = np.unique(y[:, step], return_counts=True)
            class_distribution = {int(classes[i]): int(counts[i]) for i in range(len(classes))}
            logger.info(f"Class distribution (t+{step+1}): {class_distribution}")
        
        return X, y
    
    def train_val_test_split(self, X: np.ndarray, y: np.ndarray, 
                            val_size: float = 0.15, 
                            test_size: float = 0.15) -> Dict[str, np.ndarray]:
        """
        Split the data into training, validation and test sets, respecting time order.
        
        Args:
            X: Feature matrix
            y: Target matrix
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
        feature_names = []
        for i in range(self.window_size):
            for feature in self.features:
                feature_names.append(f"{feature}_t-{self.window_size-i}")
        
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
        #self.add_technical_indicators()
        self.normalize_features()
        X, y = self.create_xgb_features()
        
        # Split the data
        data_splits = self.train_val_test_split(X, y)
        
        logger.info("Preprocessing complete")
        return data_splits 