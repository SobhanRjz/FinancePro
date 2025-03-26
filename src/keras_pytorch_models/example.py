import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

# Configure TensorFlow for optimal GPU usage
def configure_gpu():
    """Configure GPU settings with compatibility for newer CUDA versions."""
    # Set memory growth to prevent TF from taking all GPU memory
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        try:
            # Allow memory growth to prevent taking all GPU memory
            for device in physical_devices:
                tf.config.experimental.set_memory_growth(device, True)
            
            # Set visible devices if you have multiple GPUs
            tf.config.set_visible_devices(physical_devices[0], 'GPU')
            
            # Disable OneDNN optimizations that can cause issues with newer CUDA
            os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
            
            # Set TF_FORCE_GPU_ALLOW_GROWTH to prevent OOM errors
            os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
            
            print(f"Using GPU: {physical_devices[0].name}")
            return True
        except RuntimeError as e:
            print(f"GPU error: {e}")
    
    print("No GPU found. Using CPU.")
    return False

# Configure logging with more detailed format
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TimeSeriesLSTMModel:
    """
    Enhanced LSTM-based time series forecasting model for financial data.
    Handles data preprocessing, model training, and evaluation with optimized
    performance for GPU acceleration.
    """
    
    def __init__(
        self, 
        data_path, 
        window_size=75, 
        batch_size=64, 
        epochs=50,
        lstm_units=[128, 64, 32],
        dropout_rate=0.3,
        learning_rate=0.001,
        validation_split=0.2,
        use_bidirectional=True,
        forecast_horizon=4  # Number of future time steps to predict
    ):
        """
        Initialize the model with configuration parameters.
        
        Args:
            data_path: Path to the input data file
            window_size: Number of time steps to include in each sequence
            batch_size: Batch size for training
            epochs: Maximum number of training epochs
            lstm_units: List of units for each LSTM layer
            dropout_rate: Dropout rate for regularization
            learning_rate: Initial learning rate for optimizer
            validation_split: Proportion of training data to use for validation
            use_bidirectional: Whether to use bidirectional LSTM layers
            forecast_horizon: Number of future time steps to predict
        """
        self.data_path = data_path
        self.window_size = window_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.validation_split = validation_split
        self.use_bidirectional = use_bidirectional
        self.forecast_horizon = forecast_horizon
        
        # Initialize scalers for features and target
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        
        self.model = None
        self.history = None
        
        # Configure GPU
        self.using_gpu = configure_gpu()
        
        # Create directories for outputs
        self.checkpoint_dir = 'checkpoints/lstm_model'
        self.log_dir = f'logs/lstm_model_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        logger.info(f"Model initialized with window_size={window_size}, batch_size={batch_size}")
        logger.info(f"LSTM units: {lstm_units}, dropout: {dropout_rate}, learning_rate: {learning_rate}")
        logger.info(f"Forecast horizon: {forecast_horizon} time steps")
    
    def load_and_preprocess_data(self, selected_columns=None, target_column='close'):
        """
        Load and preprocess the time series data with efficient vectorized operations.
        
        Args:
            selected_columns: List of columns to use for analysis
            target_column: Column to predict
            
        Returns:
            Processed data ready for sequence creation
        """
        logger.info(f"Loading data from {self.data_path}...")
        
        # Determine file type and load accordingly
        if self.data_path.endswith('.pkl') or self.data_path.endswith('.pkl.gz'):
            data0 = pd.read_pickle(self.data_path)
        elif self.data_path.endswith('.csv') or self.data_path.endswith('.csv.gz'):
            data0 = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"Unsupported file format: {self.data_path}")
        
        # Use default feature set if none provided
        if selected_columns is None:
            selected_columns = ['timeframe', 'open', 'high', 'low', 
                               'volume', 'num_trades', 'vwap', 'AMD GPU', 'Nvidia GPU', 'Nasdaq',
                               'oil_price', 'gold_price', 'silver_price', 'S&P 500', 'sma_20',
                               'sma_50', 'sma_200', 'ema_20', 'ema_50', 'ema_200', 'macd']
        
        # Filter columns that exist in the dataframe
        available_columns = [col for col in selected_columns if col in data0.columns]
        if len(available_columns) < len(selected_columns):
            missing_cols = set(selected_columns) - set(available_columns)
            logger.warning(f"Columns not found in dataset: {missing_cols}")
        
        # Ensure target column exists
        if target_column not in data0.columns:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
        
        # Keep only the selected columns for features
        features = data0[available_columns].copy()
        target = data0[target_column].copy()
        
        # Handle missing values efficiently
        features = features.ffill().bfill()
        target = target.ffill().bfill()
        
        # Drop any remaining rows with NaN values
        valid_indices = ~(features.isna().any(axis=1) | target.isna())
        features = features.loc[valid_indices]
        target = target.loc[valid_indices]
        
        logger.info(f"Selected data shape: {features.shape}")
        logger.info(f"Selected features: {list(features.columns)}")
        
        # Check for non-numeric columns and handle them
        # The error shows we have a string '4h' in the timeframe column that can't be converted to float
        if 'timeframe' in features.columns:
            # Either drop the timeframe column
            features = features.drop(columns=['timeframe'])
            logger.info(f"Dropped 'timeframe' column as it contains non-numeric values")
            # Or alternatively, encode it if it's categorical
            # features['timeframe'] = pd.factorize(features['timeframe'])[0]
        
        # Check for any other non-numeric columns
        non_numeric_cols = features.select_dtypes(exclude=['number']).columns.tolist()
        if non_numeric_cols:
            logger.warning(f"Dropping non-numeric columns: {non_numeric_cols}")
            features = features.drop(columns=non_numeric_cols)
        
        # Normalize data with separate scalers for features and target
        features_normalized = self.feature_scaler.fit_transform(features)
        # Reshape target for scaler
        target_values = target.values.reshape(-1, 1)
        self.target_scaler.fit(target_values)
        
        return features_normalized, target_values
    
    def create_sequences(self, data, target):
        """
        Create sequences for time series prediction using efficient numpy operations.
        Modified to support multi-step forecasting.
        
        Args:
            data: Normalized feature data array
            target: Target values array
            
        Returns:
            X: Input sequences
            y: Target values for multiple future time steps
        """
        # Adjust for multi-step forecasting
        n_samples = len(data) - self.window_size - self.forecast_horizon + 1
        X = np.zeros((n_samples, self.window_size, data.shape[1]))
        y = np.zeros((n_samples, self.forecast_horizon))
        
        for i in range(n_samples):
            # For each window, use all features
            X[i] = data[i:(i + self.window_size), :]
            # Get multiple future target values
            y[i] = target[i + self.window_size:i + self.window_size + self.forecast_horizon, 0]
        
        logger.info(f"Created {n_samples} sequences with shape X: {X.shape}, y: {y.shape}")
        return X, y
    
    def split_data(self, X, y, train_ratio=0.8):
        """
        Split data into training and testing sets with chronological awareness.
        
        Args:
            X: Input sequences
            y: Target values
            train_ratio: Proportion of data to use for training
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        split = int(train_ratio * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def build_model(self, input_shape):
        """
        Build and compile an enhanced LSTM model with bidirectional layers,
        batch normalization, and advanced architecture. Modified for multi-step forecasting.
        
        Args:
            input_shape: Shape of input sequences (window_size, features)
            
        Returns:
            Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        if self.use_bidirectional:
            model.add(Bidirectional(
                LSTM(self.lstm_units[0], return_sequences=True),
                input_shape=input_shape
            ))
        else:
            model.add(LSTM(self.lstm_units[0], return_sequences=True, input_shape=input_shape))
        
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Middle LSTM layers
        for units in self.lstm_units[1:-1]:
            if self.use_bidirectional:
                model.add(Bidirectional(LSTM(units, return_sequences=True)))
            else:
                model.add(LSTM(units, return_sequences=True))
            model.add(BatchNormalization())
            model.add(Dropout(self.dropout_rate))
        
        # Final LSTM layer
        if self.use_bidirectional:
            model.add(Bidirectional(LSTM(self.lstm_units[-1])))
        else:
            model.add(LSTM(self.lstm_units[-1]))
        
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Output layer with multiple outputs for multi-step forecasting
        model.add(Dense(self.forecast_horizon))
        
        # Compile with appropriate optimizer settings
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')
        
        # Print model summary
        model.summary()
        
        return model
    
    def train(self, X_train, y_train):
        """
        Train the LSTM model with advanced callbacks and monitoring.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values
            
        Returns:
            Training history
        """
        # Add callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.checkpoint_dir}/best_lstm_model.h5',
                save_best_only=True,
                monitor='val_loss',
                verbose=1
            ),
            TensorBoard(
                log_dir=self.log_dir,
                histogram_freq=1,
                write_graph=True,
                update_freq='epoch'
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            )
        ]
        
        # Train with validation split
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=callbacks,
            verbose=1,
            shuffle=False  # Important for time series data
        )
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data with comprehensive metrics.
        Modified to handle multi-step forecasting.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values
            
        Returns:
            Dictionary of performance metrics
        """
        # Make predictions
        predicted = self.model.predict(X_test)
        
        # Inverse transform predictions and actual values
        predicted_actual = np.array([self.target_scaler.inverse_transform(pred.reshape(-1, 1)).flatten() 
                                    for pred in predicted])
        y_test_actual = np.array([self.target_scaler.inverse_transform(y.reshape(-1, 1)).flatten() 
                                 for y in y_test])
        
        # Calculate metrics for each forecast horizon
        metrics = {}
        horizon_metrics = []
        
        for h in range(self.forecast_horizon):
            h_pred = predicted_actual[:, h]
            h_actual = y_test_actual[:, h]
            
            mse = mean_squared_error(h_actual, h_pred)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(h_actual - h_pred))
            r2 = r2_score(h_actual, h_pred)
            mape = mean_absolute_percentage_error(h_actual, h_pred)
            
            # Calculate directional accuracy
            if h > 0:
                prev_actual = y_test_actual[:, h-1]
                direction_actual = h_actual > prev_actual
                direction_pred = h_pred > prev_actual
                directional_accuracy = np.mean(direction_actual == direction_pred) * 100
            else:
                # For the first step, compare with the last value in the input window
                last_actual = np.array([self.target_scaler.inverse_transform(
                    X_test[i, -1, 0].reshape(-1, 1)).flatten()[0] for i in range(len(X_test))])
                direction_actual = h_actual > last_actual
                direction_pred = h_pred > last_actual
                directional_accuracy = np.mean(direction_actual == direction_pred) * 100
            
            horizon_metrics.append({
                'horizon': h+1,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'mape': mape,
                'directional_accuracy': directional_accuracy
            })
            
            logger.info(f"Horizon {h+1} metrics:")
            logger.info(f"  RMSE: {rmse:.4f}")
            logger.info(f"  R²: {r2:.4f}")
            logger.info(f"  MAPE: {mape:.4f}")
            logger.info(f"  Directional Accuracy: {directional_accuracy:.2f}%")
        
        # Calculate average metrics across all horizons
        avg_metrics = {
            'avg_rmse': np.mean([m['rmse'] for m in horizon_metrics]),
            'avg_r2': np.mean([m['r2'] for m in horizon_metrics]),
            'avg_mape': np.mean([m['mape'] for m in horizon_metrics]),
            'avg_directional_accuracy': np.mean([m['directional_accuracy'] for m in horizon_metrics])
        }
        
        logger.info(f"Average metrics across all horizons:")
        logger.info(f"  Avg RMSE: {avg_metrics['avg_rmse']:.4f}")
        logger.info(f"  Avg R²: {avg_metrics['avg_r2']:.4f}")
        logger.info(f"  Avg MAPE: {avg_metrics['avg_mape']:.4f}")
        logger.info(f"  Avg Directional Accuracy: {avg_metrics['avg_directional_accuracy']:.2f}%")
        
        # Generate and save plots
        self._save_plots(predicted_actual, y_test_actual)
        
        metrics = {
            'horizon_metrics': horizon_metrics,
            'avg_metrics': avg_metrics,
            'predictions': predicted_actual,
            'actual': y_test_actual
        }
        
        return metrics
    
    def calculate_accuracy(self, y_true, y_pred, threshold=0.05):
        """
        Calculate various accuracy metrics for time series forecasting.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            threshold: Threshold for considering a prediction accurate (as percentage)
            
        Returns:
            Dictionary of accuracy metrics
        """
        # Ensure inputs are numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        
        # Calculate percentage error
        percent_error = np.abs((y_true - y_pred) / y_true)
        
        # Threshold accuracy (predictions within threshold % of actual)
        threshold_accuracy = np.mean(percent_error <= threshold) * 100
        
        # Directional accuracy (correctly predicting up/down movement)
        if len(y_true.shape) == 2:  # Multi-step forecasting
            direction_accuracy = []
            for h in range(y_true.shape[1]-1):
                actual_direction = y_true[:, h+1] > y_true[:, h]
                pred_direction = y_pred[:, h+1] > y_pred[:, h]
                direction_accuracy.append(np.mean(actual_direction == pred_direction) * 100)
            directional_accuracy = np.mean(direction_accuracy)
        else:  # Single-step forecasting
            # Calculate direction changes between consecutive points
            actual_direction = np.diff(y_true) > 0
            pred_direction = np.diff(y_pred) > 0
            directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        # Extreme movement accuracy (correctly predicting significant movements)
        if len(y_true.shape) == 2:  # Multi-step forecasting
            significant_moves = []
            for h in range(y_true.shape[1]-1):
                actual_pct_change = (y_true[:, h+1] - y_true[:, h]) / y_true[:, h]
                pred_pct_change = (y_pred[:, h+1] - y_pred[:, h]) / y_pred[:, h]
                # Consider movements > 1% as significant
                significant_actual = np.abs(actual_pct_change) > 0.01
                if np.sum(significant_actual) > 0:
                    correct_significant = ((actual_pct_change > 0) == (pred_pct_change > 0))[significant_actual]
                    significant_moves.append(np.mean(correct_significant) * 100)
            extreme_accuracy = np.mean(significant_moves) if significant_moves else 0
        else:
            actual_pct_change = np.diff(y_true) / y_true[:-1]
            pred_pct_change = np.diff(y_pred) / y_pred[:-1]
            # Consider movements > 1% as significant
            significant_actual = np.abs(actual_pct_change) > 0.01
            if np.sum(significant_actual) > 0:
                correct_significant = ((actual_pct_change > 0) == (pred_pct_change > 0))[significant_actual]
                extreme_accuracy = np.mean(correct_significant) * 100
            else:
                extreme_accuracy = 0
        
        # Profit factor (ratio of gains to losses if trading based on predictions)
        if len(y_true.shape) == 2:  # Multi-step forecasting
            profit_factors = []
            for h in range(y_true.shape[1]-1):
                pred_direction = y_pred[:, h] < y_pred[:, h+1]  # Predicted price increase
                actual_change = y_true[:, h+1] - y_true[:, h]
                
                # Calculate gains and losses based on predicted direction
                gains = np.sum(actual_change[pred_direction]) if np.any(pred_direction) else 0
                losses = -np.sum(actual_change[~pred_direction]) if np.any(~pred_direction) else 0
                
                # Calculate profit factor (avoid division by zero)
                if losses > 0:
                    profit_factors.append(gains / losses)
                elif gains > 0:
                    profit_factors.append(float('inf'))  # No losses but some gains
                else:
                    profit_factors.append(1.0)  # No gains or losses
            
            profit_factor = np.mean([pf for pf in profit_factors if not np.isinf(pf)]) if profit_factors else 1.0
        else:
            pred_direction = np.diff(y_pred) > 0  # Predicted price increase
            actual_change = np.diff(y_true)
            
            # Calculate gains and losses based on predicted direction
            gains = np.sum(actual_change[pred_direction]) if np.any(pred_direction) else 0
            losses = -np.sum(actual_change[~pred_direction]) if np.any(~pred_direction) else 0
            
            # Calculate profit factor (avoid division by zero)
            profit_factor = gains / losses if losses > 0 else float('inf') if gains > 0 else 1.0
        
        return {
            'threshold_accuracy': threshold_accuracy,
            'directional_accuracy': directional_accuracy,
            'extreme_movement_accuracy': extreme_accuracy,
            'profit_factor': profit_factor if not np.isinf(profit_factor) else 999.99
        }
    
    def _save_plots(self, predicted, actual):
        """
        Generate and save comprehensive evaluation plots for multi-step forecasting.
        
        Args:
            predicted: Predicted values
            actual: Actual values
        """
        try:
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(15, 15))
            
            # Plot 1: Training and validation loss
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(self.history.history['loss'], label='Training Loss')
            ax1.plot(self.history.history['val_loss'], label='Validation Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Actual vs predicted values for each horizon
            ax2 = fig.add_subplot(3, 1, 2)
            colors = ['blue', 'red', 'green', 'purple', 'orange', 'brown', 'pink', 'gray']
            
            # Plot a subset of the test data for clarity
            sample_size = min(100, len(actual))
            x_range = np.arange(sample_size)
            
            for h in range(min(self.forecast_horizon, 4)):  # Plot up to 4 horizons for clarity
                ax2.plot(x_range, actual[:sample_size, h], 
                         label=f'Actual (t+{h+1})', 
                         color=colors[h], 
                         linestyle='-')
                ax2.plot(x_range, predicted[:sample_size, h], 
                         label=f'Predicted (t+{h+1})', 
                         color=colors[h], 
                         linestyle='--')
            
            ax2.set_title('Actual vs Predicted Values (Sample)')
            ax2.set_xlabel('Sample Index')
            ax2.set_ylabel('Value')
            ax2.legend()
            ax2.grid(True)
            
            # Plot 3: Prediction error for each horizon
            ax3 = fig.add_subplot(3, 1, 3)
            
            for h in range(min(self.forecast_horizon, 4)):  # Plot up to 4 horizons for clarity
                error = actual[:sample_size, h] - predicted[:sample_size, h]
                ax3.plot(x_range, error, label=f'Error (t+{h+1})', color=colors[h])
            
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('Prediction Error by Horizon (Sample)')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Error')
            ax3.legend()
            ax3.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{self.checkpoint_dir}/lstm_results.png', dpi=300)
            plt.close()
            
            # Create additional plot for error distribution by horizon
            plt.figure(figsize=(12, 8))
            
            for h in range(min(self.forecast_horizon, 4)):  # Plot up to 4 horizons for clarity
                error = actual[:, h] - predicted[:, h]
                plt.hist(error, bins=50, alpha=0.5, label=f'Horizon {h+1}', color=colors[h])
            
            plt.title('Error Distribution by Forecast Horizon')
            plt.xlabel('Prediction Error')
            plt.ylabel('Frequency')
            plt.legend()
            plt.grid(True)
            plt.savefig(f'{self.checkpoint_dir}/error_distribution.png', dpi=300)
            plt.close()
            
            logger.info(f"Plots saved to {self.checkpoint_dir}")
        except Exception as e:
            logger.error(f"Error generating plots: {str(e)}")
    
    def run_pipeline(self):
        """
        Execute the complete modeling pipeline from data loading to evaluation.
        
        Returns:
            Evaluation metrics
        """
        # Load and preprocess data
        data_normalized, target = self.load_and_preprocess_data()
        
        # Create sequences
        X, y = self.create_sequences(data_normalized, target)
        logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = self.split_data(X, y)
        
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Train model
        self.train(X_train, y_train)
        
        # Evaluate model
        metrics = self.evaluate(X_test, y_test)
        
        # Calculate additional accuracy metrics
        accuracy_metrics = self.calculate_accuracy(metrics['actual'], metrics['predictions'])
        metrics['accuracy_metrics'] = accuracy_metrics
        
        logger.info("===== Accuracy Metrics =====")
        logger.info(f"Threshold Accuracy (within 5%): {accuracy_metrics['threshold_accuracy']:.2f}%")
        logger.info(f"Directional Accuracy: {accuracy_metrics['directional_accuracy']:.2f}%")
        logger.info(f"Extreme Movement Accuracy: {accuracy_metrics['extreme_movement_accuracy']:.2f}%")
        logger.info(f"Profit Factor: {accuracy_metrics['profit_factor']:.2f}")
        
        # Save the final model
        self.model.save(f'{self.checkpoint_dir}/final_lstm_model.h5')
        logger.info(f"Final model saved to {self.checkpoint_dir}/final_lstm_model.h5")
        
        return metrics


# Execute the pipeline
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    # Initialize and run the model with enhanced parameters
    lstm_pipeline = TimeSeriesLSTMModel(
        data_path='data/merged_features/merged_BTCUSDT_4h_5y.pkl.gz',
        window_size=96,  # Increased window size for better context
        batch_size=32,   # Smaller batch size for better generalization
        epochs=100,      # More epochs with early stopping
        lstm_units=[128, 64, 32],  # LSTM layer sizes
        dropout_rate=0.3,          # Regularization
        learning_rate=0.001,       # Initial learning rate
        validation_split=0.2,      # Validation data proportion
        use_bidirectional=True,    # Use bidirectional LSTM for better performance
        forecast_horizon=4         # Predict 4 future time steps
    )
    
    # Run the full pipeline
    metrics = lstm_pipeline.run_pipeline()
    
    # Print final accuracy metrics
    logger.info("===== Final Model Performance =====")
    logger.info(f"Average R² Score: {metrics['avg_metrics']['avg_r2']:.4f}")
    logger.info(f"Average RMSE: {metrics['avg_metrics']['avg_rmse']:.4f}")
    logger.info(f"Average MAPE: {metrics['avg_metrics']['avg_mape']:.4f}%")
    logger.info(f"Average Directional Accuracy: {metrics['avg_metrics']['avg_directional_accuracy']:.2f}%")
    logger.info(f"Threshold Accuracy: {metrics['accuracy_metrics']['threshold_accuracy']:.2f}%")
    logger.info(f"Profit Factor: {metrics['accuracy_metrics']['profit_factor']:.2f}")