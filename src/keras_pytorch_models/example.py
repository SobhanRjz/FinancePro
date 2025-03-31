import numpy as np
import pandas as pd
import tensorflow as tf
import logging
import os
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

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

class TimeSeriesClassificationModel:
    """
    LSTM-based time series classification model for financial data.
    Predicts Buy, Sell, or Hold signals based on price movements.
    """
    
    def __init__(
        self, 
        data_path, 
        window_size=75, 
        batch_size=32, 
        epochs=50,
        lstm_units=[128, 64, 32],
        dropout_rate=0.5,
        learning_rate=0.001,
        validation_split=0.2,
        use_bidirectional=True,
        threshold=0.004,    # 0.4% price change threshold for classification
        future_candles=4    # Number of candles ahead to predict
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
            threshold: Percentage threshold for classification (0.4%)
            future_candles: Number of candles ahead to predict (default: 4)
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
        self.threshold = threshold
        self.future_candles = future_candles
        
        # Initialize feature scaler with custom range
        self.feature_scaler = MinMaxScaler(feature_range=(-100, 100))
        
        self.model = None
        self.history = None
        
        # Configure GPU
        self.using_gpu = configure_gpu()
        
        # Create directories for outputs
        self.checkpoint_dir = 'checkpoints/lstm_binary_classification'
        self.log_dir = f'logs/lstm_binary_classification_{datetime.now().strftime("%Y%m%d-%H%M%S")}'
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Class mapping (binary)
        self.class_names = ['Sell', 'Buy']
        self.num_classes = len(self.class_names)
        
        logger.info(f"Binary classification model initialized with window_size={window_size}, batch_size={batch_size}")
        logger.info(f"LSTM units: {lstm_units}, dropout: {dropout_rate}, learning_rate: {learning_rate}")
        logger.info(f"Threshold: {threshold*100}%, Predicting {future_candles} candles ahead")
        logger.info(f"Feature scaling range: (-100, 100)")
    
    def load_and_preprocess_data(self, selected_columns=None, target_column='close'):
        """
        Load and preprocess the time series data with scaling.
        
        Args:
            selected_columns: List of columns to use for analysis
            target_column: Column to predict
            
        Returns:
            Processed data ready for sequence creation
        """
        logger.info(f"Loading data from {self.data_path}...")
        
        try:
            # Determine file type and load accordingly
            if self.data_path.endswith('.pkl') or self.data_path.endswith('.pkl.gz'):
                data0 = pd.read_pickle(self.data_path)
            elif self.data_path.endswith('.csv') or self.data_path.endswith('.csv.gz'):
                data0 = pd.read_csv(self.data_path)
            else:
                raise ValueError(f"Unsupported file format: {self.data_path}")
            
            logger.info(f"Data loaded successfully. Shape: {data0.shape}")
            logger.info(f"Columns in dataset: {data0.columns.tolist()}")
            
            # Check for non-numeric columns before filtering
            non_numeric_cols = data0.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_cols:
                logger.warning(f"Non-numeric columns found in dataset: {non_numeric_cols}")
            
            # Use default feature set if none provided
            if selected_columns is None:
                # Exclude known non-numeric columns like 'timeframe' from default selection
                selected_columns = ['returns', 'log_returns', 'price_acceleration', 'stochastic_k', 
                                   'z_score', 'mean_reversion_signal', 'ema_slope', 'above_ema', 
                                   'trend_continuation_probability', 'momentum_score', 'cci_14', 
                                   'rsi_14', 'trend_strength', 'plus_di', 'minus_di', 'stochastic_d', 
                                   'macd_histogram', 'whale_tx_count', 'oil_price', 'price_slope', 
                                   'inflation_rate_yoy', 'crypto', 'hour_of_day', 'whale_btc_volume', 
                                   'macd', 'whale_usd_volume', 'inflation_rate_mom']
            
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
            
            # Check for non-numeric columns in selected features and handle them
            non_numeric_feature_cols = features.select_dtypes(exclude=['number']).columns.tolist()
            if non_numeric_feature_cols:
                logger.warning(f"Dropping non-numeric feature columns: {non_numeric_feature_cols}")
                features = features.drop(columns=non_numeric_feature_cols)
            
            # Verify all remaining columns are numeric
            if not features.select_dtypes(exclude=['number']).empty:
                remaining_non_numeric = features.select_dtypes(exclude=['number']).columns.tolist()
                raise ValueError(f"Still have non-numeric columns after filtering: {remaining_non_numeric}")
            
            # Scale features to range (-100, 100)
            features_normalized = self.feature_scaler.fit_transform(features)
            logger.info(f"Features scaled to range (-100, 100)")
            
            # Keep the original target values for creating class labels
            target_values = target.values
            
            return features_normalized, target_values
            
        except Exception as e:
            logger.error(f"Error in data preprocessing: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            raise
    
    def create_sequences_with_labels(self, data, target, future_candles=4):
        """
        Create sequences for binary classification using efficient numpy operations.
        Predicts price movement several candles into the future.
        
        Args:
            data: Normalized feature data array
            target: Target values array
            future_candles: Number of candles ahead to predict (default: 4)
            
        Returns:
            X: Input sequences
            y: Classification labels (0=Sell, 1=Buy)
        """
        n_samples = len(data) - self.window_size - future_candles
        X = np.zeros((n_samples, self.window_size, data.shape[1]))
        y_classes = np.zeros(n_samples, dtype=int)
        
        for i in range(n_samples):
            # For each window, use all features up to current time t
            # This ensures no future leakage in the input features
            X[i] = data[i:(i + self.window_size), :]
            
            # Calculate percentage change from current close (t) to future close (t+future_candles)
            current_price = target[i + self.window_size - 1]  # Price at time t (end of current window)
            future_price = target[i + self.window_size + future_candles - 1]  # Price at time t+future_candles
            pct_change = (future_price - current_price) / current_price
            
            # Binary classification with threshold
            # Buy (1) if price increases beyond threshold, otherwise Sell (0)
            if pct_change >= self.threshold:
                y_classes[i] = 1  # Buy
            else:
                y_classes[i] = 0  # Sell
        
        # Convert to one-hot encoding
        y_onehot = to_categorical(y_classes, num_classes=self.num_classes)
        
        # Log class distribution
        class_counts = np.bincount(y_classes, minlength=self.num_classes)
        class_percentages = class_counts / len(y_classes) * 100
        
        logger.info(f"Created {n_samples} sequences with shape X: {X.shape}, y: {y_onehot.shape}")
        logger.info(f"Class distribution: Sell: {class_percentages[0]:.1f}%, Buy: {class_percentages[1]:.1f}%")
        logger.info(f"Using threshold of {self.threshold*100:.1f}% for classification")
        logger.info(f"Predicting {future_candles} candles ahead")
        
        return X, y_onehot, y_classes
    
    def split_data(self, X, y, y_classes, train_ratio=0.8):
        """
        Split data into training and testing sets with chronological awareness.
        
        Args:
            X: Input sequences
            y: One-hot encoded target values
            y_classes: Class indices for evaluation
            train_ratio: Proportion of data to use for training
            
        Returns:
            X_train, X_test, y_train, y_test, y_classes_test
        """
        split = int(train_ratio * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        y_classes_test = y_classes[split:]
        
        logger.info(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
        
        return X_train, X_test, y_train, y_test, y_classes_test
    
    def build_model(self, input_shape):
        """
        Build and compile an LSTM classification model with additional layers.
        
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
        
        # Middle LSTM layers - add more layers here
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
        
        # # Add dense layers for better feature extraction
        # model.add(Dense(64, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(self.dropout_rate/2))
        
        # model.add(Dense(32, activation='relu'))
        # model.add(BatchNormalization())
        # model.add(Dropout(self.dropout_rate/2))
        
        # Output layer with 2 classes (Sell, Buy)
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile with appropriate optimizer settings for classification
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer, 
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        model.summary()
        
        return model
    
    def train(self, X_train, y_train):
        """
        Train the LSTM classification model with advanced callbacks and monitoring.
        
        Args:
            X_train: Training input sequences
            y_train: Training target values (one-hot encoded)
            
        Returns:
            Training history
        """
        # Add callbacks for better training
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'{self.checkpoint_dir}/best_classification_model.h5',
                save_best_only=True,
                monitor='val_accuracy',
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
        
        # Train with validation split (no class weights)
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
    
    def evaluate(self, X_test, y_test, y_classes_test):
        """
        Evaluate the classification model on test data with comprehensive metrics.
        
        Args:
            X_test: Test input sequences
            y_test: Test target values (one-hot encoded)
            y_classes_test: Test target values (class indices)
            
        Returns:
            Dictionary of performance metrics
        """
        import json
        # Make predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred_classes = np.argmax(y_pred_proba, axis=1)
        
        # Save predictions and test data to JSON
        prediction_data = {
            'X_test': X_test.tolist(),
            'y_classes_test': y_classes_test.tolist(),
            'y_pred_proba': y_pred_proba.tolist(),
            'y_pred_classes': y_pred_classes.tolist()
        }
        
        try:
            with open(f'{self.checkpoint_dir}/prediction_data.json', 'w') as f:
                json.dump(prediction_data, f)
            logger.info(f"Prediction data saved to {self.checkpoint_dir}/prediction_data.json")
        except Exception as e:
            logger.error(f"Error saving prediction data to JSON: {str(e)}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_classes_test, y_pred_classes)
        conf_matrix = confusion_matrix(y_classes_test, y_pred_classes)
        class_report = classification_report(y_classes_test, y_pred_classes, 
                                            target_names=self.class_names, 
                                            output_dict=True)
        
        # Log results
        logger.info(f"Classification Accuracy: {accuracy:.4f}")
        logger.info(f"Confusion Matrix:\n{conf_matrix}")
        logger.info("Classification Report:")
        for cls in self.class_names:
            logger.info(f"  {cls}: Precision={class_report[cls]['precision']:.4f}, "
                       f"Recall={class_report[cls]['recall']:.4f}, "
                       f"F1-Score={class_report[cls]['f1-score']:.4f}")
        
        # Generate and save plots
        self._save_plots(y_classes_test, y_pred_classes, y_pred_proba)
        
        metrics = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred_classes,
            'prediction_probabilities': y_pred_proba,
            'actual': y_classes_test
        }
        
        return metrics
    
    def calculate_trading_metrics(self, y_true, y_pred):
        """
        Calculate trading-specific metrics for the classification model.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            
        Returns:
            Dictionary of trading metrics
        """
        # Count correct predictions by class
        correct_buys = np.sum((y_true == 1) & (y_pred == 1))
        correct_sells = np.sum((y_true == 0) & (y_pred == 0))
        
        # Count total predictions by class
        total_buy_signals = np.sum(y_pred == 1)
        total_sell_signals = np.sum(y_pred == 0)
        
        # Count actual occurrences by class
        actual_buys = np.sum(y_true == 1)
        actual_sells = np.sum(y_true == 0)
        
        # Calculate precision for buy/sell signals
        buy_precision = correct_buys / total_buy_signals if total_buy_signals > 0 else 0
        sell_precision = correct_sells / total_sell_signals if total_sell_signals > 0 else 0
        
        # Calculate recall for buy/sell signals
        buy_recall = correct_buys / actual_buys if actual_buys > 0 else 0
        sell_recall = correct_sells / actual_sells if actual_sells > 0 else 0
        
        # Calculate potential profit/loss (simplified)
        # Assuming 1% gain on correct buy, 1% loss on incorrect buy
        # Assuming 1% gain on correct sell, 1% loss on incorrect sell
        correct_buy_profit = correct_buys * 0.01
        incorrect_buy_loss = (total_buy_signals - correct_buys) * 0.01
        correct_sell_profit = correct_sells * 0.01
        incorrect_sell_loss = (total_sell_signals - correct_sells) * 0.01
        
        total_profit = correct_buy_profit + correct_sell_profit
        total_loss = incorrect_buy_loss + incorrect_sell_loss
        
        # Calculate profit factor
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Calculate win rate
        total_trades = total_buy_signals + total_sell_signals
        winning_trades = correct_buys + correct_sells
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        return {
            'buy_precision': buy_precision,
            'sell_precision': sell_precision,
            'buy_recall': buy_recall,
            'sell_recall': sell_recall,
            'profit_factor': profit_factor if not np.isinf(profit_factor) else 999.99,
            'win_rate': win_rate,
            'total_trades': total_trades,
            'winning_trades': winning_trades
        }
    
    def _save_plots(self, y_true, y_pred, y_pred_proba):
        """
        Generate and save comprehensive evaluation plots for classification.
        
        Args:
            y_true: True class labels
            y_pred: Predicted class labels
            y_pred_proba: Predicted class probabilities
        """
        try:
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(15, 15))
            
            # Plot 1: Training and validation metrics
            ax1 = fig.add_subplot(3, 1, 1)
            ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
            ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
            ax1.set_title('Training and Validation Accuracy')
            ax1.set_xlabel('Epochs')
            ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax1.grid(True)
            
            # Plot 2: Confusion matrix as heatmap
            ax2 = fig.add_subplot(3, 1, 2)
            conf_matrix = confusion_matrix(y_true, y_pred)
            im = ax2.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
            ax2.set_title('Confusion Matrix')
            
            # Add labels and values to confusion matrix
            tick_marks = np.arange(len(self.class_names))
            ax2.set_xticks(tick_marks)
            ax2.set_yticks(tick_marks)
            ax2.set_xticklabels(self.class_names)
            ax2.set_yticklabels(self.class_names)
            ax2.set_xlabel('Predicted Label')
            ax2.set_ylabel('True Label')
            
            # Add text annotations to confusion matrix
            thresh = conf_matrix.max() / 2.
            for i in range(conf_matrix.shape[0]):
                for j in range(conf_matrix.shape[1]):
                    ax2.text(j, i, format(conf_matrix[i, j], 'd'),
                            ha="center", va="center",
                            color="white" if conf_matrix[i, j] > thresh else "black")
            
            plt.colorbar(im, ax=ax2)
            
            # Plot 3: Class prediction probabilities over time
            ax3 = fig.add_subplot(3, 1, 3)
            
            # Plot a subset of the test data for clarity
            sample_size = min(100, len(y_true))
            x_range = np.arange(sample_size)
            
            ax3.plot(x_range, y_pred_proba[:sample_size, 0], 'r-', label='Sell Probability')
            ax3.plot(x_range, y_pred_proba[:sample_size, 1], 'b-', label='Buy Probability')
            
            # Add true class as background color
            for i in range(sample_size):
                if y_true[i] == 0:  # Sell
                    ax3.axvspan(i-0.5, i+0.5, alpha=0.2, color='red')
                elif y_true[i] == 1:  # Buy
                    ax3.axvspan(i-0.5, i+0.5, alpha=0.2, color='green')
            
            ax3.set_title('Prediction Probabilities (Sample)')
            ax3.set_xlabel('Sample Index')
            ax3.set_ylabel('Probability')
            ax3.legend()
            ax3.grid(True)
            
            plt.tight_layout()
            plt.savefig(f'{self.checkpoint_dir}/classification_results.png', dpi=300)
            plt.close()
            
            # Create additional plot for class distribution
            plt.figure(figsize=(10, 6))
            class_counts = np.bincount(y_true, minlength=self.num_classes)
            plt.bar(self.class_names, class_counts, color=['red', 'green'])
            plt.title('Class Distribution in Test Set')
            plt.xlabel('Class')
            plt.ylabel('Count')
            for i, count in enumerate(class_counts):
                plt.text(i, count + 5, str(count), ha='center')
            plt.savefig(f'{self.checkpoint_dir}/class_distribution.png', dpi=300)
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
        try:
            # Load and preprocess data
            data_normalized, target = self.load_and_preprocess_data()
            
            # Create sequences with classification labels
            X, y, y_classes = self.create_sequences_with_labels(
                data_normalized, 
                target, 
                future_candles=self.future_candles
            )
            logger.info(f"X shape: {X.shape}, y shape: {y.shape}")
            
            # Split data
            X_train, X_test, y_train, y_test, y_classes_test = self.split_data(X, y, y_classes)
            
            # Build model
            self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            
            # Train model
            self.train(X_train, y_train)
            
            # Evaluate model
            metrics = self.evaluate(X_test, y_test, y_classes_test)
            
            # Calculate trading-specific metrics
            trading_metrics = self.calculate_trading_metrics(metrics['actual'], metrics['predictions'])
            metrics['trading_metrics'] = trading_metrics
            
            logger.info("===== Trading Metrics =====")
            logger.info(f"Buy Precision: {trading_metrics['buy_precision']:.4f}")
            logger.info(f"Sell Precision: {trading_metrics['sell_precision']:.4f}")
            logger.info(f"Buy Recall: {trading_metrics['buy_recall']:.4f}")
            logger.info(f"Sell Recall: {trading_metrics['sell_recall']:.4f}")
            logger.info(f"Win Rate: {trading_metrics['win_rate']*100:.2f}%")
            logger.info(f"Profit Factor: {trading_metrics['profit_factor']:.2f}")
            logger.info(f"Total Trades: {trading_metrics['total_trades']}")
            logger.info(f"Winning Trades: {trading_metrics['winning_trades']}")
            
            # Save the final model
            self.model.save(f'{self.checkpoint_dir}/final_classification_model.h5')
            logger.info(f"Final model saved to {self.checkpoint_dir}/final_classification_model.h5")
            
            return metrics
        
        except Exception as e:
            logger.error(f"Error in pipeline execution: {str(e)}")
            logger.error(f"Exception type: {type(e).__name__}")
            import traceback
            logger.error(traceback.format_exc())
            raise


# Execute the pipeline
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from datetime import datetime
    import traceback
    import os
    
    try:
        # Initialize and run the model with enhanced parameters
        classification_model = TimeSeriesClassificationModel(
            data_path='data/merged_features/merged_BTCUSDT_4h_5y.pkl.gz',
            window_size=96,  # Increased window size for better context
            batch_size=16,   # Smaller batch size for better generalization
            epochs=100,      # More epochs with early stopping
            lstm_units=[128, 64, 32],  # LSTM layer sizes
            dropout_rate=0.5,          # Regularization
            learning_rate=0.001,       # Initial learning rate
            validation_split=0.2,      # Validation data proportion
            use_bidirectional=True,    # Use bidirectional LSTM for better performance
            threshold=0.004,           # 0.4% price change threshold for classification
            future_candles=4           # Predict 4 candles ahead
        )
        
        # Check if data file exists
        if not os.path.exists(classification_model.data_path):
            logger.error(f"Data file not found: {classification_model.data_path}")
            logger.info("Available files in data directory:")
            data_dir = os.path.dirname(classification_model.data_path)
            if os.path.exists(data_dir):
                logger.info(f"Files in {data_dir}: {os.listdir(data_dir)}")
            else:
                logger.info(f"Directory {data_dir} does not exist")
            raise FileNotFoundError(f"Data file not found: {classification_model.data_path}")
        
        # Run the full pipeline
        metrics = classification_model.run_pipeline()
        
        # Print final accuracy metrics
        logger.info("===== Final Model Performance =====")
        logger.info(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"Buy Precision: {metrics['classification_report']['Buy']['precision']:.4f}")
        logger.info(f"Sell Precision: {metrics['classification_report']['Sell']['precision']:.4f}")
        logger.info(f"Win Rate: {metrics['trading_metrics']['win_rate']*100:.2f}%")
        logger.info(f"Profit Factor: {metrics['trading_metrics']['profit_factor']:.2f}")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(traceback.format_exc())