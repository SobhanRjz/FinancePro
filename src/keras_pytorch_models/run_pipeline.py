import logging
import pandas as pd
import yaml
import argparse
import sys
import os
from pathlib import Path
import torch
import numpy as np
import gc
import time
from datetime import datetime
import mlflow
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('keras_pytorch_pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

# Import local modules
from src.keras_pytorch_models.keras_lstm import KerasLSTM
from src.keras_pytorch_models.gru_model import KerasGRU
from src.keras_pytorch_models.train import train_model
from src.LSTM_model.preprocess import BitcoinTradingPreprocessor  # Reuse the same preprocessor


def optimize_gpu_memory():
    """Settings to optimize GPU memory usage"""
    if torch.cuda.is_available():
        # Enable memory efficient features
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Empty cache
        torch.cuda.empty_cache()
        
        # Get GPU info
        device_props = torch.cuda.get_device_properties(0)
        gpu_name = device_props.name
        total_memory = device_props.total_memory / (1024**3)  # GB
        
        logger.info(f"Using GPU: {gpu_name}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"Available memory: {total_memory:.2f} GB")
        
        return True
    else:
        logger.info("No GPU available, using CPU")
        return False

def get_device(device_arg=None):
    """Get the device to use (GPU if available unless overridden)"""
    if device_arg is not None:
        if device_arg.lower() == 'cpu':
            return torch.device('cpu')
        elif device_arg.lower() == 'cuda' and torch.cuda.is_available():
            optimize_gpu_memory()
            return torch.device('cuda')
        else:
            logger.warning(f"Requested device {device_arg} not available, checking alternatives")
    
    # No specific device requested or requested device not available
    if torch.cuda.is_available():
        optimize_gpu_memory()
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_data(data_dir: Path, config: dict, force_preprocess: bool = False) -> tuple:
    """
    Load cached processed data or generate new data
    """
    processed_dir = data_dir / 'merged_features'
    processed_dir.mkdir(exist_ok=True)
    
    train_path = processed_dir / 'train.pkl.gz'
    val_path = processed_dir / 'val.pkl.gz'
    test_path = processed_dir / 'test.pkl.gz'
    
    # Check if all processed files exist and force_preprocess is False
    if not force_preprocess and train_path.exists() and val_path.exists() and test_path.exists():
        logger.info("Loading preprocessed data...")
        start_time = time.time()
        train_data = pd.read_pickle(train_path)
        val_data = pd.read_pickle(val_path)
        test_data = pd.read_pickle(test_path)
        logger.info(f"Data loaded in {time.time() - start_time:.2f} seconds")
        return train_data, val_data, test_data
    
    # If any file doesn't exist, preprocess
    logger.info("Preprocessing data...")
    # Initialize preprocessor
    preprocessor = BitcoinTradingPreprocessor(
        data_path=str(processed_dir / config['data']['filename']),
        target_col=config['data']['target_column'],
        window_size=config['data'].get('window_size'),
        future_offset=config['data'].get('future_offset'),
        threshold=config['data'].get('threshold'),
        feature_selection=config['data'].get('feature_selection'),
        scaler_type=config['data'].get('scaler_type', 'robust'),
        timeframe=config['data'].get('timeframe')
    )
    
    data_splits = preprocessor.preprocess()
    train_data, val_data, test_data = _create_dataframes_from_splits(data_splits, config['data']['target_column'])
    
    # Save preprocessed data
    logger.info("Saving preprocessed data...")
    train_data.to_pickle(train_path)
    val_data.to_pickle(val_path)
    test_data.to_pickle(test_path)
    
    return train_data, val_data, test_data


def optimize_memory():
    """Free up memory by clearing caches and running garbage collection"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _create_dataframes_from_splits(data_splits, target_col):
    """Helper function to create DataFrames from preprocessor output"""
    # Extract data
    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    X_val, y_val = data_splits['X_val'], data_splits['y_val']
    X_test, y_test = data_splits['X_test'], data_splits['y_test']
    
    # Get feature names
    feature_cols = data_splits['feature_names']
    
    # Initialize DataFrames
    train_data = pd.DataFrame()
    val_data = pd.DataFrame()
    test_data = pd.DataFrame()
    
    # Add features (using the last timestep of each sequence)
    for i, col in enumerate(feature_cols):
        train_data[col] = X_train[:, -1, i]
        val_data[col] = X_val[:, -1, i]
        test_data[col] = X_test[:, -1, i]
    
    # Add target column
    train_data[target_col] = y_train
    val_data[target_col] = y_val
    test_data[target_col] = y_test
    
    return train_data, val_data, test_data


def prepare_tensors(train_data, val_data, test_data, target_col, sequence_length):
    """
    Convert pandas dataframes to numpy arrays for model training
    """
    # Extract features and targets
    X_train = train_data.drop(columns=[target_col]).values
    y_train = train_data[[target_col]].values
    
    X_val = val_data.drop(columns=[target_col]).values
    y_val = val_data[[target_col]].values
    
    X_test = test_data.drop(columns=[target_col]).values
    y_test = test_data[[target_col]].values
    
    # Reshape features for sequential models [samples, time steps, features]
    features = train_data.shape[1] - 1  # Exclude target column
    
    # Create sequences of 'sequence_length' timesteps
    X_train_seq = []
    y_train_seq = []
    for i in range(len(X_train) - sequence_length + 1):
        X_train_seq.append(X_train[i:i+sequence_length])
        y_train_seq.append(y_train[i+sequence_length-1])
    
    X_val_seq = []
    y_val_seq = []
    for i in range(len(X_val) - sequence_length + 1):
        X_val_seq.append(X_val[i:i+sequence_length])
        y_val_seq.append(y_val[i+sequence_length-1])
        
    X_test_seq = []
    y_test_seq = []
    for i in range(len(X_test) - sequence_length + 1):
        X_test_seq.append(X_test[i:i+sequence_length])
        y_test_seq.append(y_test[i+sequence_length-1])
    
    # Convert to numpy arrays
    X_train_seq = np.array(X_train_seq, dtype=np.float32)
    y_train_seq = np.array(y_train_seq, dtype=np.float32)
    
    X_val_seq = np.array(X_val_seq, dtype=np.float32)
    y_val_seq = np.array(y_val_seq, dtype=np.float32)
    
    X_test_seq = np.array(X_test_seq, dtype=np.float32)
    y_test_seq = np.array(y_test_seq, dtype=np.float32)
    
    logger.info(f"X_train shape: {X_train_seq.shape}, y_train shape: {y_train_seq.shape}")
    logger.info(f"X_val shape: {X_val_seq.shape}, y_val shape: {y_val_seq.shape}")
    logger.info(f"X_test shape: {X_test_seq.shape}, y_test shape: {y_test_seq.shape}")
    
    return (X_train_seq, y_train_seq), (X_val_seq, y_val_seq), (X_test_seq, y_test_seq)


def main():
    """Main pipeline function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Keras-PyTorch pipeline')
    parser.add_argument('--config', type=str, default='src/LSTM_model/configs/model_config.yaml',
                        help='Path to config file')
    parser.add_argument('--force_preprocess', action='store_true', default=True,
                        help='Force preprocessing of data')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cpu, cuda, or leave empty for auto-detection)')
    parser.add_argument('--model_type', type=str, default='lstm', choices=['lstm', 'gru'],
                        help='Type of model to use (lstm or gru)')
    args = parser.parse_args()
    
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Determine device
    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    logger.info(f"Using device: {device}")
    
    # Set PyTorch to use the selected device
    if device == 'cuda' and torch.cuda.is_available():
        torch.cuda.set_device(0)
        logger.info(f"CUDA Version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
        # Set PyTorch to use TF32 and cuDNN benchmarking for performance
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    
    # Load configuration
    config = load_config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set experiment directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"keras_{args.model_type}_model_{timestamp}"
    
    data_dir = Path('data')
    checkpoint_dir = Path('checkpoints') / exp_name
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    
    try:
        # MLflow setup
        mlflow.set_experiment(f"KerasPyTorch_{args.model_type.upper()}")
        
        with mlflow.start_run(run_name=exp_name):
            # Log configuration parameters
            mlflow.log_params({
                "model_type": args.model_type,
                "device": device,
                "sequence_length": config['model']['sequence_length'],
                "hidden_size": 50,  # Hardcoded for Keras-like model 
                "dropout_rate": 0.2,  # Hardcoded for Keras-like model
                "batch_size": config['training']['batch_size'],
                "epochs": config['training']['num_epochs'],
                "learning_rate": config['training']['learning_rate']
            })
            
            # Prepare data
            train_data, val_data, test_data = prepare_data(data_dir, config, args.force_preprocess)
            # Prepare sequences for Keras-style models
            sequence_length = config['model']['sequence_length']
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_tensors(
                train_data, val_data, test_data, 
                config['data']['target_column'],
                sequence_length
            )
            
            # Create model
            logger.info(f"Creating {args.model_type.upper()} model...")
            input_shape = (sequence_length, X_train.shape[2])
            
            # Use a much lower learning rate for stability
            learning_rate = 0.001  # Reduce learning rate by 10x

            if args.model_type.lower() == 'lstm':
                model = KerasLSTM(
                    input_shape=input_shape,
                    hidden_size=50,
                    dropout_rate=0.3,  # Increased dropout
                    output_size=3
                )
            else:  # GRU
                model = KerasGRU(
                    input_shape=input_shape,
                    hidden_size=50,
                    dropout_rate=0.3,
                    output_size=3
                )
            
            # Print model summary
            model.summary()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"Model parameters: {trainable_params:,} trainable, {total_params:,} total")
            
            # Move model to device and enable optimization
            model = model.to(device)
            if device == 'cuda':
                # Mixed precision for faster training
                scaler = torch.cuda.amp.GradScaler()
                logger.info("Using mixed precision training")
            else:
                scaler = None
            
            # Train model
            logger.info("Starting training...")
            history = train_model(
                model=model,
                X_train=X_train,
                y_train=y_train,
                epochs=config['training']['num_epochs'],
                batch_size=config['training']['batch_size'],
                learning_rate=learning_rate,  # Use the reduced learning rate
                validation_data=(X_val, y_val),
                verbose=1,
                grad_clip=0.5  # Use a smaller gradient clipping threshold
            )
            
            # Save model
            model_path = checkpoint_dir / f"{args.model_type}_model.pt"
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_shape': input_shape,
                'hidden_size': 50,
                'dropout_rate': 0.2,
                'output_size': 1
            }, model_path)
            mlflow.log_artifact(str(model_path))
            
            # Log metrics
            for i, loss in enumerate(history['loss']):
                mlflow.log_metric("train_loss", loss, step=i)
                if history['val_loss']:
                    mlflow.log_metric("val_loss", history['val_loss'][i], step=i)
            
            # Evaluate on test set
            model.eval()
            with torch.no_grad():
                X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_test_tensor = torch.tensor(y_test, dtype=torch.long).squeeze(-1).to(device)  # Ensure target is 1D
                
                # Get model predictions
                y_pred = model(X_test_tensor)
                
                # Calculate loss - ensure inputs and targets have correct shape
                criterion = torch.nn.CrossEntropyLoss()
                test_loss = criterion(y_pred, y_test_tensor).item()
                
                # Get class predictions
                preds = torch.argmax(y_pred, dim=1)  # shape: (batch_size,)
            
            mlflow.log_metric("test_loss", test_loss)
            logger.info(f"Test Loss: {test_loss:.6f}")
            
            # Save predictions vs actual
            y_pred_np = preds.cpu().numpy()
            y_test_np = y_test_tensor.cpu().numpy()  # Use the already squeezed tensor
            
            # Create a more meaningful results dataframe
            results_df = pd.DataFrame({
                'predicted_class': y_pred_np,
                'actual_class': y_test_np
            })
            # Add confusion matrix calculation
            conf_matrix = confusion_matrix(y_test_np, y_pred_np)
            class_report = classification_report(y_test_np, y_pred_np)
            
            # Log these to both console and file
            logger.info(f"Confusion Matrix:\n{conf_matrix}")
            logger.info(f"Classification Report:\n{class_report}")
            
            results_path = checkpoint_dir / "test_predictions.csv"
            results_df.to_csv(results_path, index=False)
            mlflow.log_artifact(str(results_path))
            
            logger.info(f"Training complete. Results saved to {checkpoint_dir}")
    
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
    finally:
        # Clean up resources
        optimize_memory()


if __name__ == '__main__':
    main() 