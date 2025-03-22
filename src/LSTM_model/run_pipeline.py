import logging
from pathlib import Path
import pandas as pd
import yaml
import argparse
from sklearn.model_selection import train_test_split
import torch
import mlflow

# Get PyTorch version for logging and compatibility checks
torch_version = torch.__version__
logging.info(f"Using PyTorch version: {torch_version}")

# Check CUDA availability for GPU acceleration
cuda_available = torch.cuda.is_available()
cuda_version = torch.version.cuda if cuda_available else "Not available"
logging.info(f"CUDA available: {cuda_available}, version: {cuda_version}")

# Determine device for model training
device = torch.device("cuda" if cuda_available else "cpu")
logging.info(f"Using device: {device}")

import os
import sys

# Add src directory to path
src_path = str(Path(__file__).parent.parent.parent)
if src_path not in sys.path:
    sys.path.append(src_path)

from src.LSTM_model.model import TradingLSTM, ModelConfig
from src.LSTM_model.dataset import create_dataloaders
from src.LSTM_model.trainer import ModelTrainer
from src.LSTM_model.preprocess import BitcoinTradingPreprocessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def prepare_data(
    data_dir: Path,
    config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare and split data for training
    """
    logger.info("Loading and preprocessing data...")
    
    # Initialize preprocessor
    preprocessor = BitcoinTradingPreprocessor(
        data_path=str(data_dir / config['data']['filename']),
        target_col=config['data']['target_column'],
        window_size=config['data'].get('window_size', 72),
        future_offset=config['data'].get('future_offset', 1),
        threshold=config['data'].get('threshold', 0.005),
        feature_selection=config['data'].get('feature_selection'),
        scaler_type=config['data'].get('scaler_type', 'robust'),
        timeframe=config['data'].get('timeframe', '4h')
    )
    
    # Load and preprocess data
    data_splits = preprocessor.preprocess()
    
    # Extract train, validation, and test data
    X_train, y_train = data_splits['X_train'], data_splits['y_train']
    X_val, y_val = data_splits['X_val'], data_splits['y_val']
    X_test, y_test = data_splits['X_test'], data_splits['y_test']
    
    # Get feature names from data_splits
    feature_cols = data_splits['feature_names']
    target_col = config['data']['target_column']
    
    # Convert to DataFrames
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
    
    logger.info(f"Data split sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    
    # Save processed datasets
    output_dir = data_dir / 'processed'
    output_dir.mkdir(exist_ok=True)
    
    train_data.to_pickle(output_dir / 'train.pkl.gz', compression='gzip')
    val_data.to_pickle(output_dir / 'val.pkl.gz', compression='gzip')
    test_data.to_pickle(output_dir / 'test.pkl.gz', compression='gzip')
    
    return train_data, val_data, test_data

def setup_mlflow(config: dict, experiment_name: str = "bitcoin_trading"):
    """
    Setup MLflow tracking
    """
    mlflow.set_experiment(experiment_name)
    mlflow.log_params({
        "model": config['model'],
        "training": config['training']
    })

def main():
    parser = argparse.ArgumentParser(description='Run complete training pipeline')
    parser.add_argument('--config', type=str, default='src/LSTM_model/configs/model_config.yaml')
    parser.add_argument('--data_dir', type=str, default='data/merged_features')
    parser.add_argument('--force_preprocess', default=True, action='store_true', help='Force data preprocessing')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    data_dir = Path(args.data_dir)
    
    # Setup MLflow
    setup_mlflow(config)
    
    try:
        # Check if processed data exists
        processed_dir = data_dir / 'processed'
        if not processed_dir.exists() or args.force_preprocess:
            train_data, val_data, test_data = prepare_data(data_dir, config)
        else:
            logger.info("Loading preprocessed data...")
            train_data = pd.read_pickle(processed_dir / 'train.pkl.gz')
            val_data = pd.read_pickle(processed_dir / 'val.pkl.gz')
            test_data = pd.read_pickle(processed_dir / 'test.pkl.gz')
        
        # Create dataloaders
        logger.info("Creating dataloaders...")
        train_loader, val_loader, test_loader = create_dataloaders(
            train_data=train_data,
            val_data=val_data,
            test_data=test_data,
            target_col=config['data']['target_column'],
            sequence_length=config['model']['sequence_length'],
            batch_size=config['training']['batch_size']
        )
        
        # Initialize model
        logger.info("Initializing model...")
        model_config = ModelConfig(
            input_size=len(train_data.columns) - 1,  # Exclude target column
            **config['model']
        )
        model = TradingLSTM(model_config)
        
        # Setup trainer
        logger.info("Setting up trainer...")
        trainer = ModelTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            learning_rate=config['training']['learning_rate']
        )
        
        # Train model
        logger.info("Starting training...")
        test_metrics = trainer.train(
            num_epochs=config['training']['num_epochs'],
            checkpoint_dir=config['training']['checkpoint_dir'],
            early_stopping_patience=config['training']['early_stopping_patience']
        )
        
        logger.info(f"Training completed. Final test metrics: {test_metrics}")
        
    except Exception as e:
        logger.error(f"Error in pipeline: {str(e)}", exc_info=True)
        raise
    
    finally:
        # End MLflow run
        mlflow.end_run()

if __name__ == '__main__':
    main() 