import pandas as pd
import torch
from pathlib import Path
import yaml
import logging
import argparse

from .model import TradingLSTM, ModelConfig
from .dataset import create_dataloaders
from .trainer import ModelTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_config(config_path: str) -> dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/model_config.yaml')
    parser.add_argument('--data_dir', type=str, default='data/merged_features')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], 
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--mixed_precision', action='store_true', 
                        help='Enable mixed precision training')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Load data
    data_path = Path(args.data_dir)
    train_data = pd.read_pickle(data_path / 'train.pkl.gz')
    val_data = pd.read_pickle(data_path / 'val.pkl.gz')
    test_data = pd.read_pickle(data_path / 'test.pkl.gz')
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        sequence_length=config['sequence_length'],
        batch_size=config['batch_size']
    )
    
    # Initialize model
    model_config = ModelConfig(
        input_size=len(train_data.columns) - 1,  # Exclude target column
        **config['model']
    )
    model = TradingLSTM(model_config)
    
    # Initialize trainer
    trainer = ModelTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        learning_rate=config['training']['learning_rate'],
        device=args.device,
        mixed_precision=args.mixed_precision
    )
    
    
    # Train model
    test_metrics = trainer.train(
        num_epochs=config['training']['num_epochs'],
        checkpoint_dir=config['training']['checkpoint_dir'],
        early_stopping_patience=config['training']['early_stopping_patience']
    )
    
    logger.info(f'Final test metrics: {test_metrics}')

if __name__ == '__main__':
    main() 