from typing import Dict, Optional
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from tqdm import tqdm
import logging
import mlflow
from pathlib import Path

from .model import TradingLSTM, ModelConfig
from .dataset import create_dataloaders

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model: TradingLSTM,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        learning_rate: float = 0.001,
        device: str = None,
        mixed_precision: bool = True
    ):
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Log device information
        if self.device == 'cuda':
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.info("Using CPU for training")
        
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        # Enable mixed precision training if requested and GPU is available
        self.mixed_precision = mixed_precision and self.device == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=learning_rate)
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.1,
            patience=5,
            verbose=True
        )
        
    def train_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Use mixed precision if enabled
            if self.mixed_precision:
                with torch.cuda.amp.autocast():
                    output, _ = self.model(data)
                    loss = self.criterion(output, target.squeeze())
                    
                # Scale gradients and optimize
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                output, _ = self.model(data)
                loss = self.criterion(output, target.squeeze())
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
        return {
            'loss': total_loss / len(self.train_loader),
            'accuracy': 100. * correct / total
        }
    
    def evaluate(self, loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _ = self.model(data)
                
                total_loss += self.criterion(output, target.squeeze()).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
                
        return {
            'loss': total_loss / len(loader),
            'accuracy': 100. * correct / total
        }
    
    def train(
        self,
        num_epochs: int,
        checkpoint_dir: str = 'checkpoints',
        early_stopping_patience: int = 10
    ):
        best_val_loss = float('inf')
        patience_counter = 0
        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(exist_ok=True)
        
        # Handle existing MLflow runs to prevent errors
        try:
            mlflow.end_run()  # End any existing run
        except:
            pass  # Ignore if no run is active
            
        mlflow.start_run()
        for epoch in range(num_epochs):
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            
            # Train
            train_metrics = self.train_epoch()
            logger.info(f'Training metrics: {train_metrics}')
            
            # Validate
            val_metrics = self.evaluate(self.val_loader)
            logger.info(f'Validation metrics: {val_metrics}')
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['loss'])
            
            # Log metrics
            mlflow.log_metrics({
                'train_loss': train_metrics['loss'],
                'train_accuracy': train_metrics['accuracy'],
                'val_loss': val_metrics['loss'],
                'val_accuracy': val_metrics['accuracy']
            }, step=epoch)
            
            # Save checkpoint if validation loss improved
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                }, checkpoint_path / 'best_model.pt')
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info('Early stopping triggered')
                break
                
        mlflow.end_run()
        
        # Final evaluation on test set
        test_metrics = self.evaluate(self.test_loader)
        logger.info(f'Test metrics: {test_metrics}')
        
        return test_metrics 