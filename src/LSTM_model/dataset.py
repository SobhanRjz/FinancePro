from typing import Tuple, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class TradingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        target_column: str = 'target',
        feature_columns: Optional[List[str]] = None,
        scaler: Optional[StandardScaler] = None
    ):
        self.sequence_length = sequence_length
        self.target_column = target_column
        
        # Select features
        if feature_columns is None:
            self.feature_columns = [col for col in data.columns if col != target_column]
        else:
            self.feature_columns = feature_columns
            
        # Prepare features
        features = data[self.feature_columns].values
        
        # Scale features
        if scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            self.scaler = scaler
            features = self.scaler.transform(features)
            
        # Create sequences
        self.X = self._create_sequences(features)
        self.y = data[target_column].values[sequence_length-1:]
        
    def _create_sequences(self, data: np.ndarray) -> np.ndarray:
        sequences = []
        for i in range(len(data) - self.sequence_length + 1):
            sequences.append(data[i:i+self.sequence_length])
        return np.array(sequences)
    
    def __len__(self) -> int:
        return len(self.X)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.FloatTensor(self.X[idx]),
            torch.LongTensor([self.y[idx]])
        )

def create_dataloaders(
    train_data: pd.DataFrame,
    val_data: pd.DataFrame,
    test_data: pd.DataFrame,
    target_col: str,
    sequence_length: int,
    batch_size: int,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    # Create datasets
    train_dataset = TradingDataset(train_data, sequence_length, target_column=target_col)
    val_dataset = TradingDataset(
        val_data, 
        sequence_length,
        scaler=train_dataset.scaler,
        target_column=target_col
    )
    test_dataset = TradingDataset(
        test_data,
        sequence_length,
        scaler=train_dataset.scaler,
        target_column=target_col
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2  # Prefetch 2 batches per worker
    )
    
    return train_loader, val_loader, test_loader 