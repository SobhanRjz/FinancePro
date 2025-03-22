from typing import Tuple, Optional
import torch
import torch.nn as nn
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    input_size: int
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    num_classes: int = 3  # Bearish, Neutral, Bullish
    bidirectional: bool = True
    attention_heads: int = 4
    sequence_length: int = 60  # Last 60 timeframes

class AttentionLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads)
        self.norm = nn.LayerNorm(hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_output, _ = self.attention(x, x, x)
        return self.norm(x + attn_output)

class TradingLSTM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=True
        )
        
        # Attention layer
        lstm_output_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        self.attention = AttentionLayer(lstm_output_size, config.attention_heads)
        
        # Fully connected layers
        self.fc1 = nn.Linear(lstm_output_size, lstm_output_size // 2)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(lstm_output_size // 2, config.num_classes)
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x: torch.Tensor, hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> torch.Tensor:
        # LSTM forward pass
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Apply attention
        attended = self.attention(lstm_out)
        
        # Get the last output
        last_output = attended[:, -1, :]
        
        # Fully connected layers
        x = self.fc1(last_output)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        return self.softmax(x), hidden
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            predictions, _ = self.forward(x)
        return predictions 