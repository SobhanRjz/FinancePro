import torch
import torch.nn as nn

class KerasGRU(nn.Module):
    """
    PyTorch implementation of a Keras GRU model.
    
    Example Keras code:
    gru_model = Sequential()
    gru_model.add(GRU(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    gru_model.add(Dropout(0.2))
    gru_model.add(GRU(50, return_sequences=True))
    gru_model.add(Dropout(0.2))
    gru_model.add(GRU(50, return_sequences=True))
    gru_model.add(Dropout(0.2))
    gru_model.add(GRU(50))
    gru_model.add(Dropout(0.2))
    gru_model.add(Dense(1))
    """
    
    def __init__(self, input_shape, hidden_size=50, dropout_rate=0.2, output_size=1):
        """
        Initialize the model with the same structure as a Keras GRU model
        
        Args:
            input_shape: Tuple of (sequence_length, num_features)
            hidden_size: Size of GRU hidden layers (default: 50)
            dropout_rate: Dropout rate (default: 0.2)
            output_size: Size of output (default: 1)
        """
        super(KerasGRU, self).__init__()
        
        sequence_length, num_features = input_shape
        
        # First GRU layer with dropout - PyTorch doesn't use return_sequences parameter
        self.gru1 = nn.GRU(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second GRU layer with dropout
        self.gru2 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third GRU layer with dropout
        self.gru3 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth GRU layer with dropout
        self.gru4 = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.dense = nn.Linear(hidden_size, output_size)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly for better training"""
        for m in self.modules():
            if isinstance(m, nn.GRU):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0)
    
    def forward(self, x):
        """
        Forward pass through the network
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # First GRU layer + dropout (return_sequences=True equivalent)
        out, _ = self.gru1(x)
        out = self.dropout1(out)
        
        # Second GRU layer + dropout (return_sequences=True equivalent)
        out, _ = self.gru2(out)
        out = self.dropout2(out)
        
        # Third GRU layer + dropout (return_sequences=True equivalent)
        out, _ = self.gru3(out)
        out = self.dropout3(out)
        
        # Fourth GRU layer + dropout (return_sequences=False equivalent - take only last output)
        out, _ = self.gru4(out)
        # In PyTorch we need to manually select the last time step to mimic return_sequences=False
        out = out[:, -1, :]  # Get the last time step output
        out = self.dropout4(out)
        
        # Dense output layer
        out = self.dense(out)
        
        return out 