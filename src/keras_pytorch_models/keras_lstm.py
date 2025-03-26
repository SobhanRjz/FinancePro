import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class KerasLSTM(nn.Module):
    """
    PyTorch implementation of the Keras LSTM model:
    
    lstm_model = Sequential()
    lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50, return_sequences=True))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(50))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(1))
    """
    
    def __init__(self, input_shape, hidden_size=50, dropout_rate=0.2, output_size=1):
        """
        Initialize the model with the exact same structure as the Keras example
        but with added stabilization techniques
        
        Args:
            input_shape: Tuple of (sequence_length, num_features)
            hidden_size: Size of LSTM hidden layers (default: 50)
            dropout_rate: Dropout rate (default: 0.2)
            output_size: Size of output (default: 1)
        """
        super(KerasLSTM, self).__init__()
        
        sequence_length, num_features = input_shape
        
        # First LSTM layer with dropout - PyTorch doesn't use return_sequences parameter
        self.lstm1 = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.norm1 = nn.LayerNorm(hidden_size)  # Add layer normalization
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second LSTM layer with dropout
        self.lstm2 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.norm2 = nn.LayerNorm(hidden_size)  # Add layer normalization
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third LSTM layer with dropout
        self.lstm3 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.norm3 = nn.LayerNorm(hidden_size)  # Add layer normalization
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth LSTM layer with dropout
        self.lstm4 = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.norm4 = nn.LayerNorm(hidden_size)  # Add layer normalization
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.dense = nn.Linear(hidden_size, output_size)
        
        # Initialize weights with a more conservative method
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with a more conservative method for stability"""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-to-hidden weights: use xavier with smaller gain
                nn.init.xavier_uniform_(param.data, gain=0.5)
            elif 'weight_hh' in name:
                # Hidden-to-hidden weights: use orthogonal with smaller gain
                nn.init.orthogonal_(param.data, gain=0.5)
            elif 'bias' in name:
                # Biases: initialize to small positive values (helps with forgetting)
                param.data.fill_(0.1)
            elif 'weight' in name and 'norm' not in name:
                # Linear layer weights
                nn.init.xavier_uniform_(param.data, gain=0.5)
            elif 'bias' in name and 'norm' not in name:
                # Linear layer biases
                param.data.fill_(0.0)
    
    def forward(self, x):
        """
        Forward pass through the network with added layer normalization
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, features)
            
        Returns:
            Output tensor of shape (batch_size, output_size)
        """
        # Apply residual connections and layer normalization for stability
        
        # First LSTM + normalization + dropout
        out, _ = self.lstm1(x)
        out = self.norm1(out)  # Apply layer normalization
        out = self.dropout1(out)
        
        # Second LSTM + normalization + dropout
        out_main, _ = self.lstm2(out)
        out = self.norm2(out_main)  # Apply layer normalization
        # Add residual connection if shapes match
        if out.shape == out_main.shape:
            out = out + out_main
        out = self.dropout2(out)
        
        # Third LSTM + normalization + dropout
        out_main, _ = self.lstm3(out)
        out = self.norm3(out_main)  # Apply layer normalization
        # Add residual connection if shapes match
        if out.shape == out_main.shape:
            out = out + out_main
        out = self.dropout3(out)
        
        # Fourth LSTM + normalization + dropout (take only last timestep)
        out_main, _ = self.lstm4(out)
        out = self.norm4(out_main)  # Apply layer normalization
        # In PyTorch we manually select the last time step to mimic return_sequences=False
        out = out[:, -1, :]  # Get the last time step output
        out = self.dropout4(out)
        
        # Dense output layer
        out = self.dense(out)
        
        return out
    
    def summary(self):
        """Print a summary of the model architecture similar to Keras"""
        print("Model: KerasLSTM")
        print("_" * 80)
        print("{:<20} {:<20} {:<20}".format("Layer (type)", "Output Shape", "Param #"))
        print("=" * 80)
        
        total_params = 0
        
        # Calculate LSTM1 params
        lstm1_input_size = self.lstm1.input_size
        lstm1_hidden_size = self.lstm1.hidden_size
        lstm1_params = 4 * ((lstm1_input_size * lstm1_hidden_size) + 
                           (lstm1_hidden_size * lstm1_hidden_size) + 
                           lstm1_hidden_size)
        total_params += lstm1_params
        print("{:<20} {:<20} {:<20}".format(
            "lstm1 (LSTM)", f"(None, None, {lstm1_hidden_size})", lstm1_params))
        
        # Dropout1 has no parameters
        print("{:<20} {:<20} {:<20}".format(
            "dropout1 (Dropout)", f"(None, None, {lstm1_hidden_size})", 0))
        
        # Calculate LSTM2 params
        lstm2_input_size = self.lstm2.input_size
        lstm2_hidden_size = self.lstm2.hidden_size
        lstm2_params = 4 * ((lstm2_input_size * lstm2_hidden_size) + 
                           (lstm2_hidden_size * lstm2_hidden_size) + 
                           lstm2_hidden_size)
        total_params += lstm2_params
        print("{:<20} {:<20} {:<20}".format(
            "lstm2 (LSTM)", f"(None, None, {lstm2_hidden_size})", lstm2_params))
        
        # Dropout2 has no parameters
        print("{:<20} {:<20} {:<20}".format(
            "dropout2 (Dropout)", f"(None, None, {lstm2_hidden_size})", 0))
        
        # Calculate LSTM3 params
        lstm3_input_size = self.lstm3.input_size
        lstm3_hidden_size = self.lstm3.hidden_size
        lstm3_params = 4 * ((lstm3_input_size * lstm3_hidden_size) + 
                           (lstm3_hidden_size * lstm3_hidden_size) + 
                           lstm3_hidden_size)
        total_params += lstm3_params
        print("{:<20} {:<20} {:<20}".format(
            "lstm3 (LSTM)", f"(None, None, {lstm3_hidden_size})", lstm3_params))
        
        # Dropout3 has no parameters
        print("{:<20} {:<20} {:<20}".format(
            "dropout3 (Dropout)", f"(None, None, {lstm3_hidden_size})", 0))
        
        # Calculate LSTM4 params
        lstm4_input_size = self.lstm4.input_size
        lstm4_hidden_size = self.lstm4.hidden_size
        lstm4_params = 4 * ((lstm4_input_size * lstm4_hidden_size) + 
                           (lstm4_hidden_size * lstm4_hidden_size) + 
                           lstm4_hidden_size)
        total_params += lstm4_params
        print("{:<20} {:<20} {:<20}".format(
            "lstm4 (LSTM)", f"(None, {lstm4_hidden_size})", lstm4_params))
        
        # Dropout4 has no parameters
        print("{:<20} {:<20} {:<20}".format(
            "dropout4 (Dropout)", f"(None, {lstm4_hidden_size})", 0))
        
        # Calculate Dense params
        dense_in_features = self.dense.in_features
        dense_out_features = self.dense.out_features
        dense_params = dense_in_features * dense_out_features + dense_out_features
        total_params += dense_params
        print("{:<20} {:<20} {:<20}".format(
            "dense (Linear)", f"(None, {dense_out_features})", dense_params))
        
        print("=" * 80)
        print(f"Total params: {total_params:,}")
        print(f"Trainable params: {total_params:,}")
        print(f"Non-trainable params: 0")
        print("_" * 80) 