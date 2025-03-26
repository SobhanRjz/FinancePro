import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

def train_model(model, X_train, y_train, epochs=10, batch_size=32, 
               learning_rate=0.001, validation_data=None, verbose=1, 
               grad_clip=1.0):  # Add gradient clipping parameter
    """
    Train the model similar to Keras' model.fit()
    
    Args:
        model: PyTorch model to train
        X_train: Training features (numpy array)
        y_train: Training targets (numpy array)
        epochs: Number of epochs to train
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        validation_data: Tuple of (X_val, y_val) for validation
        verbose: 0 = silent, 1 = progress bar, 2 = one line per epoch
        grad_clip: Maximum gradient norm for gradient clipping
        
    Returns:
        Dictionary containing training history
    """
    # Get the device that the model is on
    device = next(model.parameters()).device
    
    # Check if we can use mixed precision training
    use_amp = device.type == 'cuda'
    if use_amp:
        # Initialize GradScaler for mixed precision training
        scaler = torch.cuda.amp.GradScaler()
    
    # Convert numpy arrays to torch tensors and move to model's device
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long).squeeze(-1).to(device)
    
    # Create dataset and dataloader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              pin_memory=False)
    
    # Prepare validation data if provided
    if validation_data is not None:
        X_val, y_val = validation_data
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long).squeeze(-1).to(device)
        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, pin_memory=False)
    
    # Define loss function and optimizer (MSE loss for regression)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=learning_rate,
        weight_decay=1e-5,  # Small weight decay for regularization
        eps=1e-8  # Higher epsilon for numerical stability
    )
    
    # Add learning rate scheduler to reduce LR when training plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.2,  # More aggressive reduction
        patience=3,   # Wait fewer epochs
        min_lr=1e-6,  # Don't let LR get too small
        verbose=True
    )
    
    # Training history
    history = {
        'loss': [],
        'val_loss': [] if validation_data is not None else None
    }
    
    # Training loop
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        # Use tqdm for progress bar if verbose
        if verbose == 1:
            train_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        else:
            train_iter = train_loader
        
        # Train on batches
        for X_batch, y_batch in train_iter:
            # Zero the gradients
            optimizer.zero_grad()
            
            # Check for NaN values in input
            if torch.isnan(X_batch).any() or torch.isnan(y_batch).any():
                print("Warning: NaN values found in input data. Skipping batch.")
                continue
            
            if use_amp:
                # Mixed precision forward pass (use updated API)
                with torch.amp.autocast(device_type='cuda'):
                    outputs = model(X_batch)
                    
                    # Check for NaN in outputs
                    if torch.isnan(outputs).any():
                        print("Warning: NaN values in model output. Skipping batch.")
                        continue
                        
                    loss = criterion(outputs, y_batch)
                
                # Backward pass with gradient scaling
                scaler.scale(loss).backward()
                
                # Gradient clipping (on scaled gradients)
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                outputs = model(X_batch)
                
                # Check for NaN in outputs
                if torch.isnan(outputs).any():
                    print("Warning: NaN values in model output. Skipping batch.")
                    continue
                    
                loss = criterion(outputs, y_batch)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                
                # Optimizer step
                optimizer.step()
            
            # Update statistics (only if loss is not NaN)
            if not torch.isnan(loss).any():
                running_loss += loss.item() * X_batch.size(0)
            else:
                print("Warning: NaN loss detected. Skipping batch for stats.")
        
        # Calculate epoch loss
        epoch_loss = running_loss / len(train_dataset)
        history['loss'].append(epoch_loss)
        
        # Validate if validation data is provided
        if validation_data is not None:
            model.eval()
            val_loss = 0.0
            val_samples = 0
            
            with torch.no_grad():
                for X_val_batch, y_val_batch in val_loader:
                    if use_amp:
                        with torch.amp.autocast(device_type='cuda'):
                            val_outputs = model(X_val_batch)
                            batch_val_loss = criterion(val_outputs, y_val_batch)
                    else:
                        val_outputs = model(X_val_batch)
                        batch_val_loss = criterion(val_outputs, y_val_batch)
                    
                    # Only count non-NaN values
                    if not torch.isnan(batch_val_loss).any():
                        val_loss += batch_val_loss.item() * X_val_batch.size(0)
                        val_samples += X_val_batch.size(0)
            
            # Calculate average validation loss
            if val_samples > 0:
                val_loss = val_loss / val_samples
                history['val_loss'].append(val_loss)
                
                # Update learning rate based on validation loss
                scheduler.step(val_loss)
                
                if verbose >= 1:
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: {val_loss:.4f} - lr: {current_lr:.6f}")
            else:
                # Handle case where all validation batches were skipped
                history['val_loss'].append(float('nan'))
                if verbose >= 1:
                    print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f} - val_loss: NaN")
        else:
            if verbose >= 1:
                print(f"Epoch {epoch+1}/{epochs} - loss: {epoch_loss:.4f}")
    
    return history 