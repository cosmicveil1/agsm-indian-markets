"""
Training script for AGSMNet
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import json
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models.agsm_net import AGSMNet
from utils.dataset import StockSpectrogramDataset

def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    for batch_idx, (spectrograms, targets) in enumerate(train_loader):
        # Move to device
        spectrograms = spectrograms.to(device)
        targets = targets.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(spectrograms)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    return avg_loss

def evaluate(model, test_loader, criterion, device, dataset=None):
    """Evaluate on test set"""
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for spectrograms, targets in test_loader:
            spectrograms = spectrograms.to(device)
            targets = targets.to(device)
            
            predictions = model(spectrograms)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    targets = np.array(all_targets).flatten()
    
    # Compute metrics
    mae = np.mean(np.abs(predictions - targets))
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    
    # R² score
    ss_res = np.sum((targets - predictions) ** 2)
    ss_tot = np.sum((targets - np.mean(targets)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # If dataset provided, compute metrics in original scale
    if dataset:
        pred_original = dataset.denormalize(predictions)
        target_original = dataset.denormalize(targets)
        
        mae_original = np.mean(np.abs(pred_original - target_original))
        rmse_original = np.sqrt(np.mean((pred_original - target_original) ** 2))
        
        return {
            'loss': avg_loss,
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'r2': r2,
            'mae_original': mae_original,
            'rmse_original': rmse_original
        }
    
    return {
        'loss': avg_loss,
        'mae': mae,
        'mse': mse,
        'rmse': rmse,
        'r2': r2
    }

def train_model(
    stock_name='RELIANCE',
    window_size=20,
    nperseg=20,
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001,
    device='cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Complete training pipeline
    """
    print("=" * 60)
    print(f"Training AGSMNet on {stock_name}")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Window size: {window_size}")
    print(f"STFT segment: {nperseg}")
    print(f"Batch size: {batch_size}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning rate: {learning_rate}")
    print("=" * 60)
    
    # Create datasets
    csv_path = Path('data/raw') / f'{stock_name}.csv'
    
    train_dataset = StockSpectrogramDataset(
       csv_path=csv_path,
       window_size=window_size,
       nperseg=nperseg,
       train=True,
       train_ratio=0.8
    )
    
    test_dataset = StockSpectrogramDataset(
       csv_path=csv_path,
       window_size=window_size,
       nperseg=nperseg,
       train=False,
       train_ratio=0.8,
      scaler=train_dataset.scaler  # ← Share the scaler!
    )
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0  # Set to 0 for Windows compatibility
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    # Get sample to determine dimensions
    sample_spec, _ = train_dataset[0]
    freq_bins, time_steps = sample_spec.shape[1], sample_spec.shape[2]
    
    print(f"\nData loaded:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Test samples: {len(test_dataset)}")
    print(f"  Spectrogram shape: (4, {freq_bins}, {time_steps})")
    
    # Create model
    model = AGSMNet(
        freq_bins=freq_bins,
        time_steps=time_steps,
        in_channels=4,
        hidden_dim=64,
        n_mamba_layers=3,
        d_state=64,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60)
    
    best_r2 = -float('inf')
    history = {
        'train_loss': [],
        'test_loss': [],
        'test_metrics': []
    }
    
    for epoch in range(num_epochs):
        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, criterion, device, test_dataset)
        
        # Update scheduler
        scheduler.step(test_metrics['loss'])
        
        # Save history
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_metrics['loss'])
        history['test_metrics'].append(test_metrics)
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Test Loss:  {test_metrics['loss']:.6f}")
            print(f"  MAE (₹):    {test_metrics['mae_original']:.2f}")
            print(f"  RMSE (₹):   {test_metrics['rmse_original']:.2f}")
            print(f"  R²:         {test_metrics['r2']:.4f}")
        
        # Save best model
        if test_metrics['r2'] > best_r2:
            best_r2 = test_metrics['r2']
            
            # Create checkpoint directory
            checkpoint_dir = Path('experiments/checkpoints')
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Save model
            checkpoint_path = checkpoint_dir / f'{stock_name}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'metrics': test_metrics,
                'config': {
                    'stock_name': stock_name,
                    'window_size': window_size,
                    'nperseg': nperseg,
                    'freq_bins': freq_bins,
                    'time_steps': time_steps
                }
            }, checkpoint_path)
    
    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)
    print(f"\nBest Results:")
    print(f"  R²:       {best_r2:.4f}")
    
    # Save final results
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'stock_name': stock_name,
        'best_r2': float(best_r2),
        'final_metrics': history['test_metrics'][-1],
        'config': {
            'window_size': window_size,
            'nperseg': nperseg,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'learning_rate': learning_rate
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(results_dir / f'{stock_name}_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✅ Results saved to experiments/results/{stock_name}_results.json")
    print(f"✅ Best model saved to experiments/checkpoints/{stock_name}_best.pth")
    
    return model, history

if __name__ == "__main__":
    # Train on RELIANCE

    model, history = train_model(
    stock_name='RELIANCE',
    window_size=60,   # ← Larger window = more data
    nperseg=60,       # ← Larger STFT segment
    batch_size=32,
    num_epochs=50,
    learning_rate=0.001
    )