"""
Training Script for AGSMNet
Based on Huang et al. (2025) - AGSMNet Paper

Training Pipeline:
1. Load data and create spectrograms via AG-STFT
2. Train AGSMNet to predict next day's closing price
3. Evaluate using MAE, MSE, RMSE, R² metrics (Section 4.2)
4. Save best model and training history
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime
import json
from typing import Dict, Tuple, Optional
import warnings

# Suppress sklearn feature names warning
warnings.filterwarnings("ignore", message=".*X has feature names.*")

sys.path.append(str(Path(__file__).parent.parent))

from models.agsm_net import AGSMNet, AGSMNetLite, create_model
from utils.dataset import StockSpectrogramDataset, create_dataloaders
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def set_seed(seed: int = 42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    grad_clip: float = 1.0
) -> float:
    """
    Train for one epoch.
    
    Args:
        model: AGSMNet model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        grad_clip: Gradient clipping threshold
        
    Returns:
        Average training loss
    """
    model.train()
    total_loss = 0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}', leave=False)
    
    for batch_idx, (spec, target, _, _, _) in enumerate(pbar):
        spec = spec.to(device)
        target = target.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(spec)
        
        # Calculate loss
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping (prevents exploding gradients)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
    
    return total_loss / n_batches


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: str,
    train_dataset = None, # We need the dataset for denormalize_price method
    return_predictions: bool = False
) -> Dict:
    """
    Evaluate model on dataset.
    
    Computes metrics from Section 4.2:
    - MAE (Mean Absolute Error)
    - MSE (Mean Square Error)
    - RMSE (Root Mean Square Error)
    - R² (Coefficient of Determination)
    
    Args:
        model: AGSMNet model
        data_loader: Data loader
        criterion: Loss function
        device: Device
        train_dataset: Dataset instance (for denormalize_price)
        return_predictions: Whether to return predictions and targets
        
    Returns:
        Dictionary of metrics
    """
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_predictions = []
    all_targets = []
    all_last_close = []
    
    with torch.no_grad():
        for spec, target, mean, std, last_close in data_loader:
            spec = spec.to(device)
            target = target.to(device)
            # stats for denormalization
            mean = mean.numpy()
            std = std.numpy()
            last_close = last_close.numpy().flatten()
            
            # Forward pass
            output = model(spec)
            
            # Calculate loss
            loss = criterion(output, target)
            total_loss += loss.item()
            n_batches += 1
            
            # Store predictions and targets (denormalized on the fly for simplicity or later)
            # Actually easier to store raw and denormalize later, BUT we need window stats
            # So let's denormalize HERE to avoid storing means/stds for all batches
            
            pred_denorm = train_dataset.denormalize_price(output, mean, std)
            target_denorm = train_dataset.denormalize_price(target, mean, std)
            
            all_predictions.append(pred_denorm)
            all_targets.append(target_denorm)
            all_last_close.append(last_close)
            
            # Also store normalized for loss calc if needed separately, but we already have loss
            # We can track normalized metrics if we want
    
    # Concatenate all batches
    all_predictions_orig = np.concatenate(all_predictions)
    all_targets_orig = np.concatenate(all_targets)
    all_last_close_orig = np.concatenate(all_last_close)
    
    # Calculate metrics on ORIGINAL scale (as in paper)
    avg_loss = total_loss / n_batches
    mae = mean_absolute_error(all_targets_orig, all_predictions_orig)
    mse = mean_squared_error(all_targets_orig, all_predictions_orig)
    rmse = np.sqrt(mse)
    r2 = r2_score(all_targets_orig, all_predictions_orig)
    
    r2 = r2_score(all_targets_orig, all_predictions_orig)
    
    # Directional Accuracy
    # Up/Down direction: Sign(Predicted - Last) == Sign(Actual - Last)
    pred_diff = all_predictions_orig - all_last_close_orig
    actual_diff = all_targets_orig - all_last_close_orig
    
    # Correct direction match (ignoring 0 diffs for simplicity or counting as correct if both 0)
    direction_match = np.sign(pred_diff) == np.sign(actual_diff)
    da = np.mean(direction_match) * 100
    
    # Naive Baseline (Last Value Prediction)
    # Predict next = last
    naive_mae = mean_absolute_error(all_targets_orig, all_last_close_orig)
    naive_r2 = r2_score(all_targets_orig, all_last_close_orig)
    
    # Normalized metrics (approximate since we denormalized per batch)
    # We can skip exact normalized metrics or compute them if we stored tensors
    # For now, let's just use 0 or remove them if not critical, or approximate
    mae_norm = 0.0 
    r2_norm = 0.0 
    
    metrics = {
        'loss': avg_loss,
        'mae': mae,          # MAE in original currency
        'mse': mse,          # MSE in original currency²
        'rmse': rmse,        # RMSE in original currency
        'r2': r2,            # R² on original scale
        'da': da,            # Directional Accuracy (%)
        'naive_mae': naive_mae,
        'naive_r2': naive_r2,
        'mae_normalized': mae_norm,
        'r2_normalized': r2_norm,
    }
    
    if return_predictions:
        return metrics, all_predictions_orig, all_targets_orig
    
    return metrics


def train_model(
    csv_path: str,
    model_save_path: str = 'experiments/checkpoints/agsm_best.pth',
    model_type: str = 'lite',  # 'full' or 'lite'
    epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-5,
    window_size: int = 60,
    nperseg: int = 32,
    train_ratio: float = 0.8,
    patience: int = 15,
    device: str = None,
    seed: int = 42,
    verbose: bool = True,
    normalization_type: str = 'window'
) -> Tuple[nn.Module, Dict, Dict]:
    """
    Complete training pipeline for AGSMNet.
    
    Args:
        csv_path: Path to stock data CSV
        model_save_path: Where to save best model
        model_type: 'full' (AGSMNet) or 'lite' (AGSMNetLite)
        epochs: Maximum training epochs
        batch_size: Batch size
        learning_rate: Initial learning rate
        weight_decay: L2 regularization
        window_size: Lookback window (days)
        nperseg: STFT segment length
        train_ratio: Train/test split ratio
        patience: Early stopping patience
        device: Device to use (auto-detect if None)
        seed: Random seed
        verbose: Print progress
        
    Returns:
        model, train_history, test_history
    """
    # Set seed for reproducibility
    set_seed(seed)
    
    # Auto-detect device
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if verbose:
        print("=" * 70)
        print("AGSMNet Training Pipeline")
        print("=" * 70)
        print(f"Device: {device}")
        print(f"Model type: {model_type}")
        print(f"Dataset: {csv_path}")
        print(f"Window size: {window_size} days")
        print(f"STFT segment: {nperseg}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max epochs: {epochs}")
        print(f"Normalization: {normalization_type}")
    
    # Create dataloaders
    train_loader, test_loader, train_dataset = create_dataloaders(
        csv_path=csv_path,
        batch_size=batch_size,
        window_size=window_size,
        nperseg=nperseg,
        train_ratio=train_ratio,
        num_workers=0,
        normalization_type=normalization_type
    )
    
    # Get spectrogram dimensions
    spec_shape = train_dataset.get_sample_spectrogram_shape()
    freq_bins = spec_shape[1]
    time_steps = spec_shape[2]
    
    if verbose:
        print(f"\nSpectrogram shape: ({spec_shape[0]}, {freq_bins}, {time_steps})")
    
    # Create model
    if model_type == 'full':
        model = AGSMNet(
            in_channels=4,
            freq_bins=freq_bins,
            time_steps=time_steps,
            mfe_hidden=32,
            mfe_out=64,
            n_rssg_groups=2,
            n_rssb_per_group=2,
            d_state=32,
            predictor_hidden=64,
            dropout=0.2
        ).to(device)
    else:
        model = AGSMNetLite(
            in_channels=4,
            freq_bins=freq_bins,
            time_steps=time_steps,
            hidden_dim=64,
            n_mamba_layers=2,
            d_state=32,
            dropout=0.2
        ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    if verbose:
        print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler (reduce on plateau)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=verbose
    )
    
    # Training history
    train_history = {'loss': [], 'mae': [], 'rmse': [], 'r2': []}
    test_history = {'loss': [], 'mae': [], 'rmse': [], 'r2': []}
    
    best_test_r2 = -np.inf
    best_test_rmse = np.inf
    patience_counter = 0
    
    if verbose:
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
    
    for epoch in range(1, epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Evaluate on train set
        train_metrics = evaluate(
            model, train_loader, criterion, device, train_dataset
        )
        
        # Evaluate on test set
        test_metrics = evaluate(
            model, test_loader, criterion, device, train_dataset
        )
        
        # Update scheduler
        scheduler.step(test_metrics['loss'])
        
        # Store history
        for key in ['loss', 'mae', 'rmse', 'r2']:
            train_history[key].append(train_metrics[key])
            test_history[key].append(test_metrics[key])
        
        # Print progress
        if verbose:
            print(f"\nEpoch {epoch}/{epochs}")
            print(f"  Train - Loss: {train_loss:.6f} | MAE: ₹{train_metrics['mae']:.2f} | "
                  f"RMSE: ₹{train_metrics['rmse']:.2f} | R²: {train_metrics['r2']:.4f}")
            print(f"  Test  - Loss: {test_metrics['loss']:.6f} | MAE: ₹{test_metrics['mae']:.2f} | "
                  f"RMSE: ₹{test_metrics['rmse']:.2f} | R²: {test_metrics['r2']:.4f} | DA: {test_metrics['da']:.1f}%")
        
        # Save best model (based on R²)
        if test_metrics['r2'] > best_test_r2:
            best_test_r2 = test_metrics['r2']
            best_test_rmse = test_metrics['rmse']
            patience_counter = 0
            
            # Save checkpoint
            Path(model_save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'scaler': train_dataset.scaler,
                'model_config': {
                    'model_type': model_type,
                    'freq_bins': freq_bins,
                    'time_steps': time_steps,
                    'window_size': window_size,
                    'nperseg': nperseg
                }
            }, model_save_path)
            
            if verbose:
                print(f"  ✅ New best model saved! R² = {best_test_r2:.4f}")
        else:
            patience_counter += 1
            if verbose:
                print(f"  No improvement ({patience_counter}/{patience})")
        
        # Early stopping
        if patience_counter >= patience:
            if verbose:
                print(f"\n⏹  Early stopping triggered after {epoch} epochs")
            break
    
    if verbose:
        print("\n" + "=" * 70)
        print("Training Complete")
        print("=" * 70)
        print(f"Best Test R²: {best_test_r2:.4f}")
        print(f"Best Test RMSE: ₹{best_test_rmse:.2f}")
        print(f"Model saved to: {model_save_path}")
    
    # Plot training history
    plot_training_history(train_history, test_history, save_dir='experiments/results')
    
    # Final evaluation with predictions
    if verbose:
        print("\n" + "=" * 70)
        print("Final Evaluation")
        print("=" * 70)
    
    # Load best model
    checkpoint = torch.load(model_save_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    final_metrics, predictions, targets = evaluate(
        model, test_loader, criterion, device,
        train_dataset, return_predictions=True
    )
    
    if verbose:
        print(f"Final Test Metrics:")
        print(f"  MAE:  ₹{final_metrics['mae']:.2f} (Naive: ₹{final_metrics['naive_mae']:.2f})")
        print(f"  MSE:  {final_metrics['mse']:.2f}")
        print(f"  RMSE: ₹{final_metrics['rmse']:.2f}")
        print(f"  R²:   {final_metrics['r2']:.4f} (Naive: {final_metrics['naive_r2']:.4f})")
        print(f"  DA:   {final_metrics['da']:.2f}%")
        
        if final_metrics['r2'] > final_metrics['naive_r2']:
             print(f"✅ Model beats naive baseline by {final_metrics['r2'] - final_metrics['naive_r2']:.4f} R²")
        else:
             print(f"❌ Model fails to beat naive baseline")
    
    # Plot predictions
    plot_predictions(predictions, targets, save_dir='experiments/results')
    
    return model, train_history, test_history


def plot_training_history(
    train_history: Dict,
    test_history: Dict,
    save_dir: str = 'experiments/results'
):
    """Plot and save training history."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    ax = axes[0, 0]
    ax.plot(train_history['loss'], label='Train', color='blue')
    ax.plot(test_history['loss'], label='Test', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training & Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # MAE
    ax = axes[0, 1]
    ax.plot(train_history['mae'], label='Train', color='blue')
    ax.plot(test_history['mae'], label='Test', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MAE (₹)')
    ax.set_title('Mean Absolute Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1, 0]
    ax.plot(train_history['rmse'], label='Train', color='blue')
    ax.plot(test_history['rmse'], label='Test', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE (₹)')
    ax.set_title('Root Mean Square Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R²
    ax = axes[1, 1]
    ax.plot(train_history['r2'], label='Train', color='blue')
    ax.plot(test_history['r2'], label='Test', color='orange')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('R² Score')
    ax.set_title('Coefficient of Determination')
    ax.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n✅ Training history saved to: {save_dir / 'training_history.png'}")


def plot_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    save_dir: str = 'experiments/results',
    n_samples: int = 200
):
    """Plot predicted vs actual prices."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Time series plot
    ax = axes[0]
    n = min(n_samples, len(predictions))
    ax.plot(targets[:n], label='Actual', color='blue', alpha=0.7)
    ax.plot(predictions[:n], label='Predicted', color='orange', alpha=0.7)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Price (₹)')
    ax.set_title(f'Predicted vs Actual Prices (First {n} samples)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Scatter plot
    ax = axes[1]
    ax.scatter(targets, predictions, alpha=0.5, s=20)
    
    # Perfect prediction line
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect prediction')
    
    ax.set_xlabel('Actual Price (₹)')
    ax.set_ylabel('Predicted Price (₹)')
    ax.set_title('Prediction Scatter Plot')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_dir / 'predictions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Predictions plot saved to: {save_dir / 'predictions.png'}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train AGSMNet')
    parser.add_argument('--data', type=str, default='data/raw/RELIANCE.csv',
                        help='Path to stock data CSV')
    parser.add_argument('--model', type=str, default='lite', choices=['full', 'lite'],
                        help='Model type: full (AGSMNet) or lite (AGSMNetLite)')
    parser.add_argument('--epochs', type=int, default=100, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--window', type=int, default=60, help='Lookback window (days)')
    parser.add_argument('--nperseg', type=int, default=32, help='STFT segment length')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--norm', type=str, default='window', choices=['window', 'global'],
                        help='Normalization type: window (adaptive) or global (fixed)')
    
    args = parser.parse_args()
    
    # Train model
    model, train_hist, test_hist = train_model(
        csv_path=args.data,
        model_type=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        window_size=args.window,
        nperseg=args.nperseg,
        patience=args.patience,
        seed=args.seed,
        normalization_type=args.norm
    )
    
    print("\n✅ Training complete!")
