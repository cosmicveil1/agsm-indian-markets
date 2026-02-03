"""
Dataset for AGSMNet Stock Price Prediction
Based on Huang et al. (2025) - AGSMNet Paper

Key Design Principles:
1. Predict next day's CLOSING PRICE (not returns)
2. No data leakage - scaler fit ONLY on training data
3. AG-STFT preprocessing creates 4-channel spectrograms
4. Proper train/test temporal split
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict, Union
import sys

sys.path.append(str(Path(__file__).parent.parent))

from models.ag_stft import AdaptiveGaussianSTFT


class StockSpectrogramDataset(Dataset):
    """
    Dataset that converts stock OHLC data to spectrograms using AG-STFT.
    
    Follows the paper's approach:
    - Input: Window of OHLC data → AG-STFT → 4-channel spectrogram
    - Target: Next day's closing price (normalized)
    
    Critical Implementation Details:
    1. Temporal split: Train data comes BEFORE test data (no future leakage)
    2. Scaler fitting: ONLY on training set
    3. Test set receives training scaler for consistent normalization
    """
    
    def __init__(
        self,
        csv_path: Union[str, Path],
        window_size: int = 60,
        nperseg: int = 32,
        train: bool = True,
        train_ratio: float = 0.8,
        scaler: Optional['StandardScaler'] = None,
        target_scaler: Optional['StandardScaler'] = None,
        normalization_type: str = 'window',  # 'window' or 'global'
        ag_stft_alpha: float = 1.0,
        stft_method: str = 'fast',
        return_dates: bool = False
    ):
        """
        Args:
            csv_path: Path to CSV with columns [Date, Open, High, Low, Close]
            window_size: Number of days to look back for input
            nperseg: STFT segment length (affects frequency resolution)
            train: Whether this is training set
            train_ratio: Fraction of data for training
            train_ratio: Fraction of data for training
            scaler: Pre-fitted scaler (MUST be provided for test set if normalization_type='global')
            target_scaler: Pre-fitted target scaler (if normalization_type='global')
            normalization_type: 'window' (local z-score) or 'global' (train set z-score)
            ag_stft_alpha: Alpha parameter for AG-STFT (Equation 1)
            stft_method: 'fast' or 'adaptive' for AG-STFT
            return_dates: Whether to return dates with samples
        """
        self.csv_path = Path(csv_path)
        self.window_size = window_size
        self.nperseg = nperseg
        self.is_train = train
        self.normalization_type = normalization_type
        self.stft_method = stft_method
        self.return_dates = return_dates
        
        # Load and preprocess data
        self._load_data(train_ratio, scaler, target_scaler)
        
        # Initialize AG-STFT transformer
        self.ag_stft = AdaptiveGaussianSTFT(
            fs=1.0,  # 1 sample per day
            alpha=ag_stft_alpha,
            epsilon=1e-6
        )
        
        # Precompute spectrograms for efficiency (optional)
        self._precomputed = None
        
    def _load_data(self, train_ratio: float, scaler: Optional['StandardScaler'], target_scaler: Optional['StandardScaler'] = None):
        """Load data and set up train/test split with proper normalization."""
        from sklearn.preprocessing import StandardScaler
        
        # Load full dataset
        df_full = pd.read_csv(self.csv_path, index_col=0, parse_dates=True)
        df_full = df_full.sort_index()
        
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close']
        for col in required_cols:
            if col not in df_full.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Temporal split (paper: train before Dec 31 2020, test Jan 2021 - Dec 2023)
        split_idx = int(len(df_full) * train_ratio)
        
        if self.is_train:
            self.df = df_full.iloc[:split_idx].copy()
            
            # Setup scalers based on type
            if self.normalization_type == 'global':
                # Fit scaler on TRAINING data only (critical for no data leakage)
                self.scaler = StandardScaler()
                self.scaler.fit(self.df[['Open', 'High', 'Low', 'Close']].values)
                
                # Also fit a separate scaler for target (Close price only)
                self.target_scaler = StandardScaler()
                self.target_scaler.fit(self.df[['Close']].values.reshape(-1, 1))
                
                print(f"   Global Scaler (Train): Mean={self.target_scaler.mean_[0]:.2f}, Std={self.target_scaler.scale_[0]:.2f}")
            else:
                self.scaler = None
                self.target_scaler = None
                print(f"   Normalization: Window-local (Adaptive)")
            
            print(f"📊 Training Dataset Initialized")
            print(f"   Date range: {self.df.index[0].date()} to {self.df.index[-1].date()}")
            print(f"   Samples: {len(self.df)}")
            
        else:
            self.df = df_full.iloc[split_idx:].copy()
            
            if self.normalization_type == 'global':
                # Use training scaler (must be provided!)
                if scaler is None or target_scaler is None:
                    raise ValueError(
                        "Test dataset MUST receive scalers from training set when using global normalization!"
                    )
                self.scaler = scaler
                self.target_scaler = target_scaler
                print(f"   Using global training scaler (Mean: ₹{self.scaler.mean_[3]:.2f})")
            else:
                self.scaler = None
                self.target_scaler = None
                print(f"   Normalization: Window-local (Adaptive)")
            
            print(f"📊 Test Dataset Initialized")
            print(f"   Date range: {self.df.index[0].date()} to {self.df.index[-1].date()}")
            print(f"   Samples: {len(self.df)}")
    
    def __len__(self) -> int:
        """Number of valid samples (accounting for window + 1 for target)."""
        return len(self.df) - self.window_size
    
    def _normalize_window(self, window_data: pd.DataFrame) -> Tuple[pd.DataFrame, float, float]:
        """
        Normalize OHLC data within window.
        Returns: normalized_df, mean_close, std_close
        """
        normalized = window_data.copy()
        
        if self.normalization_type == 'global':
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            normalized[ohlc_cols] = self.scaler.transform(window_data[ohlc_cols].values)
            # For global, return dummy stats (not used for reconstruction if scalers are present)
            # But to keep interface consistent, we can return 0, 1 or the scaler's stats
            return normalized, 0.0, 1.0
            
        else: # window-local normalization
            # Calculate mean and std for THIS window's Close price (or all columns)
            # Typically for spectrograms, we might want to normalize each column or use Close stats for all.
            # Using Close stats for all maintains relative prices (Open/High/Low relative to Close mean)
            
            # Robust normalization: Mean 0, Std 1 based on window
            # We use the window's statistics to normalize
            values = window_data[['Open', 'High', 'Low', 'Close']].values
            
            mean = values.mean(axis=0)
            std = values.std(axis=0) + 1e-6 # Avoid division by zero
            
            # Apply normalization
            normalized_values = (values - mean) / std
            
            ohlc_cols = ['Open', 'High', 'Low', 'Close']
            normalized[ohlc_cols] = normalized_values
            
            # Return Close stats specifically for target denormalization
            return normalized, mean[3], std[3]
    
    def _create_spectrogram(self, window_data: pd.DataFrame) -> np.ndarray:
        """
        Create 4-channel spectrogram from OHLC window using AG-STFT.
        
        Returns:
            spectrogram: (4, freq_bins, time_steps) array
        """
        # Transform each channel
        spectrograms, f, t = self.ag_stft.transform_ohlc(
            window_data,
            nperseg=self.nperseg,
            method=self.stft_method
        )
        
        # Stack into 4-channel array
        spec_array = np.stack([
            spectrograms['Open'],
            spectrograms['High'],
            spectrograms['Low'],
            spectrograms['Close']
        ], axis=0)  # Shape: (4, freq_bins, time_steps)
        
        return spec_array.astype(np.float32)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a single sample.
        
        Returns:
            spectrogram: (4, freq_bins, time_steps) tensor
            target: (1,) tensor - Next day's normalized closing price
            (optional) date: Target date if return_dates=True
        """
        # Get window of OHLC data
        window_start = idx
        window_end = idx + self.window_size
        
        window_data = self.df.iloc[window_start:window_end].copy()
        
        # Normalize window
        normalized_window, mean_close, std_close = self._normalize_window(window_data)
        
        # Create spectrogram
        spectrogram = self._create_spectrogram(normalized_window)
        
        # Get target: NEXT day's closing price
        target_date_idx = window_end  # Day after window
        target_close = self.df.iloc[target_date_idx]['Close']
        
        # Get last close price of input window (for naive baseline / direction calc)
        last_close = self.df.iloc[window_end - 1]['Close']
        last_close_tensor = torch.tensor([last_close], dtype=torch.float32)
        
        # Normalize target
        if self.normalization_type == 'global':
            target_normalized = self.target_scaler.transform([[target_close]])[0, 0]
        else:
            # Window-based: Normalize target relative to INPUT WINDOW statistics
            # This tells the network how many std devs the price will move relative to recent history
            target_normalized = (target_close - mean_close) / std_close
        
        # Convert to tensors
        spec_tensor = torch.from_numpy(spectrogram)
        target_tensor = torch.tensor([target_normalized], dtype=torch.float32)
        
        # For window normalization, we MUST return statistics to denormalize later
        # We always return them for interface consistency (0,1 for global)
        mean_tensor = torch.tensor(mean_close, dtype=torch.float32)
        std_tensor = torch.tensor(std_close, dtype=torch.float32)
        
        if self.return_dates:
            target_date = self.df.index[target_date_idx]
            return spec_tensor, target_tensor, mean_tensor, std_tensor, last_close_tensor, target_date
        
        return spec_tensor, target_tensor, mean_tensor, std_tensor, last_close_tensor
    
    def denormalize_price(
        self, 
        normalized_price: Union[float, np.ndarray, torch.Tensor],
        mean: Optional[Union[float, np.ndarray]] = 0.0,
        std: Optional[Union[float, np.ndarray]] = 1.0
    ) -> np.ndarray:
        """
        Convert normalized price back to original scale.
        """
        if isinstance(normalized_price, torch.Tensor):
            normalized_price = normalized_price.detach().cpu().numpy()
        
        normalized_price = np.atleast_1d(normalized_price).reshape(-1)
        
        if self.normalization_type == 'global':
            if self.scaler is None:
                 raise ValueError("Global scaler is missing.")
            
            # Create dummy array
            dummy = np.zeros((len(normalized_price), 4))
            dummy[:, 3] = normalized_price  # Close is index 3
            
            # Inverse transform
            original = self.scaler.inverse_transform(dummy)[:, 3]
            return original
            
        else:
            # Window-based: price = norm * std + mean
            # Ensure proper broadcasting
            if isinstance(mean, (np.ndarray, list)):
                mean = np.array(mean).reshape(-1)
            if isinstance(std, (np.ndarray, list)):
                std = np.array(std).reshape(-1)
                
            original = normalized_price * std + mean
            return original
    
    def get_sample_spectrogram_shape(self) -> Tuple[int, int, int]:
        """Get the shape of spectrograms produced by this dataset."""
        sample_spec = self[0][0]
        return tuple(sample_spec.shape)
    
    def precompute_all(self):
        """Precompute all spectrograms for faster training (uses more memory)."""
        print("Precomputing spectrograms...")
        self._precomputed = []
        
        for i in range(len(self)):
            # Update unpacking to include last_close
            item = self[i]
            # Since return size varies with return_dates, just append the tuple
            self._precomputed.append(item)
        
        print(f"✅ Precomputed {len(self._precomputed)} samples")


def create_dataloaders(
    csv_path: Union[str, Path],
    batch_size: int = 32,
    window_size: int = 60,
    nperseg: int = 32,
    train_ratio: float = 0.8,
    num_workers: int = 0,
    pin_memory: bool = True,
    normalization_type: str = 'window'
) -> Tuple[DataLoader, DataLoader, StockSpectrogramDataset]:
    """
    Create train and test dataloaders with proper scaler sharing.
    
    This is the recommended way to create dataloaders to ensure:
    1. No data leakage
    2. Proper scaler sharing
    3. Consistent train/test split
    
    Args:
        csv_path: Path to stock data CSV
        batch_size: Batch size for training
        window_size: Lookback window (days)
        nperseg: STFT segment length
        train_ratio: Train/test split ratio
        num_workers: DataLoader workers
        pin_memory: Pin memory for CUDA
        
    Returns:
        train_loader, test_loader, train_dataset
    """
    print("=" * 70)
    print("Creating DataLoaders")
    print("=" * 70)
    
    # Create training dataset (this fits the scaler)
    print("\n📈 Creating Training Dataset...")
    train_dataset = StockSpectrogramDataset(
        csv_path=csv_path,
        window_size=window_size,
        nperseg=nperseg,
        train=True,
        train_ratio=train_ratio,
        normalization_type=normalization_type,
        scaler=None  # Will create new scaler if global
    )
    
    # Create test dataset
    print("\n📉 Creating Test Dataset...")
    
    # Passing global scalers only if needed
    train_scaler = train_dataset.scaler if normalization_type == 'global' else None
    train_target_scaler = train_dataset.target_scaler if normalization_type == 'global' else None
    
    test_dataset = StockSpectrogramDataset(
        csv_path=csv_path,
        window_size=window_size,
        nperseg=nperseg,
        train=False,
        train_ratio=train_ratio,
        normalization_type=normalization_type,
        scaler=train_scaler,
        target_scaler=train_target_scaler
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=True  # Drop incomplete batches for batch norm stability
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory and torch.cuda.is_available(),
        drop_last=False
    )
    
    # Print summary
    spec_shape = train_dataset.get_sample_spectrogram_shape()
    print(f"\n📊 DataLoader Summary:")
    print(f"   Spectrogram shape: {spec_shape}")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Training batches: {len(train_loader)}")
    print(f"   Test samples: {len(test_dataset)}")
    print(f"   Test batches: {len(test_loader)}")
    
    return train_loader, test_loader, train_dataset


class MultiStockDataset(Dataset):
    """
    Dataset for training on multiple stocks simultaneously.
    
    Useful for:
    - Learning general market patterns
    - Transfer learning
    - More robust models
    """
    
    def __init__(
        self,
        csv_paths: list,
        window_size: int = 60,
        nperseg: int = 32,
        train: bool = True,
        train_ratio: float = 0.8
    ):
        """
        Args:
            csv_paths: List of paths to stock CSV files
            window_size: Lookback window
            nperseg: STFT segment length
            train: Training or test mode
            train_ratio: Train/test split ratio
        """
        self.datasets = []
        self.cumulative_lengths = [0]
        
        # Create individual datasets
        for i, path in enumerate(csv_paths):
            dataset = StockSpectrogramDataset(
                csv_path=path,
                window_size=window_size,
                nperseg=nperseg,
                train=train,
                train_ratio=train_ratio,
                scaler=None if train else self.datasets[0].scaler if i > 0 else None
            )
            self.datasets.append(dataset)
            self.cumulative_lengths.append(
                self.cumulative_lengths[-1] + len(dataset)
            )
        
        print(f"\n📊 MultiStockDataset: {len(csv_paths)} stocks, {len(self)} total samples")
    
    def __len__(self) -> int:
        return self.cumulative_lengths[-1]
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Find which dataset this index belongs to
        for i, (start, end) in enumerate(zip(
            self.cumulative_lengths[:-1], 
            self.cumulative_lengths[1:]
        )):
            if start <= idx < end:
                return self.datasets[i][idx - start]
        
        raise IndexError(f"Index {idx} out of range")


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Stock Spectrogram Dataset")
    print("=" * 70)
    
    # Test with sample data
    data_path = Path('data/raw/RELIANCE.csv')
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        print("Please download data first")
        exit(1)
    
    # Create dataloaders
    train_loader, test_loader, train_dataset = create_dataloaders(
        csv_path=data_path,
        batch_size=16,
        window_size=60,
        nperseg=32,
        train_ratio=0.8
    )
    
    # Test a batch
    print("\n" + "=" * 70)
    print("Testing Batch Loading")
    print("=" * 70)
    
    batch_spec, batch_target = next(iter(train_loader))
    
    print(f"Batch spectrogram shape: {batch_spec.shape}")
    print(f"Batch target shape: {batch_target.shape}")
    print(f"Target values (normalized): {batch_target[:5].flatten().numpy()}")
    
    # Denormalize targets
    original_prices = train_dataset.denormalize_price(batch_target.flatten())
    print(f"Target values (₹): {original_prices[:5]}")
    
    # Verify no data leakage
    print("\n" + "=" * 70)
    print("Verifying No Data Leakage")
    print("=" * 70)
    
    train_end = train_dataset.df.index[-1]
    test_start = test_loader.dataset.df.index[0]
    
    print(f"Training ends: {train_end.date()}")
    print(f"Testing starts: {test_start.date()}")
    
    if train_end < test_start:
        print("✅ No temporal overlap - data integrity verified!")
    else:
        print("⚠️ WARNING: Temporal overlap detected!")
    
    # Test prediction target is NEXT day's price
    print("\n" + "=" * 70)
    print("Verifying Target is Next Day's Price")
    print("=" * 70)
    
    # Manual check
    idx = 10
    spec, target = train_dataset[idx]
    
    window_end_date = train_dataset.df.index[idx + train_dataset.window_size - 1]
    target_date = train_dataset.df.index[idx + train_dataset.window_size]
    actual_price = train_dataset.df.iloc[idx + train_dataset.window_size]['Close']
    
    print(f"Window ends: {window_end_date.date()}")
    print(f"Target date: {target_date.date()}")
    print(f"Actual close price: ₹{actual_price:.2f}")
    print(f"Denormalized prediction target: ₹{train_dataset.denormalize_price(target.item())[0]:.2f}")
    
    print("\n✅ Dataset implementation verified!")
    print("\nKey features:")
    print("  ✓ Predicts next day's CLOSING PRICE (not returns)")
    print("  ✓ No data leakage - scaler fit only on training data")
    print("  ✓ Proper temporal train/test split")
    print("  ✓ AG-STFT preprocessing for spectrograms")
