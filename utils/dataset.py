"""
Dataset for stock price prediction with AG-STFT
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.ag_stft import AdaptiveGaussianSTFT

class StockSpectrogramDataset(Dataset):
    """
    Dataset that converts stock OHLC data to spectrograms using AG-STFT
    """
    
    def __init__(self, csv_path, window_size=20, nperseg=20, train=True, train_ratio=0.8, scaler=None):
        """
        Args:
            csv_path: Path to CSV file with OHLC data
            window_size: Number of days to look back
            nperseg: STFT segment length
            train: Whether this is training or test set
            train_ratio: Ratio of training data
            scaler: Shared scaler (use same normalization for train/test)
        """
        self.csv_path = Path(csv_path)
        self.window_size = window_size
        self.nperseg = nperseg
        
        # Load FULL data first
        df_full = pd.read_csv(csv_path, index_col=0, parse_dates=True)
        df_full = df_full.sort_index()
        
        # Split train/test
        split_idx = int(len(df_full) * train_ratio)
        
        # FIX: Normalize on FULL dataset, not just train
        if scaler is None:
            from sklearn.preprocessing import MinMaxScaler
            self.scaler = MinMaxScaler()
            # Fit on FULL data (prevents train/test mismatch)
            self.scaler.fit(df_full['Close'].values.reshape(-1, 1))
        else:
            self.scaler = scaler
        
        # Now split
        if train:
            self.df = df_full.iloc[:split_idx]
        else:
            self.df = df_full.iloc[split_idx:]
        
        # Initialize AG-STFT
        self.ag_stft = AdaptiveGaussianSTFT(fs=1.0, alpha=1.0)
        
        # Store stats for denormalization
        self.mean = self.scaler.data_min_[0]
        self.std = self.scaler.data_max_[0] - self.scaler.data_min_[0]
        
        print(f"{'Training' if train else 'Test'} set: {len(self.df)} rows")
        print(f"  Price range: ₹{self.df['Close'].min():.2f} - ₹{self.df['Close'].max():.2f}")
        
    def __len__(self):
        return len(self.df) - self.window_size
    
    def __getitem__(self, idx):
        """
        Returns:
            spectrogram: (4, freq_bins, time_steps) tensor
            target: Normalized next day's closing price
        """
        # Get window of OHLC data
        window_data = self.df.iloc[idx:idx + self.window_size]
        
        # Transform to spectrograms
        spectrograms, f, t = self.ag_stft.transform_ohlc(window_data, nperseg=self.nperseg)
        
        # Stack into 4-channel tensor (OHLC)
        spec_array = np.stack([
            spectrograms['Open'],
            spectrograms['High'],
            spectrograms['Low'],
            spectrograms['Close']
        ], axis=0)
        
        # Get target (next day's closing price, normalized)
        target_price = self.df.iloc[idx + self.window_size]['Close']
        target_normalized = self.scaler.transform([[target_price]])[0, 0]
        
        # Convert to tensors
        spec_tensor = torch.FloatTensor(spec_array)
        target_tensor = torch.FloatTensor([target_normalized])
        
        return spec_tensor, target_tensor
    

    def denormalize(self, normalized_price):
       """Convert normalized price back to original scale"""
       # Handle both single values and arrays
       if isinstance(normalized_price, (int, float)):
         return self.scaler.inverse_transform([[normalized_price]])[0, 0]
       else:
        # It's an array
        normalized_price = np.array(normalized_price).reshape(-1, 1)
        return self.scaler.inverse_transform(normalized_price).flatten()

# Test
if __name__ == "__main__":
    print("Testing StockSpectrogramDataset...")
    
    dataset = StockSpectrogramDataset(
        csv_path='data/raw/RELIANCE.csv',
        window_size=20,
        nperseg=20,
        train=True
    )
    
    print(f"\nDataset length: {len(dataset)}")
    
    # Get one sample
    spec, target = dataset[0]
    
    print(f"\nSample:")
    print(f"  Spectrogram shape: {spec.shape}")
    print(f"  Target shape: {target.shape}")
    print(f"  Target (normalized): {target.item():.4f}")
    print(f"  Target (original): ₹{dataset.denormalize(target.item()):.2f}")
    
    print("\n✅ Dataset working!")