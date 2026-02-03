import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Union

class FI2010Dataset(Dataset):
    """
    Dataset for FI-2010 Benchmark Limit Order Book Data.
    
    Data Format:
    - 147 columns original
    - Columns 0-39: Raw LOB (AskP, AskV, BidP, BidV) for 10 levels
    - Columns 144-146: Labels for varying horizons (k=10, 50, 100)
    
    We map the 100-step window into an 'image':
    - Height: 40 (Price Levels + Volumes)
    - Width: 100 (Time steps)
    """
    
    def __init__(
        self, 
        data_path: Union[str, Path], 
        train: bool = True, 
        T: int = 100, 
        k: int = 10, # Prediction horizon (10, 50, 100)
        normalization: str = 'zscore'
    ):
        """
        Args:
            data_path: Path to the dataset file (e.g. 'Train_Dst_NoAuction_ZScore_CF_7.txt')
            train: If True, uses the train split (first 7 days), else test (last 3 days)
                   Note: For FI-2010, usually files are pre-split.
            T: Window size (Time steps) typically 100
            k: Prediction horizon (used to select correct label column)
            normalization: 'zscore' or 'none' (Dataset is often already z-scored)
        """
        self.data_path = Path(data_path)
        self.T = T
        self.k = k
        self.train = train
        
        # Load data
        # Assuming the standard format where rows are features, cols are time (transposed)
        # Or rows are time, cols are features. 
        # Standard benchmark files: Rows = Features (147), Cols = Time
        print(f"Loading LOB data from {self.data_path}...")
        try:
            # Try efficient load first? No, pandas is safer for unknown CSV formats
            # The error suggests a header/index issue.
            df = pd.read_csv(self.data_path)
            
            # Check if first column is an index (often Unnamed: 0 in pandas exports)
            if df.columns[0].startswith('Unnamed') or df.columns[0] == '':
                df = df.iloc[:, 1:]
                
            # If that failed and we still have object/string types, force conversion
            # Sometimes headers are read as data if header=None was needed but we removed it?
            # Actually, standardizing on pandas auto-detect is best.
            
            # Use values
            self.data = df.values.astype(np.float32)
            
        except Exception as e:
            print(f"Pandas load failed: {e}")
            # Fallback to header=None if the above failed (maybe no header?)
            df = pd.read_csv(self.data_path, header=None)
            self.data = df.values
        if self.data.shape[1] > self.data.shape[0]:
            # Rows are features, Cols are time. This is standard FI-2010.
            self.data = self.data.T
            # Now Rows=Time (N), Cols=Features (147ish)
        else:
            # Already N x 147
            pass
            
        print(f"Data Loaded. Shape: {self.data.shape}")
        
        # Split features and labels
        # First 40 columns are the raw LOB (10 levels * 4 values)
        self.features = self.data[:, :40]
        
        # Label columns:
        # -5: k=10
        # -4: k=20 
        # -3: k=30
        # -2: k=50
        # -1: k=100
        # Map k to column index relative to end
        k_map = {10: -5, 20: -4, 30: -3, 50: -2, 100: -1}
        if k not in k_map:
            raise ValueError(f"k must be one of {list(k_map.keys())}")
            
        self.labels = self.data[:, k_map[k]]
        
        # Remap labels from {1, 2, 3} to {0, 1, 2} for PyTorch CrossEntropy
        # 1(Up) -> 0, 2(Stationary) -> 1, 3(Down) -> 2
        self.labels = self.labels - 1
        
        # Extract samples
        self.samples = self._prepare_samples()
        print(f"Prepared {len(self.samples)} samples (T={T}, k={k})")

    def _prepare_samples(self):
        """
        Create indices for valid windows.
        We need T steps.
        """
        n_samples = self.features.shape[0] - self.T + 1
        # Use simple stride of 1
        indices = np.arange(n_samples)
        return indices

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns:
            x: (1, 40, 100) Tensor - The LOB Snapshot Image
            y: (1,) Tensor - The Class Label (0, 1, 2)
        """
        start = idx
        end = idx + self.T
        
        # Extract window
        # Shape: (100, 40)
        window = self.features[start:end]
        
        # Get label at the END of the window (prediction for T+k)
        # Note: In FI-2010, the label at row 't' is the target for horizon k.
        # So we usually take label at index 'end-1' which corresponds to the last state
        label = self.labels[end - 1]
        
        # Reshape to Image format: (Channels, Height, Width)
        # Here: (1, 40, 100) -> (1, PriceLevels, Time)
        # Transpose (100, 40) -> (40, 100)
        x = window.T
        x = x[np.newaxis, :, :] # Add channel dim
        
        return torch.FloatTensor(x), torch.LongTensor([int(label)])

def create_lob_loader(path, batch_size=32, shuffle=True):
    dataset = FI2010Dataset(path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
