"""
Data preprocessing utilities
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import pickle

class StockDataset:
    """
    Dataset class for stock price data
    """
    
    def __init__(self, data_path, window_size=20, train_ratio=0.8):
        """
        Args:
            data_path: Path to CSV file
            window_size: Number of days to look back
            train_ratio: Ratio of training data
        """
        self.data_path = Path(data_path)
        self.window_size = window_size
        self.train_ratio = train_ratio
        
        # Load data
        self.df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        self.df = self.df.sort_index()
        
        # Split train/test
        split_idx = int(len(self.df) * train_ratio)
        self.train_df = self.df.iloc[:split_idx]
        self.test_df = self.df.iloc[split_idx:]
        
        # Fit scaler on training data
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_df[['Open', 'High', 'Low', 'Close']])
        
    def create_sequences(self, df, target_col='Close'):
        """
        Create sequences for time series prediction
        
        Args:
            df: DataFrame with OHLC data
            target_col: Column to predict
            
        Returns:
            X: Input sequences (N, window_size, 4) - OHLC
            y: Target values (N,)
        """
        # Normalize
        ohlc = df[['Open', 'High', 'Low', 'Close']].values
        ohlc_scaled = self.scaler.transform(ohlc)
        
        X, y = [], []
        
        for i in range(len(df) - self.window_size):
            # Input: window_size days of OHLC
            X.append(ohlc_scaled[i:i+self.window_size])
            
            # Target: next day's closing price
            y.append(ohlc_scaled[i+self.window_size, 3])  # Close is index 3
        
        return np.array(X), np.array(y)
    
    def get_train_data(self):
        """Get training sequences"""
        return self.create_sequences(self.train_df)
    
    def get_test_data(self):
        """Get test sequences"""
        return self.create_sequences(self.test_df)
    
    def inverse_transform_price(self, scaled_price):
        """
        Convert scaled price back to original scale
        
        Args:
            scaled_price: Scaled closing price
            
        Returns:
            Original price
        """
        # Create dummy array with zeros for O, H, L and our price for C
        dummy = np.zeros((1, 4))
        dummy[0, 3] = scaled_price
        
        # Inverse transform
        original = self.scaler.inverse_transform(dummy)
        
        return original[0, 3]

# Test
if __name__ == "__main__":
    dataset = StockDataset('data/raw/RELIANCE.csv', window_size=20)
    
    X_train, y_train = dataset.get_train_data()
    X_test, y_test = dataset.get_test_data()
    
    print(f"Training set: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set: X={X_test.shape}, y={y_test.shape}")
    print(f"Sample input shape: {X_train[0].shape}")
    print(f"Sample input:\n{X_train[0][:5]}")  # First 5 days
    print(f"Sample target: {y_train[0]}")
```

**END OF DAY 1 CHECKPOINT:**
```
✅ Data downloaded for 12 Indian stocks
✅ AG-STFT implementation working
✅ Data preprocessing pipeline ready
✅ Visualization working

Commit: "Day 1: Data collection and AG-STFT implementation"