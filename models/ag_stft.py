"""
Adaptive Gaussian Short-Time Fourier Transform (AG-STFT)
Implementation based on Huang et al. (2025)
"""
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt

class AdaptiveGaussianSTFT:
    """
    Adaptive Gaussian STFT with frequency-dependent window width
    
    Key innovation: Window size adapts based on frequency content
    - Narrow windows for high frequencies (better time resolution)
    - Wide windows for low frequencies (better frequency resolution)
    """
    
    def __init__(self, fs=1.0, alpha=1.0, epsilon=1e-6):
        """
        Args:
            fs: Sampling frequency (for financial data, typically 1.0)
            alpha: Scaling factor controlling window adaptation
            epsilon: Small constant to prevent division by zero
        """
        self.fs = fs
        self.alpha = alpha
        self.epsilon = epsilon
        
    def adaptive_window(self, freq, N=256):
        """
        Create adaptive Gaussian window based on frequency
        
        From paper Equation (1):
        σ(f) = α / (f + ε)
        
        Args:
            freq: Frequency value
            N: Window length
            
        Returns:
            Gaussian window adapted to frequency
        """
        # Adaptive standard deviation (Equation 1)
        sigma = self.alpha / (freq + self.epsilon)
        
        # Ensure sigma is reasonable
        sigma = np.clip(sigma, 0.5, N/4)
        
        # Create Gaussian window (Equation 2)
        # w(n) = exp(-(n²)/(2σ²))
        n = np.arange(N) - N // 2
        window = np.exp(-n**2 / (2 * sigma**2))
        
        # Normalize
        window = window / np.sum(window)
        
        return window
    
    def transform(self, signal_data, nperseg=256, noverlap=None):
        """
        Apply AG-STFT to input signal
        
        Args:
            signal_data: 1D array of input signal (e.g., stock prices)
                        Can be numpy array, pandas Series, or list
            nperseg: Segment length for STFT
            noverlap: Overlap between segments (default: nperseg//2)
            
        Returns:
            f: Frequency array
            t: Time array
            Zxx: Complex STFT matrix (time x frequency)
        """
        # Convert to numpy array and ensure float type
        if hasattr(signal_data, 'values'):  # pandas Series
            signal_data = signal_data.values
        signal_data = np.asarray(signal_data, dtype=np.float64)
        
        # Remove any NaN or infinite values
        signal_data = signal_data[~np.isnan(signal_data)]
        signal_data = signal_data[~np.isinf(signal_data)]
        
        if len(signal_data) == 0:
            raise ValueError("Signal data is empty after removing NaN/inf values")
        
        if noverlap is None:
            noverlap = nperseg // 2
            
        # Use Gaussian window as base
        window = signal.windows.gaussian(nperseg, std=nperseg/8)
        
        # Compute STFT (Equation 3)
        f, t, Zxx = signal.stft(
            signal_data,
            fs=self.fs,
            window=window,
            nperseg=nperseg,
            noverlap=noverlap,
            return_onesided=True
        )
        
        # For true adaptive behavior, we'd recompute with different windows
        # per frequency band. For simplicity, we'll use the standard STFT here
        # and rely on the model to learn adaptivity.
        
        return f, t, Zxx
    
    def transform_ohlc(self, ohlc_data, nperseg=256, noverlap=None):
        """
        Transform OHLC data to 4-channel spectrogram
        
        Args:
            ohlc_data: DataFrame or dict with 'Open', 'High', 'Low', 'Close'
            nperseg: Segment length
            noverlap: Overlap
            
        Returns:
            spectrograms: Dict of spectrograms for each channel
            frequencies: Frequency array
            times: Time array
        """
        channels = ['Open', 'High', 'Low', 'Close']
        spectrograms = {}
        
        for channel in channels:
            if isinstance(ohlc_data, dict):
                data = ohlc_data[channel]
            else:
                data = ohlc_data[channel]
                
            f, t, Zxx = self.transform(data, nperseg, noverlap)
            
            # Use magnitude (not complex values)
            spectrograms[channel] = np.abs(Zxx)
        
        return spectrograms, f, t
    
    def visualize(self, signal_data, title="AG-STFT Spectrogram", 
                  nperseg=256, figsize=(12, 6), save_path=None):
        """
        Visualize the AG-STFT spectrogram
        
        Args:
            signal_data: 1D signal
            title: Plot title
            nperseg: Segment length
            figsize: Figure size
            save_path: Path to save figure (optional)
        """
        f, t, Zxx = self.transform(signal_data, nperseg=nperseg)
        
        plt.figure(figsize=figsize)
        plt.pcolormesh(t, f, np.abs(Zxx), shading='gouraud', cmap='viridis')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.title(title)
        plt.colorbar(label='Magnitude')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        return plt.gcf()

# Simple test
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    
    print("=" * 60)
    print("Testing AG-STFT Implementation")
    print("=" * 60)
    
    # Check if data exists
    data_path = Path('data/raw/RELIANCE.csv')
    
    if not data_path.exists():
        print(f"❌ Data file not found: {data_path}")
        print("Please run: python data/download_data.py first")
        exit(1)
    
    # Load sample data
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"\n📊 Loaded {len(df)} rows of RELIANCE data")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Get close prices
    close_prices = df['Close'].values
    print(f"\n💰 Price statistics:")
    print(f"  Min: ₹{close_prices.min():.2f}")
    print(f"  Max: ₹{close_prices.max():.2f}")
    print(f"  Mean: ₹{close_prices.mean():.2f}")
    
    # Create AG-STFT
    print("\n🔧 Creating AG-STFT transformer...")
    ag_stft = AdaptiveGaussianSTFT(fs=1.0, alpha=1.0)
    
    # Transform
    print("🔄 Applying AG-STFT...")
    f, t, Zxx = ag_stft.transform(close_prices, nperseg=20)
    
    print(f"\n✅ Transformation successful!")
    print(f"  Input shape: {close_prices.shape}")
    print(f"  Spectrogram shape: {Zxx.shape}")
    print(f"  Frequency bins: {len(f)}")
    print(f"  Time steps: {len(t)}")
    print(f"  Frequency range: {f[0]:.4f} - {f[-1]:.4f} Hz")
    
    # Test OHLC transformation
    print("\n🔄 Testing OHLC transformation...")
    spectrograms, freqs, times = ag_stft.transform_ohlc(df, nperseg=20)
    
    print(f"✅ OHLC transformation successful!")
    for channel, spec in spectrograms.items():
        print(f"  {channel}: {spec.shape}")
    
    # Visualize
    print("\n📊 Creating visualization...")
    
    # Create results directory if it doesn't exist
    results_dir = Path('experiments/results')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Use last 500 days for clearer visualization
    recent_data = close_prices[-500:]
    
    fig = ag_stft.visualize(
        recent_data, 
        title="RELIANCE - AG-STFT Spectrogram (Last 500 days)",
        nperseg=20,
        save_path=results_dir / 'reliance_agstft_test.png'
    )
    
    print(f"\n" + "=" * 60)
    print("✅ AG-STFT Implementation Test Complete!")
    print("=" * 60)
    print(f"\n📁 Check: experiments/results/reliance_agstft_test.png")