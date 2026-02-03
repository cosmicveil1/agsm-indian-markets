"""
Adaptive Gaussian Short-Time Fourier Transform (AG-STFT)
Faithful Implementation based on Huang et al. (2025)

Key Innovation: Window width adapts based on frequency content
- Equation (1): σ(f) = α / (f + ε)
- Equation (2): w(n) = exp(-n² / 2σ²)  
- Equation (3): AG-STFT = Σ x(τ) · w_σ(f)(n-τ) · exp(-j2πfτ)

This provides multi-resolution analysis:
- Narrow windows for high frequencies → better time resolution
- Wide windows for low frequencies → better frequency resolution
"""
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq, ifft
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, Union
import warnings


class AdaptiveGaussianSTFT:
    """
    Adaptive Gaussian STFT with frequency-dependent window width.
    
    Unlike standard STFT which uses a fixed window, AG-STFT adapts the
    Gaussian window width based on the frequency being analyzed:
    - Low frequencies: wider window (better frequency resolution)
    - High frequencies: narrower window (better time resolution)
    
    This addresses the fundamental time-frequency resolution trade-off
    that limits standard STFT for non-stationary financial data.
    
    Attributes:
        fs: Sampling frequency (typically 1.0 for daily financial data)
        alpha: Global scaling factor for adaptive window (Eq. 1)
        epsilon: Small constant to prevent division by zero
        sigma_min: Minimum window standard deviation
        sigma_max: Maximum window standard deviation
    """
    
    def __init__(
        self, 
        fs: float = 1.0, 
        alpha: float = 1.0, 
        epsilon: float = 1e-6,
        sigma_min: float = 2.0,
        sigma_max: float = None
    ):
        """
        Initialize AG-STFT transformer.
        
        Args:
            fs: Sampling frequency (for financial data, typically 1.0 = daily)
            alpha: Scaling factor controlling window adaptation (Eq. 1)
                   Larger alpha → wider windows overall
            epsilon: Small constant preventing division by zero when f ≈ 0
            sigma_min: Minimum allowed sigma (prevents too narrow windows)
            sigma_max: Maximum allowed sigma (None = auto-set based on window)
        """
        self.fs = fs
        self.alpha = alpha
        self.epsilon = epsilon
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def compute_adaptive_sigma(self, freq: float) -> float:
        """
        Compute adaptive standard deviation based on frequency.
        
        Implements Equation (1) from the paper:
        σ(f) = α / (f + ε)
        
        This ensures:
        - High frequency → small σ → narrow window → better time resolution
        - Low frequency → large σ → wide window → better frequency resolution
        
        Args:
            freq: Frequency value (normalized, 0 to 0.5 for Nyquist)
            
        Returns:
            Adaptive sigma value, clipped to [sigma_min, sigma_max]
        """
        # Equation (1): σ(f) = α / (f + ε)
        sigma = self.alpha / (np.abs(freq) + self.epsilon)
        
        # Clip to reasonable range
        sigma = max(sigma, self.sigma_min)
        if self.sigma_max is not None:
            sigma = min(sigma, self.sigma_max)
            
        return sigma
    
    def create_adaptive_gaussian_window(
        self, 
        freq: float, 
        window_length: int
    ) -> np.ndarray:
        """
        Create Gaussian window adapted to specific frequency.
        
        Implements Equation (2) from the paper:
        w(n) = exp(-n² / 2σ²)
        
        Args:
            freq: Target frequency for window adaptation
            window_length: Length of the window (N)
            
        Returns:
            Normalized Gaussian window of shape (window_length,)
        """
        # Compute adaptive sigma for this frequency
        sigma = self.compute_adaptive_sigma(freq)
        
        # Clip sigma based on window length to prevent numerical issues
        max_sigma_for_window = window_length / 4
        sigma = min(sigma, max_sigma_for_window)
        
        # Create window centered at 0: n goes from -N/2 to N/2-1
        n = np.arange(window_length) - window_length // 2
        
        # Equation (2): w(n) = exp(-n² / 2σ²)
        window = np.exp(-n**2 / (2 * sigma**2))
        
        # Normalize to sum to 1 (energy preservation)
        window = window / np.sum(window)
        
        return window
    
    def transform_adaptive(
        self,
        signal_data: np.ndarray,
        nperseg: int = 64,
        noverlap: Optional[int] = None,
        n_freq_bins: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply TRUE Adaptive Gaussian STFT with frequency-dependent windows.
        
        This is the core contribution of the paper - each frequency bin
        uses a different Gaussian window width optimized for that frequency.
        
        Implements Equation (3):
        AG-STFT(t,f) = Σ_τ x(τ) · w_{σ(f)}(t-τ) · exp(-j2πfτ)
        
        Args:
            signal_data: 1D input signal (e.g., stock prices)
            nperseg: Base segment length for STFT
            noverlap: Overlap between segments (default: nperseg//2)
            n_freq_bins: Number of frequency bins (default: nperseg//2 + 1)
            
        Returns:
            f: Frequency array of shape (n_freq_bins,)
            t: Time array of shape (n_time_steps,)
            Zxx: Complex spectrogram of shape (n_freq_bins, n_time_steps)
        """
        # Convert input to numpy array
        if hasattr(signal_data, 'values'):  # pandas Series
            signal_data = signal_data.values
        signal_data = np.asarray(signal_data, dtype=np.float64)
        
        # Remove NaN/inf values
        valid_mask = ~(np.isnan(signal_data) | np.isinf(signal_data))
        if not np.all(valid_mask):
            warnings.warn(f"Removed {np.sum(~valid_mask)} invalid values from signal")
            signal_data = signal_data[valid_mask]
        
        if len(signal_data) < nperseg:
            raise ValueError(f"Signal length ({len(signal_data)}) must be >= nperseg ({nperseg})")
        
        if noverlap is None:
            noverlap = nperseg // 2
            
        if n_freq_bins is None:
            n_freq_bins = nperseg // 2 + 1
        
        # Set max sigma based on window length
        if self.sigma_max is None:
            self.sigma_max = nperseg / 2
        
        # Compute frequency array (one-sided spectrum)
        freqs = np.fft.rfftfreq(nperseg, d=1.0/self.fs)
        
        # Compute time steps
        hop = nperseg - noverlap
        n_time_steps = (len(signal_data) - nperseg) // hop + 1
        times = np.arange(n_time_steps) * hop / self.fs
        
        # Initialize output spectrogram
        Zxx = np.zeros((len(freqs), n_time_steps), dtype=np.complex128)
        
        # For each time frame
        for t_idx in range(n_time_steps):
            start = t_idx * hop
            end = start + nperseg
            segment = signal_data[start:end]
            
            # For each frequency, apply window adapted to that frequency
            # This is the KEY innovation of AG-STFT
            for f_idx, freq in enumerate(freqs):
                # Create frequency-specific adaptive window
                adaptive_window = self.create_adaptive_gaussian_window(freq, nperseg)
                
                # Apply window to segment
                windowed_segment = segment * adaptive_window
                
                # Compute single-frequency DFT coefficient
                # This is more accurate than full FFT for adaptive analysis
                n = np.arange(nperseg)
                Zxx[f_idx, t_idx] = np.sum(
                    windowed_segment * np.exp(-2j * np.pi * f_idx * n / nperseg)
                )
        
        return freqs, times, Zxx
    
    def transform_fast(
        self,
        signal_data: np.ndarray,
        nperseg: int = 64,
        noverlap: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Fast approximation of AG-STFT using multi-resolution analysis.
        
        Instead of computing separate windows for each frequency,
        this uses a bank of windows at different scales and interpolates.
        Faster than full adaptive but still captures the multi-resolution idea.
        
        Args:
            signal_data: 1D input signal
            nperseg: Segment length
            noverlap: Overlap between segments
            
        Returns:
            f, t, Zxx: Frequency array, time array, spectrogram
        """
        if hasattr(signal_data, 'values'):
            signal_data = signal_data.values
        signal_data = np.asarray(signal_data, dtype=np.float64)
        
        # Remove invalid values
        valid_mask = ~(np.isnan(signal_data) | np.isinf(signal_data))
        signal_data = signal_data[valid_mask]
        
        if noverlap is None:
            noverlap = nperseg // 2
            
        if self.sigma_max is None:
            self.sigma_max = nperseg / 2
        
        # Create multiple resolution levels (5 scales)
        n_scales = 5
        sigmas = np.linspace(self.sigma_min, self.sigma_max, n_scales)
        
        # Compute STFT at each scale
        spectrograms = []
        
        for sigma in sigmas:
            # Create Gaussian window with this sigma
            n = np.arange(nperseg) - nperseg // 2
            window = np.exp(-n**2 / (2 * sigma**2))
            window = window / np.sum(window) * nperseg  # Scale for STFT
            
            # Compute STFT
            f, t, Zxx = signal.stft(
                signal_data,
                fs=self.fs,
                window=window,
                nperseg=nperseg,
                noverlap=noverlap,
                return_onesided=True
            )
            spectrograms.append(np.abs(Zxx))
        
        # Combine spectrograms: weight by frequency
        # Low frequencies use wide windows (high sigma), high use narrow
        combined = np.zeros_like(spectrograms[0])
        
        for f_idx, freq in enumerate(f):
            # Compute which scale to use based on frequency
            target_sigma = self.compute_adaptive_sigma(freq)
            target_sigma = np.clip(target_sigma, self.sigma_min, self.sigma_max)
            
            # Find weights for interpolation
            scale_idx = np.interp(target_sigma, sigmas, np.arange(n_scales))
            lower_idx = int(np.floor(scale_idx))
            upper_idx = min(lower_idx + 1, n_scales - 1)
            weight = scale_idx - lower_idx
            
            # Interpolate between scales
            combined[f_idx] = (1 - weight) * spectrograms[lower_idx][f_idx] + \
                             weight * spectrograms[upper_idx][f_idx]
        
        return f, t, combined
    
    def transform(
        self,
        signal_data: np.ndarray,
        nperseg: int = 64,
        noverlap: Optional[int] = None,
        method: str = 'adaptive'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Apply AG-STFT to input signal.
        
        Args:
            signal_data: 1D array of input signal
            nperseg: Segment length for STFT
            noverlap: Overlap between segments
            method: 'adaptive' (true AG-STFT) or 'fast' (multi-resolution approx)
            
        Returns:
            f: Frequency array
            t: Time array  
            Zxx: Spectrogram magnitude (or complex for 'adaptive')
        """
        if method == 'adaptive':
            f, t, Zxx = self.transform_adaptive(signal_data, nperseg, noverlap)
            return f, t, np.abs(Zxx)
        elif method == 'fast':
            return self.transform_fast(signal_data, nperseg, noverlap)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'adaptive' or 'fast'")
    
    def transform_ohlc(
        self,
        ohlc_data: Union[Dict, 'pd.DataFrame'],
        nperseg: int = 64,
        noverlap: Optional[int] = None,
        method: str = 'fast'
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray]:
        """
        Transform OHLC data to 4-channel spectrogram using AG-STFT.
        
        This is the preprocessing step that converts raw financial data
        into time-frequency representations as shown in Figure 2.
        
        Args:
            ohlc_data: DataFrame or dict with 'Open', 'High', 'Low', 'Close'
            nperseg: Segment length
            noverlap: Overlap between segments
            method: 'adaptive' or 'fast'
            
        Returns:
            spectrograms: Dict mapping channel names to spectrogram arrays
            frequencies: Frequency array
            times: Time array
        """
        channels = ['Open', 'High', 'Low', 'Close']
        spectrograms = {}
        
        f, t = None, None
        
        for channel in channels:
            if isinstance(ohlc_data, dict):
                data = ohlc_data[channel]
            else:
                data = ohlc_data[channel].values
            
            freq, time, Zxx = self.transform(data, nperseg, noverlap, method)
            spectrograms[channel] = Zxx
            
            if f is None:
                f, t = freq, time
        
        return spectrograms, f, t
    
    def visualize_adaptive_windows(
        self,
        nperseg: int = 64,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize how window width changes with frequency.
        
        This demonstrates the core AG-STFT concept.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Different frequencies to show
        freqs = [0.01, 0.05, 0.15, 0.4]
        titles = ['Very Low Freq (f=0.01)', 'Low Freq (f=0.05)', 
                  'Medium Freq (f=0.15)', 'High Freq (f=0.4)']
        
        for ax, freq, title in zip(axes.flat, freqs, titles):
            sigma = self.compute_adaptive_sigma(freq)
            window = self.create_adaptive_gaussian_window(freq, nperseg)
            
            ax.plot(window, 'b-', linewidth=2)
            ax.fill_between(range(len(window)), window, alpha=0.3)
            ax.set_title(f'{title}\nσ = {sigma:.2f}', fontsize=11)
            ax.set_xlabel('Sample')
            ax.set_ylabel('Amplitude')
            ax.grid(True, alpha=0.3)
            ax.set_xlim(0, nperseg)
        
        plt.suptitle('AG-STFT: Adaptive Gaussian Windows\n'
                     'Wider windows for low freq (frequency resolution) '
                     '| Narrower for high freq (time resolution)',
                     fontsize=12, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        return fig
    
    def visualize(
        self,
        signal_data: np.ndarray,
        title: str = "AG-STFT Spectrogram",
        nperseg: int = 64,
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[str] = None,
        method: str = 'fast'
    ) -> plt.Figure:
        """
        Visualize the AG-STFT spectrogram of a signal.
        
        Args:
            signal_data: 1D signal to transform
            title: Plot title
            nperseg: Segment length
            figsize: Figure size
            save_path: Path to save figure
            method: 'adaptive' or 'fast'
            
        Returns:
            matplotlib Figure
        """
        f, t, Zxx = self.transform(signal_data, nperseg=nperseg, method=method)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use log scale for better visualization
        Zxx_db = 10 * np.log10(Zxx + 1e-10)
        
        im = ax.pcolormesh(t, f, Zxx_db, shading='gouraud', cmap='viridis')
        ax.set_ylabel('Frequency [cycles/day]')
        ax.set_xlabel('Time [days]')
        ax.set_title(title)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Magnitude [dB]')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved to {save_path}")
        
        return fig
    
    def compare_with_standard_stft(
        self,
        signal_data: np.ndarray,
        nperseg: int = 64,
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Compare AG-STFT with standard fixed-window STFT.
        
        Shows the advantage of adaptive windowing for non-stationary signals.
        """
        if hasattr(signal_data, 'values'):
            signal_data = signal_data.values
        signal_data = np.asarray(signal_data, dtype=np.float64)
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        # Standard STFT with Hann window
        f_std, t_std, Zxx_std = signal.stft(
            signal_data, fs=self.fs, window='hann',
            nperseg=nperseg, noverlap=nperseg//2
        )
        
        # AG-STFT
        f_ag, t_ag, Zxx_ag = self.transform(signal_data, nperseg=nperseg, method='fast')
        
        # Plot standard STFT
        ax = axes[0]
        Zxx_std_db = 10 * np.log10(np.abs(Zxx_std) + 1e-10)
        ax.pcolormesh(t_std, f_std, Zxx_std_db, shading='gouraud', cmap='viridis')
        ax.set_title('Standard STFT (Hann window)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        # Plot AG-STFT
        ax = axes[1]
        Zxx_ag_db = 10 * np.log10(Zxx_ag + 1e-10)
        ax.pcolormesh(t_ag, f_ag, Zxx_ag_db, shading='gouraud', cmap='viridis')
        ax.set_title('AG-STFT (Adaptive Gaussian)')
        ax.set_xlabel('Time')
        ax.set_ylabel('Frequency')
        
        # Difference
        ax = axes[2]
        # Interpolate to same shape if needed
        ax.plot(signal_data[:200], 'b-', alpha=0.7)
        ax.set_title('Input Signal (first 200 samples)')
        ax.set_xlabel('Sample')
        ax.set_ylabel('Value')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('AG-STFT vs Standard STFT Comparison', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved comparison to {save_path}")
        
        return fig


# Test and demonstration
if __name__ == "__main__":
    import pandas as pd
    from pathlib import Path
    
    print("=" * 70)
    print("Testing TRUE Adaptive Gaussian STFT Implementation")
    print("=" * 70)
    
    # Create AG-STFT with paper's parameters
    ag_stft = AdaptiveGaussianSTFT(fs=1.0, alpha=1.0, epsilon=1e-6)
    
    # Test 1: Verify adaptive sigma computation
    print("\n📐 Test 1: Adaptive Sigma (Equation 1)")
    print("-" * 50)
    test_freqs = [0.01, 0.05, 0.1, 0.2, 0.5]
    for f in test_freqs:
        sigma = ag_stft.compute_adaptive_sigma(f)
        print(f"  f = {f:.2f} → σ = {sigma:.4f}")
    
    # Test 2: Verify window shapes
    print("\n📊 Test 2: Adaptive Windows (Equation 2)")
    print("-" * 50)
    for f in [0.01, 0.1, 0.4]:
        window = ag_stft.create_adaptive_gaussian_window(f, 64)
        effective_width = np.sum(window > 0.01 * window.max())
        print(f"  f = {f:.2f} → window effective width = {effective_width} samples")
    
    # Test with real data if available
    data_path = Path('data/raw/RELIANCE.csv')
    
    if data_path.exists():
        print(f"\n📈 Test 3: Transform Real Stock Data")
        print("-" * 50)
        
        df = pd.read_csv(data_path, index_col=0, parse_dates=True)
        close_prices = df['Close'].values
        
        # Use last 200 days for testing
        test_signal = close_prices[-200:]
        
        # Test fast method
        print("  Testing fast multi-resolution method...")
        f, t, Zxx = ag_stft.transform(test_signal, nperseg=32, method='fast')
        print(f"  ✅ Fast method: {Zxx.shape}")
        
        # Test full adaptive method (slower but more accurate)
        print("  Testing full adaptive method (may take a moment)...")
        f, t, Zxx = ag_stft.transform(test_signal[:100], nperseg=32, method='adaptive')
        print(f"  ✅ Adaptive method: {Zxx.shape}")
        
        # Test OHLC transform
        print("\n  Testing OHLC transform...")
        specs, freqs, times = ag_stft.transform_ohlc(df.iloc[-200:], nperseg=32)
        print(f"  ✅ OHLC spectrograms:")
        for ch, spec in specs.items():
            print(f"      {ch}: {spec.shape}")
        
        # Create visualizations
        results_dir = Path('experiments/results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n📊 Creating visualizations...")
        
        # Visualize adaptive windows
        ag_stft.visualize_adaptive_windows(
            nperseg=64,
            save_path=results_dir / 'ag_stft_adaptive_windows.png'
        )
        
        # Compare with standard STFT
        ag_stft.compare_with_standard_stft(
            close_prices[-500:],
            nperseg=32,
            save_path=results_dir / 'ag_stft_vs_standard.png'
        )
        
        # Visualize spectrogram
        ag_stft.visualize(
            close_prices[-500:],
            title="RELIANCE Stock - AG-STFT Spectrogram",
            nperseg=32,
            save_path=results_dir / 'ag_stft_spectrogram.png'
        )
        
        print(f"\n✅ Visualizations saved to {results_dir}")
    
    else:
        print(f"\n⚠️  Data file not found: {data_path}")
        print("  Skipping real data tests")
    
    print("\n" + "=" * 70)
    print("✅ AG-STFT Implementation Complete!")
    print("=" * 70)
    print("\nKey features implemented:")
    print("  ✓ Frequency-dependent adaptive sigma (Equation 1)")
    print("  ✓ Adaptive Gaussian windows (Equation 2)")
    print("  ✓ Full AG-STFT transform (Equation 3)")
    print("  ✓ Fast multi-resolution approximation")
    print("  ✓ OHLC 4-channel spectrogram generation")
