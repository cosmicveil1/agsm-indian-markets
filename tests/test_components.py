"""
Unit Tests for AGSMNet Implementation
Tests all major components to ensure correctness.

Run with: python -m pytest tests/test_components.py -v
"""
import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


class TestAdaptiveGaussianSTFT:
    """Tests for AG-STFT implementation (Equations 1-3)."""
    
    def test_adaptive_sigma_computation(self):
        """Test Equation (1): σ(f) = α / (f + ε)"""
        from models.ag_stft import AdaptiveGaussianSTFT
        
        ag_stft = AdaptiveGaussianSTFT(alpha=1.0, epsilon=1e-6)
        
        # Low frequency should give high sigma (wide window)
        sigma_low = ag_stft.compute_adaptive_sigma(0.01)
        
        # High frequency should give low sigma (narrow window)
        sigma_high = ag_stft.compute_adaptive_sigma(0.5)
        
        assert sigma_low > sigma_high, \
            "Low frequency should have higher sigma than high frequency"
        
        # Check formula: σ = α / (f + ε)
        expected = 1.0 / (0.1 + 1e-6)
        actual = ag_stft.compute_adaptive_sigma(0.1)
        # Allow for clipping
        assert actual <= expected or actual == ag_stft.sigma_min
    
    def test_adaptive_window_shape(self):
        """Test Equation (2): window shape varies with frequency."""
        from models.ag_stft import AdaptiveGaussianSTFT
        
        ag_stft = AdaptiveGaussianSTFT(alpha=1.0)
        nperseg = 64
        
        # Create windows at different frequencies
        window_low = ag_stft.create_adaptive_gaussian_window(0.01, nperseg)
        window_high = ag_stft.create_adaptive_gaussian_window(0.4, nperseg)
        
        # Check window length
        assert len(window_low) == nperseg
        assert len(window_high) == nperseg
        
        # Check windows are normalized (sum to 1)
        assert abs(window_low.sum() - 1.0) < 1e-6
        assert abs(window_high.sum() - 1.0) < 1e-6
        
        # Low freq window should be wider (higher effective width)
        # Measure width by counting samples above 10% of max
        threshold = 0.1
        width_low = np.sum(window_low > threshold * window_low.max())
        width_high = np.sum(window_high > threshold * window_high.max())
        
        assert width_low >= width_high, \
            "Low frequency window should be wider than high frequency"
    
    def test_transform_output_shape(self):
        """Test AG-STFT transform produces correct output shape."""
        from models.ag_stft import AdaptiveGaussianSTFT
        
        ag_stft = AdaptiveGaussianSTFT()
        
        # Create test signal
        signal = np.random.randn(500)
        nperseg = 32
        
        # Transform
        f, t, Zxx = ag_stft.transform(signal, nperseg=nperseg, method='fast')
        
        # Check shapes
        assert len(f) == nperseg // 2 + 1  # One-sided spectrum
        assert Zxx.shape[0] == len(f)
        assert Zxx.shape[1] == len(t)
        
        # Check no NaN values
        assert not np.isnan(Zxx).any()
    
    def test_ohlc_transform(self):
        """Test OHLC data transformation."""
        from models.ag_stft import AdaptiveGaussianSTFT
        import pandas as pd
        
        ag_stft = AdaptiveGaussianSTFT()
        
        # Create mock OHLC data
        n_samples = 200
        dates = pd.date_range('2020-01-01', periods=n_samples)
        ohlc = pd.DataFrame({
            'Open': np.random.randn(n_samples).cumsum() + 100,
            'High': np.random.randn(n_samples).cumsum() + 102,
            'Low': np.random.randn(n_samples).cumsum() + 98,
            'Close': np.random.randn(n_samples).cumsum() + 100
        }, index=dates)
        
        # Transform
        specs, f, t = ag_stft.transform_ohlc(ohlc, nperseg=32)
        
        # Check all channels present
        assert 'Open' in specs
        assert 'High' in specs
        assert 'Low' in specs
        assert 'Close' in specs
        
        # Check shapes match
        for channel, spec in specs.items():
            assert spec.shape == specs['Open'].shape


class TestMambaComponents:
    """Tests for Mamba-based components."""
    
    def test_s4_block(self):
        """Test S4 block forward pass."""
        from models.mamba_simple import S4Block
        
        batch, seq_len, d_model = 4, 20, 64
        x = torch.randn(batch, seq_len, d_model)
        
        s4 = S4Block(d_model=d_model, d_state=32)
        output = s4(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_mamba_block_selection(self):
        """Test Mamba block with selection mechanism."""
        from models.mamba_simple import MambaBlock
        
        batch, seq_len, d_model = 4, 20, 64
        x = torch.randn(batch, seq_len, d_model)
        
        mamba = MambaBlock(d_model=d_model, d_state=32)
        output = mamba(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_ss2d_4_directions(self):
        """Test 2D-SSM scans in 4 directions."""
        from models.mamba_simple import SS2D
        
        batch, channels, height, width = 2, 32, 8, 10
        x = torch.randn(batch, channels, height, width)
        
        ss2d = SS2D(d_model=channels, d_state=16)
        output = ss2d(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_channel_attention(self):
        """Test Channel Attention module."""
        from models.mamba_simple import ChannelAttention
        
        batch, channels, height, width = 2, 64, 8, 8
        x = torch.randn(batch, channels, height, width)
        
        ca = ChannelAttention(channels=channels)
        output = ca(x)
        
        assert output.shape == x.shape
        # Output should be scaled input (same sign, similar magnitude)
        assert torch.abs(output.mean() - x.mean()) < torch.abs(x.mean())
    
    def test_vssm(self):
        """Test Vision State-Space Module."""
        from models.mamba_simple import VSSM
        
        batch, channels, height, width = 2, 32, 8, 10
        x = torch.randn(batch, channels, height, width)
        
        vssm = VSSM(d_model=channels, d_state=16)
        output = vssm(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
    
    def test_rssb(self):
        """Test Residual State-Space Block (Equations 4-5)."""
        from models.mamba_simple import RSSB
        
        batch, channels, height, width = 2, 32, 8, 8
        x = torch.randn(batch, channels, height, width)
        
        rssb = RSSB(channels=channels, d_state=16)
        output = rssb(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()


class TestAGSMNet:
    """Tests for complete AGSMNet model."""
    
    def test_agsm_net_forward(self):
        """Test full AGSMNet forward pass."""
        from models.agsm_net import AGSMNet
        
        batch = 4
        in_channels = 4  # OHLC
        freq_bins = 17
        time_steps = 20
        
        x = torch.randn(batch, in_channels, freq_bins, time_steps)
        
        model = AGSMNet(
            in_channels=in_channels,
            freq_bins=freq_bins,
            time_steps=time_steps,
            mfe_hidden=16,
            mfe_out=32,
            n_rssg_groups=1,
            n_rssb_per_group=1,
            d_state=16
        )
        
        output = model(x)
        
        assert output.shape == (batch, 1)
        assert not torch.isnan(output).any()
    
    def test_agsm_net_lite(self):
        """Test lightweight AGSMNet version."""
        from models.agsm_net import AGSMNetLite
        
        batch = 4
        x = torch.randn(batch, 4, 17, 20)
        
        model = AGSMNetLite(
            in_channels=4,
            freq_bins=17,
            time_steps=20,
            hidden_dim=32,
            n_mamba_layers=1
        )
        
        output = model(x)
        
        assert output.shape == (batch, 1)
        assert not torch.isnan(output).any()
    
    def test_model_gradient_flow(self):
        """Test gradients flow through entire model."""
        from models.agsm_net import AGSMNetLite
        
        batch = 2
        x = torch.randn(batch, 4, 17, 20, requires_grad=True)
        target = torch.randn(batch, 1)
        
        model = AGSMNetLite(hidden_dim=32, n_mamba_layers=1)
        
        output = model(x)
        loss = torch.nn.functional.mse_loss(output, target)
        loss.backward()
        
        # Check all parameters have gradients
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"


class TestDataset:
    """Tests for dataset implementation."""
    
    @pytest.fixture
    def mock_csv(self, tmp_path):
        """Create a mock stock CSV file."""
        import pandas as pd
        
        n_samples = 500
        dates = pd.date_range('2018-01-01', periods=n_samples)
        
        # Create realistic-looking price data
        base = 1000
        returns = np.random.randn(n_samples) * 0.02
        close = base * np.exp(np.cumsum(returns))
        
        df = pd.DataFrame({
            'Open': close * (1 + np.random.randn(n_samples) * 0.005),
            'High': close * (1 + np.abs(np.random.randn(n_samples) * 0.01)),
            'Low': close * (1 - np.abs(np.random.randn(n_samples) * 0.01)),
            'Close': close
        }, index=dates)
        
        csv_path = tmp_path / "test_stock.csv"
        df.to_csv(csv_path)
        
        return csv_path
    
    def test_dataset_creation(self, mock_csv):
        """Test dataset can be created."""
        from utils.dataset import StockSpectrogramDataset
        
        dataset = StockSpectrogramDataset(
            csv_path=mock_csv,
            window_size=30,
            nperseg=16,
            train=True,
            train_ratio=0.8
        )
        
        assert len(dataset) > 0
    
    def test_no_data_leakage(self, mock_csv):
        """Test train/test split has no temporal overlap."""
        from utils.dataset import StockSpectrogramDataset
        
        train_dataset = StockSpectrogramDataset(
            csv_path=mock_csv,
            window_size=30,
            nperseg=16,
            train=True,
            train_ratio=0.8
        )
        
        test_dataset = StockSpectrogramDataset(
            csv_path=mock_csv,
            window_size=30,
            nperseg=16,
            train=False,
            train_ratio=0.8,
            scaler=train_dataset.scaler
        )
        
        # Get date ranges
        train_end = train_dataset.df.index[-1]
        test_start = test_dataset.df.index[0]
        
        assert train_end < test_start, "Training data must end before test data starts"
    
    def test_scaler_sharing(self, mock_csv):
        """Test scaler is properly shared between train and test."""
        from utils.dataset import StockSpectrogramDataset
        
        train_dataset = StockSpectrogramDataset(
            csv_path=mock_csv,
            window_size=30,
            nperseg=16,
            train=True,
            train_ratio=0.8
        )
        
        # Test dataset must receive scaler
        with pytest.raises(ValueError):
            StockSpectrogramDataset(
                csv_path=mock_csv,
                window_size=30,
                nperseg=16,
                train=False,
                train_ratio=0.8,
                scaler=None  # Should raise error!
            )
        
        # This should work
        test_dataset = StockSpectrogramDataset(
            csv_path=mock_csv,
            window_size=30,
            nperseg=16,
            train=False,
            train_ratio=0.8,
            scaler=train_dataset.scaler
        )
        
        assert test_dataset.scaler is train_dataset.scaler
    
    def test_sample_shape(self, mock_csv):
        """Test samples have correct shape."""
        from utils.dataset import StockSpectrogramDataset
        
        window_size = 30
        nperseg = 16
        
        dataset = StockSpectrogramDataset(
            csv_path=mock_csv,
            window_size=window_size,
            nperseg=nperseg,
            train=True,
            train_ratio=0.8
        )
        
        spec, target = dataset[0]
        
        # Spectrogram: (4, freq_bins, time_steps)
        assert spec.dim() == 3
        assert spec.shape[0] == 4  # OHLC channels
        
        # Target: (1,)
        assert target.shape == (1,)
    
    def test_target_is_next_day_price(self, mock_csv):
        """Test target is next day's closing price (not returns)."""
        from utils.dataset import StockSpectrogramDataset
        
        dataset = StockSpectrogramDataset(
            csv_path=mock_csv,
            window_size=30,
            nperseg=16,
            train=True,
            train_ratio=0.8
        )
        
        # Get sample
        idx = 10
        spec, target = dataset[idx]
        
        # Get actual next day's close
        target_idx = idx + dataset.window_size
        actual_close = dataset.df.iloc[target_idx]['Close']
        
        # Denormalize prediction target
        predicted_close = dataset.denormalize_price(target.item())[0]
        
        # Should be close (allowing for floating point)
        assert abs(predicted_close - actual_close) < 0.1, \
            f"Target {predicted_close} doesn't match actual close {actual_close}"


class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.fixture
    def mock_csv(self, tmp_path):
        """Create mock data."""
        import pandas as pd
        
        n_samples = 300
        dates = pd.date_range('2020-01-01', periods=n_samples)
        close = 1000 + np.cumsum(np.random.randn(n_samples) * 10)
        
        df = pd.DataFrame({
            'Open': close + np.random.randn(n_samples) * 2,
            'High': close + np.abs(np.random.randn(n_samples) * 5),
            'Low': close - np.abs(np.random.randn(n_samples) * 5),
            'Close': close
        }, index=dates)
        
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path)
        return csv_path
    
    def test_training_loop(self, mock_csv):
        """Test one training iteration works."""
        from utils.dataset import create_dataloaders
        from models.agsm_net import AGSMNetLite
        
        train_loader, test_loader, train_dataset = create_dataloaders(
            csv_path=mock_csv,
            batch_size=8,
            window_size=30,
            nperseg=16,
            train_ratio=0.8
        )
        
        # Get dimensions
        spec, _ = train_dataset[0]
        freq_bins = spec.shape[1]
        time_steps = spec.shape[2]
        
        # Create model
        model = AGSMNetLite(
            freq_bins=freq_bins,
            time_steps=time_steps,
            hidden_dim=32,
            n_mamba_layers=1
        )
        
        # Training step
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        
        model.train()
        batch_spec, batch_target = next(iter(train_loader))
        
        optimizer.zero_grad()
        output = model(batch_spec)
        loss = criterion(output, batch_target)
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
        assert loss.item() < 1e10  # Loss should be reasonable


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
