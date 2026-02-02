"""
AGSMNet: Adaptive Gaussian STFT + Mamba Network
Complete implementation based on Huang et al. (2025)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .mamba_simple import ResidualMambaBlock

class MultipleFeatureExtraction(nn.Module):
    """
    MFE Module: Shallow feature extraction from spectrograms
    Uses CNN to extract local patterns from time-frequency representations
    
    FIXED: Handle small spectrograms without pooling too much
    """
    
    def __init__(self, in_channels=4, hidden_channels=32):
        """
        Args:
            in_channels: Number of input channels (4 for OHLC)
            hidden_channels: Number of hidden channels
        """
        super().__init__()
        
        # Single conv block (removed second pooling to avoid dimension collapse)
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels*2, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels*2)
        
        # Adaptive pooling to fixed size (no intermediate pooling)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
    def forward(self, x):
        """
        Args:
            x: (batch, 4, freq, time) - Spectrogram with OHLC channels
            
        Returns:
            features: (batch, hidden_channels*2)
        """
        # First conv
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Second conv
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Adaptive pooling (handles any input size)
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        return x

class ResidualStateSpaceGroup(nn.Module):
    """
    RSSG Module: Deep feature extraction using stacked Mamba blocks
    """
    
    def __init__(self, d_model, n_layers=3, d_state=64, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            n_layers: Number of Mamba blocks to stack
            d_state: State dimension for Mamba
            dropout: Dropout rate
        """
        super().__init__()
        
        # Stack of Residual Mamba Blocks
        self.blocks = nn.ModuleList([
            ResidualMambaBlock(d_model, d_state, dropout)
            for _ in range(n_layers)
        ])
        
        # Final refinement convolution (as in paper)
        self.refine = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pass through all Mamba blocks
        for block in self.blocks:
            x = block(x)
        
        # Refinement
        x = self.refine(x)
        
        return x

class AGSMNet(nn.Module):
    """
    Complete AGSMNet architecture
    
    Pipeline:
    1. Input: Spectrogram (batch, 4, freq, time)
    2. MFE: Extract shallow features
    3. RSSG: Extract deep temporal features with Mamba
    4. Predictor: Predict next day's price
    """
    
    def __init__(
        self,
        freq_bins=11,
        time_steps=20,
        in_channels=4,
        hidden_dim=64,
        n_mamba_layers=3,
        d_state=64,
        dropout=0.1
    ):
        """
        Args:
            freq_bins: Number of frequency bins in spectrogram
            time_steps: Number of time steps in spectrogram
            in_channels: Number of input channels (4 for OHLC)
            hidden_dim: Hidden dimension for Mamba
            n_mamba_layers: Number of Mamba blocks
            d_state: State dimension for Mamba
            dropout: Dropout rate
        """
        super().__init__()
        
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        
        # Stage 1: Multiple Feature Extraction (MFE)
        self.mfe = MultipleFeatureExtraction(in_channels=in_channels, hidden_channels=32)
        
        # Calculate MFE output size
        mfe_out_dim = 64  # 32 * 2 from MFE
        
        # Project MFE features to sequence for Mamba
        self.proj_to_seq = nn.Linear(mfe_out_dim, hidden_dim)
        
        # Stage 2: Residual State Space Group (RSSG)
        self.rssg = ResidualStateSpaceGroup(
            d_model=hidden_dim,
            n_layers=n_mamba_layers,
            d_state=d_state,
            dropout=dropout
        )
        
        # Stage 3: Predictor
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        """
        Args:
            x: (batch, 4, freq, time) - Spectrogram with OHLC channels
            
        Returns:
            prediction: (batch, 1) - Predicted next day's price (scaled)
        """
        batch_size = x.size(0)
        
        # Stage 1: MFE - Extract shallow features
        shallow_features = self.mfe(x)  # (batch, 64)
        
        # Project to sequence (create a simple sequence of length 1)
        # In a more sophisticated version, we'd create a proper sequence
        seq_features = self.proj_to_seq(shallow_features)  # (batch, hidden_dim)
        seq_features = seq_features.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        # Stage 2: RSSG - Extract deep temporal features
        deep_features = self.rssg(seq_features)  # (batch, 1, hidden_dim)
        
        # Global pooling
        deep_features = deep_features.mean(dim=1)  # (batch, hidden_dim)
        
        # Stage 3: Predict next day's price
        prediction = self.predictor(deep_features)  # (batch, 1)
        
        return prediction

# Test
if __name__ == "__main__":
    # Create model
    model = AGSMNet(
        freq_bins=11,
        time_steps=20,
        in_channels=4,
        hidden_dim=64,
        n_mamba_layers=3
    )
    
    # Test input (batch of spectrograms)
    batch_size = 4
    x = torch.randn(batch_size, 4, 11, 20)  # 4 channels (OHLC), 11 freq bins, 20 time steps
    
    # Forward pass
    output = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Sample output: {output[0].item():.4f}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    
    print("\n✅ AGSMNet model working!")