"""
AGSMNet: Adaptive Gaussian STFT + Mamba Network
Complete Implementation based on Huang et al. (2025)

Architecture (Figure 1):
1. AG-STFT: Convert OHLC → 4-channel spectrogram (preprocessing)
2. MFE (Multiple Feature Extraction): Shallow CNN features
3. RSSG (Residual State-Space Group): Deep features with stacked RSSBs
4. Predictor: Final price prediction

Key Innovations:
- AG-STFT for adaptive time-frequency analysis
- 2D-SSM with 4-directional scanning
- VSSM + Channel Attention in RSSB
- End-to-end learning from raw OHLC to price prediction
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from .mamba_simple import RSSB, ResidualMambaBlock, ChannelAttention


class MultipleFeatureExtraction(nn.Module):
    """
    MFE Module (Section 3.3, Figure 3)
    
    Shallow feature extraction from spectrograms using CNN.
    
    Architecture:
    - Conv (3x3, 32 filters) → ReLU → MaxPool
    - Conv (3x3, 64 filters) → ReLU → MaxPool
    - AdaptiveAvgPool to fixed size
    
    Purpose: Extract local patterns and hierarchical features
    from time-frequency representations.
    """
    
    def __init__(
        self, 
        in_channels: int = 4,
        hidden_channels: int = 32,
        out_channels: int = 64,
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Number of input channels (4 for OHLC spectrograms)
            hidden_channels: Channels after first conv (paper: 32)
            out_channels: Channels after second conv (paper: 64)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(
            in_channels, hidden_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(
            hidden_channels, out_channels,
            kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Adaptive pooling for fixed output size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        self.dropout = nn.Dropout2d(dropout)
        self.out_channels = out_channels
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 4, freq, time) - OHLC spectrograms
            
        Returns:
            features: (batch, out_channels, 4, 4)
        """
        # First conv block
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        
        # Adaptive pooling to fixed size
        x = self.adaptive_pool(x)
        
        return x


class ResidualStateSpaceGroup(nn.Module):
    """
    RSSG Module (Section 3.4, Figure 1)
    
    Deep feature extraction using stacked Residual State-Space Blocks (RSSBs).
    
    Architecture:
    - Stack of n RSSBs
    - Final refinement convolution
    - Element-wise sum with input (global residual)
    
    Each RSSB contains:
    - VSSM (Vision State-Space Module) with 2D-SSM
    - Channel Attention
    - Residual connections with learnable scales
    """
    
    def __init__(
        self,
        channels: int,
        n_blocks: int = 3,
        d_state: int = 64,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            channels: Number of channels throughout the group
            n_blocks: Number of RSSB blocks to stack
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: VSSM expansion factor
            dropout: Dropout rate
        """
        super().__init__()
        
        self.channels = channels
        self.n_blocks = n_blocks
        
        # Stack of RSSB blocks
        self.blocks = nn.ModuleList([
            RSSB(
                channels=channels,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(n_blocks)
        ])
        
        # Final refinement convolution (mentioned in paper Section 3.4)
        self.refine = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.refine_bn = nn.BatchNorm2d(channels)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
            
        Returns:
            output: (batch, channels, height, width)
        """
        # Store input for global residual
        identity = x
        
        # Pass through stacked RSSB blocks
        for block in self.blocks:
            x = block(x)
        
        # Refinement convolution
        x = self.refine(x)
        x = self.refine_bn(x)
        
        # Global residual connection (element-wise sum)
        x = x + identity
        
        return x


class Predictor(nn.Module):
    """
    Predictor Module (Section 3.5)
    
    Final prediction layer for next-day stock price.
    
    Architecture:
    - Adaptive Average Pooling (robustness to input variations)
    - Flatten
    - Fully Connected layers → Price prediction
    """
    
    def __init__(
        self,
        in_channels: int,
        in_height: int,
        in_width: int,
        hidden_dim: int = 128,
        dropout: float = 0.2
    ):
        """
        Args:
            in_channels: Number of input channels
            in_height: Input feature map height
            in_width: Input feature map width
            hidden_dim: Hidden dimension for FC layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # Adaptive pooling for robustness
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # FC layers
        self.fc = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)  # Single output: next day's price
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
            
        Returns:
            prediction: (batch, 1) - Predicted next day's price
        """
        # Global average pooling
        x = self.avg_pool(x)  # (batch, channels, 1, 1)
        
        # Flatten
        x = x.view(x.size(0), -1)  # (batch, channels)
        
        # FC prediction
        prediction = self.fc(x)  # (batch, 1)
        
        return prediction


class AGSMNet(nn.Module):
    """
    AGSMNet: Adaptive Gaussian STFT + Mamba Network
    
    Complete architecture from Huang et al. (2025), Figure 1.
    
    Pipeline:
    1. Input: 4-channel spectrogram (batch, 4, freq, time)
       - Pre-computed using AG-STFT on OHLC data
    2. MFE: Extract shallow CNN features
    3. RSSG: Extract deep features using 2D-SSM with Mamba
    4. Predictor: Output next day's stock price
    
    The model learns to predict stock prices by:
    - Capturing time-frequency patterns via AG-STFT spectrograms
    - Extracting local features with CNN (MFE)
    - Modeling long-range dependencies with 2D-SSM (RSSG)
    - Aggregating features for final prediction
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        freq_bins: int = 17,
        time_steps: int = 20,
        # MFE parameters
        mfe_hidden: int = 32,
        mfe_out: int = 64,
        # RSSG parameters
        n_rssg_groups: int = 2,
        n_rssb_per_group: int = 3,
        d_state: int = 64,
        d_conv: int = 3,
        expand: int = 2,
        # Predictor parameters
        predictor_hidden: int = 128,
        # General
        dropout: float = 0.1
    ):
        """
        Args:
            in_channels: Input channels (4 for OHLC spectrograms)
            freq_bins: Number of frequency bins in spectrogram
            time_steps: Number of time steps in spectrogram
            mfe_hidden: MFE first conv channels
            mfe_out: MFE output channels
            n_rssg_groups: Number of RSSG groups to stack
            n_rssb_per_group: Number of RSSB blocks per group
            d_state: SSM state dimension
            d_conv: Convolution kernel size in RSSB
            expand: Expansion factor in VSSM
            predictor_hidden: Hidden dimension in predictor
            dropout: Dropout rate throughout the model
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.freq_bins = freq_bins
        self.time_steps = time_steps
        
        # Stage 1: Multiple Feature Extraction (MFE)
        self.mfe = MultipleFeatureExtraction(
            in_channels=in_channels,
            hidden_channels=mfe_hidden,
            out_channels=mfe_out,
            dropout=dropout
        )
        
        # Stage 2: Residual State-Space Groups (RSSG)
        # Stack multiple RSSG groups for deeper feature extraction
        self.rssg_groups = nn.ModuleList([
            ResidualStateSpaceGroup(
                channels=mfe_out,
                n_blocks=n_rssb_per_group,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout
            )
            for _ in range(n_rssg_groups)
        ])
        
        # Optional: Feature refinement after RSSG
        self.final_conv = nn.Conv2d(mfe_out, mfe_out, 1, bias=False)
        self.final_bn = nn.BatchNorm2d(mfe_out)
        
        # Stage 3: Predictor
        self.predictor = Predictor(
            in_channels=mfe_out,
            in_height=4,  # After MFE adaptive pooling
            in_width=4,
            hidden_dim=predictor_hidden,
            dropout=dropout
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights using He initialization for Conv, Xavier for Linear"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through AGSMNet.
        
        Args:
            x: (batch, 4, freq_bins, time_steps) - OHLC spectrograms
            
        Returns:
            prediction: (batch, 1) - Predicted next day's closing price
        """
        # Stage 1: MFE - Shallow feature extraction
        features = self.mfe(x)  # (batch, mfe_out, 4, 4)
        
        # Stage 2: RSSG - Deep feature extraction
        for rssg in self.rssg_groups:
            features = rssg(features)
        
        # Final refinement
        features = self.final_conv(features)
        features = self.final_bn(features)
        features = F.relu(features)
        
        # Stage 3: Predictor - Price prediction
        prediction = self.predictor(features)  # (batch, 1)
        
        return prediction
    
    def get_feature_maps(self, x: torch.Tensor) -> dict:
        """
        Get intermediate feature maps for visualization/debugging.
        
        Args:
            x: Input spectrogram
            
        Returns:
            Dictionary of intermediate features
        """
        features = {}
        
        # MFE features
        mfe_out = self.mfe(x)
        features['mfe'] = mfe_out
        
        # RSSG features
        rssg_out = mfe_out
        for i, rssg in enumerate(self.rssg_groups):
            rssg_out = rssg(rssg_out)
            features[f'rssg_{i}'] = rssg_out
        
        # Final features
        final = self.final_bn(self.final_conv(rssg_out))
        features['final'] = final
        
        return features


class AGSMNetLite(nn.Module):
    """
    Lightweight version of AGSMNet for smaller datasets or faster training.
    
    Uses simplified Mamba blocks instead of full 2D-SSM for efficiency
    while maintaining the core AG-STFT + Mamba paradigm.
    """
    
    def __init__(
        self,
        in_channels: int = 4,
        freq_bins: int = 17,
        time_steps: int = 20,
        hidden_dim: int = 64,
        n_mamba_layers: int = 2,
        d_state: int = 32,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Conv feature extraction
        self.mfe = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        
        # Project to hidden dim while preserving sequence length
        self.proj = nn.Linear(64, hidden_dim)
        
        # 1D Mamba blocks
        self.mamba_blocks = nn.ModuleList([
            ResidualMambaBlock(hidden_dim, d_state=d_state, dropout=dropout)
            for _ in range(n_mamba_layers)
        ])
        
        # Temporal aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Predictor with proper initialization
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1, bias=True)
        )
        
        # Initialize weights
        self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        
        # MFE: (batch, 4, freq, time) -> (batch, 64, freq, time)
        features = self.mfe(x)
        
        # Reshape to sequence: (batch, 64, freq, time) -> (batch, freq*time, 64)
        freq, time = features.shape[2:]
        features = features.permute(0, 2, 3, 1)  # (batch, freq, time, 64)
        features = features.reshape(batch, freq * time, 64)  # (batch, seq_len, 64)
        
        # Project to hidden dim: (batch, seq_len, 64) -> (batch, seq_len, hidden)
        features = self.proj(features)
        
        # Mamba blocks: preserve sequence structure
        for block in self.mamba_blocks:
            features = block(features)  # (batch, seq_len, hidden)
        
        # Global average pool: (batch, seq_len, hidden) -> (batch, hidden)
        features = self.global_pool(features.transpose(1, 2)).squeeze(2)
        
        # Predict: (batch, hidden) -> (batch, 1)
        prediction = self.predictor(features)
        
        return prediction
    
    def _init_weights(self):
        """Initialize weights properly to avoid vanishing gradients."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.uniform_(m.bias, -0.1, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def create_model(
    model_type: str = 'full',
    **kwargs
) -> nn.Module:
    """
    Factory function to create AGSMNet variants.
    
    Args:
        model_type: 'full' for AGSMNet, 'lite' for AGSMNetLite
        **kwargs: Model-specific arguments
        
    Returns:
        Model instance
    """
    if model_type == 'full':
        return AGSMNet(**kwargs)
    elif model_type == 'lite':
        return AGSMNetLite(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Testing AGSMNet Implementation")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    # Test dimensions (typical spectrogram shape)
    batch_size = 4
    in_channels = 4  # OHLC
    freq_bins = 17
    time_steps = 20
    
    x = torch.randn(batch_size, in_channels, freq_bins, time_steps).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Test full AGSMNet
    print("\n" + "-" * 50)
    print("Testing AGSMNet (Full)")
    print("-" * 50)
    
    model = AGSMNet(
        in_channels=in_channels,
        freq_bins=freq_bins,
        time_steps=time_steps,
        mfe_hidden=32,
        mfe_out=64,
        n_rssg_groups=2,
        n_rssb_per_group=2,
        d_state=32,
        predictor_hidden=64,
        dropout=0.1
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Forward pass
    with torch.no_grad():
        output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Sample prediction: {output[0].item():.4f}")
    
    # Test feature maps
    with torch.no_grad():
        features = model.get_feature_maps(x)
    print("\nFeature map shapes:")
    for name, feat in features.items():
        print(f"  {name}: {feat.shape}")
    
    # Test lite version
    print("\n" + "-" * 50)
    print("Testing AGSMNetLite")
    print("-" * 50)
    
    model_lite = AGSMNetLite(
        in_channels=in_channels,
        freq_bins=freq_bins,
        time_steps=time_steps,
        hidden_dim=64,
        n_mamba_layers=2,
        d_state=32
    ).to(device)
    
    lite_params = sum(p.numel() for p in model_lite.parameters())
    print(f"Total parameters: {lite_params:,}")
    
    with torch.no_grad():
        output_lite = model_lite(x)
    print(f"Output shape: {output_lite.shape}")
    
    # Compare
    print("\n" + "=" * 70)
    print("Comparison")
    print("=" * 70)
    print(f"AGSMNet (Full): {total_params:,} params")
    print(f"AGSMNetLite:    {lite_params:,} params ({lite_params/total_params*100:.1f}% of full)")
    
    print("\n✅ AGSMNet implementation complete!")
    print("\nArchitecture summary:")
    print("  1. MFE: Conv → BN → ReLU → Pool (×2) → AdaptivePool")
    print("  2. RSSG: Stack of RSSB blocks with 2D-SSM + Channel Attention")
    print("  3. Predictor: GlobalAvgPool → FC layers → Price")
