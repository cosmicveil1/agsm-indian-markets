import torch
import torch.nn as nn
import torch.utils.checkpoint
from models.mamba_simple import RSSB

class MFE_LOB(nn.Module):
    """
    Multi-Scale Feature Extraction for LOB.
    Input: (B, 1, 40, 100)
    """
    def __init__(self, channels=32):
        super().__init__()
        # 1x1 Conv to expand channels
        self.conv1 = nn.Conv2d(1, channels, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        
        # 3x3 Conv for local patterns
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        return x

class AGSMNetLOB(nn.Module):
    """
    AGSMNet adapted for LOB Classification.
    """
    def __init__(
        self,
        in_channels: int = 1,
        dim: int = 64,
        state_dim: int = 32,
        num_classes: int = 3, # Up, Stationary, Down
        num_blocks: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Feature Extraction
        self.mfe = MFE_LOB(channels=dim)
        
        # Residual State-Space Groups (RSSG)
        self.rssg_blocks = nn.ModuleList([
            RSSB(
                channels=dim, 
                d_state=state_dim,
                dropout=dropout
            )
            for _ in range(num_blocks)
        ])
        
        # Global Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        # Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, x):
        # x: (B, 1, 40, 100)
        
        # Extract Features
        x = self.mfe(x) # (B, dim, 40, 100)
        
        # Apply RSSG Blocks (2D Scanning)
        for block in self.rssg_blocks:
            # Gradient Checkpointing to save VRAM (Compute vs Memory trade-off)
            if self.training:
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
            
        # Global Pooling
        x = self.avg_pool(x).flatten(1) # (B, dim)
        
        # Classification
        logits = self.classifier(x)
        return logits

def get_lob_model(device='cuda'):
    model = AGSMNetLOB().to(device)
    return model
