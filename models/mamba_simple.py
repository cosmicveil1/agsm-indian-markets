"""
Simplified Mamba-like State Space Model
Based on S4 and Mamba architectures but simplified for implementation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class S4Block(nn.Module):
    """
    Simplified S4 (Structured State Space) block
    
    This is a simplified version that captures the core ideas:
    - State space parameterization
    - Efficient sequence modeling
    - Long-range dependencies
    """
    
    def __init__(self, d_model, d_state=64, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_state: State dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # State space parameters (learnable)
        self.A = nn.Parameter(torch.randn(d_state, d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model))
        self.C = nn.Parameter(torch.randn(d_model, d_state))
        self.D = nn.Parameter(torch.randn(d_model))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device)
        
        outputs = []
        
        # Recurrent processing (simplified state space equation)
        for t in range(seq_len):
            # State update: h_{t+1} = A @ h_t + B @ x_t
            h = torch.tanh(h @ self.A.t() + x[:, t] @ self.B.t())
            
            # Output: y_t = C @ h_t + D @ x_t
            y = h @ self.C.t() + x[:, t] * self.D
            
            outputs.append(y)
        
        # Stack outputs
        output = torch.stack(outputs, dim=1)
        
        return self.dropout(output)

class MambaBlock(nn.Module):
    """
    Simplified Mamba block with selection mechanism
    
    Key differences from standard SSM:
    - Selection mechanism (learns which parts of sequence to focus on)
    - Efficient scanning module
    """
    
    def __init__(self, d_model, d_state=64, d_conv=4, expand=2, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_state: State dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for hidden dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_model * expand
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2)
        
        # Depthwise convolution (for local context)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner
        )
        
        # State space block
        self.ssm = S4Block(self.d_inner, d_state, dropout)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model)
        
        # Activation
        self.activation = nn.SiLU()
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, dim = x.shape
        
        # Split into two paths (like Mamba's dual path)
        x_proj = self.in_proj(x)  # (batch, seq_len, d_inner * 2)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)  # Each: (batch, seq_len, d_inner)
        
        # Apply convolution (need to transpose for Conv1d)
        x_conv = self.conv1d(x_ssm.transpose(1, 2))[:, :, :seq_len]  # Trim padding
        x_conv = x_conv.transpose(1, 2)  # Back to (batch, seq_len, d_inner)
        
        # Activation
        x_conv = self.activation(x_conv)
        
        # Apply SSM
        x_ssm = self.ssm(x_conv)
        
        # Gating mechanism (selection)
        x_gate = self.activation(x_gate)
        x_out = x_ssm * x_gate
        
        # Output projection
        output = self.out_proj(x_out)
        
        return output

class ResidualMambaBlock(nn.Module):
    """
    Mamba block with residual connection and layer norm
    (Corresponds to RSSB in paper)
    """
    
    def __init__(self, d_model, d_state=64, dropout=0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state=d_state, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Pre-norm + Mamba + residual
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        x = self.dropout(x)
        return x + residual

# Test
if __name__ == "__main__":
    # Test shapes
    batch_size = 4
    seq_len = 20
    d_model = 64
    
    x = torch.randn(batch_size, seq_len, d_model)
    
    # Test S4Block
    s4 = S4Block(d_model)
    out = s4(x)
    print(f"S4Block: {x.shape} -> {out.shape}")
    
    # Test MambaBlock
    mamba = MambaBlock(d_model)
    out = mamba(x)
    print(f"MambaBlock: {x.shape} -> {out.shape}")
    
    # Test ResidualMambaBlock
    rssb = ResidualMambaBlock(d_model)
    out = rssb(x)
    print(f"ResidualMambaBlock: {x.shape} -> {out.shape}")
    
    print("✅ All Mamba blocks working!")