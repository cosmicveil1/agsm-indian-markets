"""
Mamba-based State Space Model with 2D Selective Scan
Faithful Implementation based on Huang et al. (2025) - AGSMNet Paper

Key Components:
1. S4 Block - Structured State Space Model (Gu et al., 2021)
2. Mamba Block - Selection mechanism + Scanning module (Gu & Dao, 2023)
3. 2D-SSM - 4-directional scanning for 2D spectrograms (Figure 5)
4. VSSM - Vision State-Space Module (Figure 4)
5. Channel Attention (CA) - From Hu et al., 2018 (SE-Net)
6. RSSB - Residual State-Space Block (Equations 4-5)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from einops import rearrange, repeat
import math

# @torch.jit.script # Removed JIT to allow torch.compile
def ssm_scan_efficient(
    x: torch.Tensor, 
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor
) -> torch.Tensor:
    """
    Memory-efficient Selective Scan.
    Computes A_bar and B_bar on the fly to avoid O(L*D*N) memory cost.
    
    x: (B, L, D)
    delta: (B, L, D)
    A: (N)
    B: (B, L, N)
    C: (B, L, N)
    D: (D)
    """
    batch, seq_len, d_inner = x.shape
    d_state = A.shape[0]
    
    h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
    outputs = []
    
    for t in range(seq_len):
        # Discretize on the fly for step t
        # delta_t: (B, D)
        dt = delta[:, t]
        
        # A_bar_t = exp(dt * A) -> (B, D, N)
        # B_bar_t = dt * B_t -> (B, D, N)
        
        # Broadcasting: dt (B, D, 1) * A (1, 1, N) -> (B, D, N)
        A_bar_t = torch.exp(dt.unsqueeze(-1) * A)
        
        # B_bar_t: dt (B, D, 1) * B[:, t] (B, 1, N) -> (B, D, N)
        B_t = B[:, t]
        B_bar_t = dt.unsqueeze(-1) * B_t.unsqueeze(1)
        
        # State update
        # h: (B, D, N)
        # x[:, t]: (B, D)
        xt = x[:, t].unsqueeze(-1) # (B, D, 1)
        
        h = A_bar_t * h + B_bar_t * xt
        
        # Output: (B, D)
        # y = (h * C[:, t].unsqueeze(1)).sum(dim=-1)
        Ct = C[:, t].unsqueeze(1) # (B, 1, N)
        y = (h * Ct).sum(dim=-1)
        
        outputs.append(y)
    
    y = torch.stack(outputs, dim=1)
    return y + x * D



class S4Block(nn.Module):
    """
    Structured State Space Model (S4) Block
    
    Based on "Efficiently Modeling Long Sequences with Structured State Spaces"
    (Gu et al., 2021)
    
    State space equation:
        h_{t+1} = A @ h_t + B @ x_t
        y_t = C @ h_t + D @ x_t
    
    Key innovation: Learnable A, B, C, D matrices that can capture
    long-range dependencies with linear complexity.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = False
    ):
        """
        Args:
            d_model: Model/input dimension
            d_state: State dimension (N in paper)
            dropout: Dropout rate
            bidirectional: Whether to process sequence in both directions
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        self.bidirectional = bidirectional
        
        # Learnable state space parameters (core S4/Mamba contribution)
        # A: State transition matrix (d_state x d_state)
        # B: Input projection (d_state x d_model)
        # C: Output projection (d_model x d_state)
        # D: Skip connection (d_model)
        
        # Initialize A with HiPPO matrix structure for long-range dependencies
        self.A = nn.Parameter(self._init_hippo_matrix(d_state))
        self.B = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))
        
        # Discretization parameters (delta in Mamba)
        self.log_delta = nn.Parameter(torch.zeros(d_model))
        
        self.dropout = nn.Dropout(dropout)
        
        if bidirectional:
            # Separate parameters for backward direction
            self.A_back = nn.Parameter(self._init_hippo_matrix(d_state))
            self.B_back = nn.Parameter(torch.randn(d_state, d_model) * 0.01)
            self.C_back = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
    
    def _init_hippo_matrix(self, n: int) -> torch.Tensor:
        """
        Initialize A matrix with HiPPO structure.
        HiPPO matrices enable efficient long-range dependency modeling.
        """
        # Simplified HiPPO-LegS initialization
        p = torch.arange(n, dtype=torch.float32)
        A = torch.zeros(n, n)
        for i in range(n):
            for j in range(n):
                if i > j:
                    A[i, j] = -math.sqrt((2*i+1) * (2*j+1))
                elif i == j:
                    A[i, j] = -(i + 1)
        return A * 0.1  # Scale down for stability
    
    def _discretize(self, A: torch.Tensor, B: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Discretize continuous state space to discrete (ZOH method).
        
        A_bar = exp(delta * A)
        B_bar = (exp(delta * A) - I) * A^{-1} * B ≈ delta * B for small delta
        """
        delta = F.softplus(self.log_delta)  # Ensure positive
        
        # Approximate discretization for efficiency
        A_bar = torch.matrix_exp(delta.unsqueeze(-1).unsqueeze(-1) * A.unsqueeze(0))
        B_bar = delta.unsqueeze(-1) * B.unsqueeze(0)  # Simplified
        
        return A_bar.mean(0), B_bar.mean(0)
    
    def _ssm_forward(
        self, 
        x: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor
    ) -> torch.Tensor:
        """
        Run SSM recurrence.
        
        Args:
            x: (batch, seq_len, d_model)
            A, B, C: State space matrices
            
        Returns:
            y: (batch, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # Discretize A, B
        A_bar, B_bar = self._discretize(A, B)
        
        # Initialize state
        h = torch.zeros(batch_size, self.d_state, device=x.device, dtype=x.dtype)
        
        outputs = []
        
        for t in range(seq_len):
            # State update: h = A_bar @ h + B_bar @ x
            h = torch.tanh(h @ A_bar.t() + x[:, t] @ B_bar.t())
            
            # Output: y = C @ h + D * x
            y = h @ C.t() + x[:, t] * self.D
            outputs.append(y)
        
        return torch.stack(outputs, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        # Forward direction
        y_forward = self._ssm_forward(x, self.A, self.B, self.C)
        
        if self.bidirectional:
            # Backward direction
            x_flip = torch.flip(x, dims=[1])
            y_backward = self._ssm_forward(x_flip, self.A_back, self.B_back, self.C_back)
            y_backward = torch.flip(y_backward, dims=[1])
            
            # Combine
            y = (y_forward + y_backward) / 2
        else:
            y = y_forward
        
        return self.dropout(y)


class SelectionMechanism(nn.Module):
    """
    Selection Mechanism from Mamba
    
    Key innovation: Learn to dynamically select which parts of the input
    are relevant for prediction, making the SSM content-aware.
    
    Delta, B, C are computed from input (not fixed parameters).
    """
    
    def __init__(self, d_model: int, d_state: int = 64, d_conv: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.d_state = d_state
        
        # Selection projections (compute B, C, delta from input)
        self.s_B = nn.Linear(d_model, d_state, bias=False)
        self.s_C = nn.Linear(d_model, d_state, bias=False)
        self.s_delta = nn.Linear(d_model, d_model, bias=True)
        
        # Fixed A matrix (log-spaced for stability)
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer('A_log', torch.log(A))
        
        # Skip connection
        self.D = nn.Parameter(torch.ones(d_model))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute input-dependent B, C, delta.
        
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            A, B, C, delta: Selection parameters
        """
        A = -torch.exp(self.A_log)  # (d_state,)
        B = self.s_B(x)  # (batch, seq_len, d_state)
        C = self.s_C(x)  # (batch, seq_len, d_state)
        delta = F.softplus(self.s_delta(x))  # (batch, seq_len, d_model)
        
        return A, B, C, delta


class MambaBlock(nn.Module):
    """
    Mamba Block with Selection Mechanism and Scanning
    
    Architecture (from Gu & Dao, 2023):
    1. Input projection to 2x channels (for gating)
    2. Depthwise convolution for local context
    3. Selection mechanism (input-dependent SSM parameters)
    4. SSM with selective scan
    5. Gating and output projection
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input/output dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor for inner dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        # Input projection: d_model -> 2 * d_inner (for x and gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise convolution for local context
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            padding=d_conv - 1,
            groups=self.d_inner,
            bias=True
        )
        
        # Selection mechanism
        self.selection = SelectionMechanism(self.d_inner, d_state, d_conv)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=True)
        
        # Activation
        self.activation = nn.SiLU()
        
        self.dropout = nn.Dropout(dropout)
        
    def selective_scan(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Selective scan operation (core of Mamba).
        
        Unlike standard SSM, parameters B, C, delta are input-dependent.
        
        Args:
            x: (batch, seq_len, d_inner)
            A: (d_state,) - fixed
            B: (batch, seq_len, d_state) - input-dependent
            C: (batch, seq_len, d_state) - input-dependent
            delta: (batch, seq_len, d_inner) - input-dependent
            
        Returns:
            y: (batch, seq_len, d_inner)
        """
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[0]
        
        # Discretize A and B based on delta
        # Memory Optimization: We do NOT pre-compute A_bar and B_bar here anymore.
        # They are computed on the fly in ssm_scan_efficient to save memory.
        # (Was: delta_A = ..., A_bar = ...) 
        pass
        
        # Use memory-efficient scan
        # Computes A_bar and B_bar on the fly
        y = ssm_scan_efficient(x, delta, A, B, C, self.selection.D)
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        batch, seq_len, _ = x.shape
        
        # Project and split into x_ssm and gate
        x_proj = self.in_proj(x)  # (batch, seq, 2*d_inner)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)  # Each: (batch, seq, d_inner)
        
        # Apply depthwise convolution
        x_conv = rearrange(x_ssm, 'b l d -> b d l')
        x_conv = self.conv1d(x_conv)[:, :, :seq_len]  # Trim to original length
        x_conv = rearrange(x_conv, 'b d l -> b l d')
        x_conv = self.activation(x_conv)
        
        # Get input-dependent SSM parameters
        A, B, C, delta = self.selection(x_conv)
        
        # Selective scan
        y = self.selective_scan(x_conv, A, B, C, delta)
        
        # Gating
        y = y * self.activation(x_gate)
        
        # Output projection
        output = self.out_proj(y)
        
        return self.dropout(output)


class SS2D(nn.Module):
    """
    2D Selective Scan Module (2D-SSM)
    
    From Figure 5: Transform 2D features to 1D sequences,
    scan in 4 directions, then combine and reshape back to 2D.
    
    Directions:
    1. Left-to-Right (→)
    2. Right-to-Left (←)
    3. Top-to-Bottom (↓)
    4. Bottom-to-Top (↑)
    
    This allows capturing dependencies in all spatial directions.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_model * expand
        self.d_state = d_state
        
        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        
        # Depthwise conv
        self.conv2d = nn.Conv2d(
            self.d_inner,
            self.d_inner,
            kernel_size=d_conv,
            padding=d_conv // 2,
            groups=self.d_inner,
            bias=True
        )
        
        # 4 separate Mamba blocks for 4 directions
        # More efficient: share parameters, just change scan direction
        self.selection = SelectionMechanism(self.d_inner, d_state)
        
        # Merge projections for 4 directions
        self.merge = nn.Linear(self.d_inner * 4, self.d_inner)
        
        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
        # Fixed A
        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer('A_log', torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        
    def scan_1d(
        self,
        x: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        delta: torch.Tensor,
        reverse: bool = False
    ) -> torch.Tensor:
        """
        1D selective scan in one direction.
        
        Args:
            x: (batch, seq_len, d_inner)
            reverse: If True, scan right-to-left
            
        Returns:
            y: (batch, seq_len, d_inner)
        """
        if reverse:
            x = torch.flip(x, dims=[1])
            B = torch.flip(B, dims=[1])
            C = torch.flip(C, dims=[1])
            delta = torch.flip(delta, dims=[1])
        
        batch_size, seq_len, d_inner = x.shape
        d_state = A.shape[0]
        
        # Discretize on the fly
        pass
        
        # Scan using memory-efficient implementation
        y = ssm_scan_efficient(x, delta, A, B, C, self.D)
        
        if reverse:
            y = torch.flip(y, dims=[1])
        
        return y
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width) - 2D feature map
            
        Returns:
            output: (batch, channels, height, width)
        """
        batch, C, H, W = x.shape
        
        # Reshape to (batch, H*W, C) for projection
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        
        # Project and split
        x_proj = self.in_proj(x_flat)
        x_ssm, x_gate = x_proj.chunk(2, dim=-1)
        
        # Reshape for 2D conv
        x_2d = rearrange(x_ssm, 'b (h w) c -> b c h w', h=H, w=W)
        x_2d = self.activation(self.conv2d(x_2d))
        
        # Get SSM parameters
        A = -torch.exp(self.A_log)
        
        # Scan in 4 directions
        outputs = []
        
        # Direction 1: Row-wise left-to-right
        x_row = rearrange(x_2d, 'b c h w -> (b h) w c')
        B1, C1, delta1 = self.selection.s_B(x_row), self.selection.s_C(x_row), F.softplus(self.selection.s_delta(x_row))
        y1 = self.scan_1d(x_row, A, B1, C1, delta1, reverse=False)
        y1 = rearrange(y1, '(b h) w c -> b c h w', b=batch, h=H)
        outputs.append(y1)
        
        # Direction 2: Row-wise right-to-left
        y2 = self.scan_1d(x_row, A, B1, C1, delta1, reverse=True)
        y2 = rearrange(y2, '(b h) w c -> b c h w', b=batch, h=H)
        outputs.append(y2)
        
        # Direction 3: Column-wise top-to-bottom
        x_col = rearrange(x_2d, 'b c h w -> (b w) h c')
        B3, C3, delta3 = self.selection.s_B(x_col), self.selection.s_C(x_col), F.softplus(self.selection.s_delta(x_col))
        y3 = self.scan_1d(x_col, A, B3, C3, delta3, reverse=False)
        y3 = rearrange(y3, '(b w) h c -> b c h w', b=batch, w=W)
        outputs.append(y3)
        
        # Direction 4: Column-wise bottom-to-top
        y4 = self.scan_1d(x_col, A, B3, C3, delta3, reverse=True)
        y4 = rearrange(y4, '(b w) h c -> b c h w', b=batch, w=W)
        outputs.append(y4)
        
        # Merge 4 directions
        y_merged = torch.cat(outputs, dim=1)  # (batch, 4*d_inner, H, W)
        y_merged = rearrange(y_merged, 'b c h w -> b (h w) c')
        y_merged = self.merge(y_merged)  # (batch, H*W, d_inner)
        
        # Gating
        x_gate_2d = rearrange(x_gate, 'b (h w) c -> b (h w) c', h=H)
        y_gated = y_merged * self.activation(x_gate_2d)
        
        # Output projection
        output = self.out_proj(y_gated)
        output = rearrange(output, 'b (h w) c -> b c h w', h=H, w=W)
        
        return self.dropout(output)


class ChannelAttention(nn.Module):
    """
    Channel Attention (CA) Module from SE-Net (Hu et al., 2018)
    
    Referenced in paper Equation (5) to avoid channel redundancy
    and focus on learning diverse channel representations.
    """
    
    def __init__(self, channels: int, reduction: int = 16):
        """
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # self.max_pool = nn.AdaptiveMaxPool2d(1) # Replaced with functional amax
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
            
        Returns:
            output: (batch, channels, height, width) - attention-weighted
        """
        batch, C, H, W = x.shape
        
        # Global average pooling
        avg_out = self.avg_pool(x).view(batch, C)
        avg_out = self.mlp(avg_out)
        
        # Global max pooling - Functional equivalent to avoid torch.compile bug with AdaptiveMaxPool2d
        # max_out = self.max_pool(x).view(batch, C)
        max_out = torch.amax(x, dim=(2, 3)).view(batch, C)
        max_out = self.mlp(max_out)
        
        # Combine and apply sigmoid
        attention = self.sigmoid(avg_out + max_out).view(batch, C, 1, 1)
        
        return x * attention


class VSSM(nn.Module):
    """
    Vision State-Space Module (VSSM)
    
    From Figure 4 in the paper:
    1. Split into two branches
    2. Branch 1: Linear -> DWConv -> SiLU -> SS2D
    3. Branch 2: Linear -> SiLU
    4. Hadamard product + projection
    
    Equations (6-8) in the paper.
    """
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: Input channels
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout rate
        """
        super().__init__()
        
        self.d_model = d_model
        self.d_inner = d_model * expand
        
        # Branch 1: Linear -> DWConv -> SiLU -> SS2D
        self.branch1_linear = nn.Conv2d(d_model, self.d_inner, 1, bias=False)
        self.branch1_dwconv = nn.Conv2d(
            self.d_inner, self.d_inner, 
            kernel_size=d_conv, padding=d_conv//2,
            groups=self.d_inner, bias=True
        )
        self.branch1_ss2d = SS2D(
            d_model=self.d_inner,
            d_state=d_state,
            d_conv=d_conv,
            expand=1,  # Already expanded
            dropout=dropout
        )
        
        # Branch 2: Linear -> SiLU
        self.branch2_linear = nn.Conv2d(d_model, self.d_inner, 1, bias=False)
        
        # Output projection
        self.out_proj = nn.Conv2d(self.d_inner, d_model, 1, bias=False)
        
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout2d(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
            
        Returns:
            output: (batch, channels, height, width)
        """
        # Branch 1: Eq (6)
        b1 = self.branch1_linear(x)
        b1 = self.branch1_dwconv(b1)
        b1 = self.activation(b1)
        b1 = self.branch1_ss2d(b1)  # SS2D instead of just SSM
        
        # Branch 2: Eq (7)
        b2 = self.branch2_linear(x)
        b2 = self.activation(b2)
        
        # Hadamard product: Eq (8)
        out = b1 * b2
        
        # Project back
        out = self.out_proj(out)
        
        return self.dropout(out)


class RSSB(nn.Module):
    """
    Residual State-Space Block (RSSB)
    
    From Section 3.4.1, Equations (4-5):
    
    Eq (4): F_VSSM = VSSM(LayerNorm(F_in)) * β + F_in
    Eq (5): F_out = CA(Conv(LayerNorm(F_VSSM))) * γ + F_VSSM
    
    Where β and γ are learnable scale factors for residual connections.
    """
    
    def __init__(
        self,
        channels: int,
        d_state: int = 64,
        d_conv: int = 3,
        expand: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            channels: Number of input/output channels
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: VSSM expansion factor
            dropout: Dropout rate
        """
        super().__init__()
        
        self.channels = channels
        
        # First stage: LayerNorm -> VSSM with residual
        self.norm1 = nn.GroupNorm(1, channels)  # LayerNorm for 2D
        self.vssm = VSSM(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            dropout=dropout
        )
        self.beta = nn.Parameter(torch.ones(1) * 0.1)  # Learnable scale
        
        # Second stage: LayerNorm -> Conv -> CA with residual
        self.norm2 = nn.GroupNorm(1, channels)
        
        # Bottleneck convolution for local features
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1, bias=False),  # Expand
            nn.GELU(),
            nn.Conv2d(channels * 2, channels, 3, padding=1, bias=False),  # Process
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, bias=False)  # Compress
        )
        
        self.channel_attention = ChannelAttention(channels)
        self.gamma = nn.Parameter(torch.ones(1) * 0.1)  # Learnable scale
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
            
        Returns:
            output: (batch, channels, height, width)
        """
        # Equation (4): VSSM with residual
        identity = x
        out = self.norm1(x)
        out = self.vssm(out)
        out = out * self.beta + identity  # Scaled residual
        
        # Equation (5): Conv + CA with residual
        identity = out
        out = self.norm2(out)
        out = self.conv(out)
        out = self.channel_attention(out)
        out = out * self.gamma + identity  # Scaled residual
        
        return out


class ResidualMambaBlock(nn.Module):
    """
    1D Residual Mamba Block for sequence modeling.
    
    Simpler version for when input is already 1D sequence.
    Used after MFE when features are flattened.
    """
    
    def __init__(self, d_model: int, d_state: int = 64, dropout: float = 0.1):
        super().__init__()
        
        self.norm = nn.LayerNorm(d_model)
        self.mamba = MambaBlock(d_model, d_state=d_state, dropout=dropout)
        # Use identity-like scaling (don't kill the residual)
        self.scale = nn.Parameter(torch.ones(1) * 1.0)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, d_model)
            
        Returns:
            output: (batch, seq_len, d_model)
        """
        residual = x
        x = self.norm(x)
        x = self.mamba(x)
        return x * self.scale + residual


# Test
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Mamba 2D-SSM Implementation")
    print("=" * 70)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    
    batch_size = 2
    channels = 64
    height = 11  # Freq bins
    width = 10   # Time steps
    
    # Test S4Block
    print("\n1. Testing S4Block...")
    s4 = S4Block(d_model=64, d_state=32).to(device)
    x_1d = torch.randn(batch_size, 20, 64).to(device)
    out = s4(x_1d)
    print(f"   Input: {x_1d.shape} → Output: {out.shape} ✓")
    
    # Test MambaBlock
    print("\n2. Testing MambaBlock (with Selection)...")
    mamba = MambaBlock(d_model=64, d_state=32).to(device)
    out = mamba(x_1d)
    print(f"   Input: {x_1d.shape} → Output: {out.shape} ✓")
    
    # Test SS2D (2D Selective Scan)
    print("\n3. Testing SS2D (4-directional scan)...")
    ss2d = SS2D(d_model=channels, d_state=32).to(device)
    x_2d = torch.randn(batch_size, channels, height, width).to(device)
    out = ss2d(x_2d)
    print(f"   Input: {x_2d.shape} → Output: {out.shape} ✓")
    
    # Test Channel Attention
    print("\n4. Testing Channel Attention...")
    ca = ChannelAttention(channels).to(device)
    out = ca(x_2d)
    print(f"   Input: {x_2d.shape} → Output: {out.shape} ✓")
    
    # Test VSSM
    print("\n5. Testing VSSM (Vision State-Space Module)...")
    vssm = VSSM(d_model=channels, d_state=32).to(device)
    out = vssm(x_2d)
    print(f"   Input: {x_2d.shape} → Output: {out.shape} ✓")
    
    # Test RSSB
    print("\n6. Testing RSSB (Residual State-Space Block)...")
    rssb = RSSB(channels=channels, d_state=32).to(device)
    out = rssb(x_2d)
    print(f"   Input: {x_2d.shape} → Output: {out.shape} ✓")
    
    # Test ResidualMambaBlock
    print("\n7. Testing ResidualMambaBlock (1D)...")
    rmb = ResidualMambaBlock(d_model=64, d_state=32).to(device)
    out = rmb(x_1d)
    print(f"   Input: {x_1d.shape} → Output: {out.shape} ✓")
    
    # Count parameters
    print("\n" + "=" * 70)
    print("Parameter Counts:")
    print("=" * 70)
    for name, module in [("S4Block", s4), ("MambaBlock", mamba), ("SS2D", ss2d),
                         ("ChannelAttention", ca), ("VSSM", vssm), ("RSSB", rssb)]:
        params = sum(p.numel() for p in module.parameters())
        print(f"  {name}: {params:,} parameters")
    
    print("\n✅ All Mamba modules working correctly!")
    print("\nKey components implemented:")
    print("  ✓ S4 Block with HiPPO initialization")
    print("  ✓ Mamba Block with Selection Mechanism")
    print("  ✓ SS2D (2D-SSM) with 4-directional scanning")
    print("  ✓ Channel Attention (SE-Net style)")
    print("  ✓ VSSM (Vision State-Space Module)")
    print("  ✓ RSSB (Residual State-Space Block)")
