# AGSMNet: Adaptive Gaussian STFT + Mamba Network for Stock Price Prediction

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Implementation of **AGSMNet** from the paper:

> **"AGSMNet: A Novel Approach for Stock Price Prediction Using Adaptive Gaussian STFT and Mamba Architecture"**
> Huang et al. (2025), Engineering Applications of Artificial Intelligence

##  Key Contributions

1. **Adaptive Gaussian STFT (AG-STFT)**: Novel time-frequency transform with frequency-dependent window width
2. **2D-SSM Module**: 4-directional scanning for spectrogram features  
3. **RSSG Architecture**: Residual State-Space Groups with VSSM and Channel Attention
4. **End-to-end Pipeline**: Raw OHLC → Spectrogram → Price Prediction

##  Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                            AGSMNet Pipeline                              │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   Input Data ──► AG-STFT ──► MFE ──► RSSG (×N) ──► Predictor ──► Price   │
│   (OHLC or LOB)                                                         │
│                                                                          │
│   ┌─────────┐   ┌─────────┐   ┌─────────────────┐   ┌──────────┐        │
│   │ Open    │   │ Adaptive│   │  Conv + Pool    │   │ RSSB ×M  │        │
│   │ High    │──►│ Gaussian│──►│  Shallow        │──►│ 2D-SSM   │──► FC  │
│   │ LOB     │   │ STFT    │   │  Features       │   │ + CA     │        │
│   │ ...     │   │         │   │                 │   │          │        │
│   └─────────┘   └─────────┘   └─────────────────┘   └──────────┘        │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

##  Data & Strategy
The model supports two modes of operation:
1.  **OHLC (Long-term)**: Standard daily candlesticks (Open, High, Low, Close).
2.  **LOB (High-Frequency)**: Limit Order Book data containing Bid/Ask queues.
    *   *Why LOB?*
        *   **Data Volume**: Standard OHLC data (~5k points) is insufficient for deep training.
        *   **Microstructure**: LOB data provides **millions of data points**, enabling the Mamba architecture to capture high-frequency market microstructure without overfitting.

### AG-STFT: Adaptive Gaussian STFT

Unlike standard STFT with fixed windows, AG-STFT adapts window width based on frequency:

```
σ(f) = α / (f + ε)     # Equation (1): Adaptive sigma

Low frequency  → Wide window  → Better frequency resolution
High frequency → Narrow window → Better time resolution
```

This provides optimal time-frequency resolution for non-stationary financial data.

### 2D-SSM: 4-Directional Selective Scan

Scans spectrogram features in 4 directions to capture dependencies:
- Left → Right (→)
- Right → Left (←)  
- Top → Bottom (↓)
- Bottom → Top (↑)

### RSSB: Residual State-Space Block

Each RSSB contains:
1. **VSSM**: Vision State-Space Module with 2D-SSM
2. **Channel Attention**: SE-Net style attention
3. **Learnable residual scales**: β and γ parameters

##  Project Structure

```
agsm-indian-markets/
├── models/
│   ├── ag_stft.py          # Adaptive Gaussian STFT (Equations 1-3)
│   ├── mamba_simple.py     # Mamba, 2D-SSM, VSSM, RSSB, CA
│   ├── agsm_net.py         # AGSMNet for OHLC (Daily)
│   └── agsm_lob.py         # AGSMNet for LOB (High-Frequency)
├── utils/
│   ├── dataset.py          # OHLC Dataset (Window-based Norm)
│   ├── dataset_lob.py      # LOB Dataset (Limit Order Book)
│   └── data_utils.py       # Data preprocessing utilities
├── experiments/
│   ├── train.py            # Training script for OHLC
│   ├── train_lob.py        # Training script for LOB
│   ├── checkpoints/        # Saved models
│   └── results/            # Training curves, predictions
├── data/
│   ├── download_data.py    # Download stock data
│   └── raw/                # Raw CSV files
├── tests/
│   └── test_components.py  # Unit tests
├── walkthrough.md          # Technical Deep Dive & Architecture
└── notebooks/              # Jupyter notebooks for analysis
```

##  Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/agsm-indian-markets.git
cd agsm-indian-markets

# Create virtual environment
python -m venv agsm_venv
source agsm_venv/bin/activate  # Linux/Mac
# or: agsm_venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
python data/download_data.py
```

### Train Model

```bash
# Train AGSMNet Lite (faster, fewer parameters)
python experiments/train.py --data data/raw/RELIANCE.csv --model lite --epochs 100

# Train full AGSMNet (more accurate, more parameters)
python experiments/train.py --data data/raw/RELIANCE.csv --model full --epochs 100
```

### Run Tests

```bash
python -m pytest tests/test_components.py -v
```

##  Evaluation Metrics

Following the paper (Section 4.2):

| Metric | Formula | Description |
|--------|---------|-------------|
| MAE | $\frac{1}{n}\sum\|y_i - \hat{y}_i\|$ | Mean Absolute Error |
| MSE | $\frac{1}{n}\sum(y_i - \hat{y}_i)^2$ | Mean Square Error |
| RMSE | $\sqrt{MSE}$ | Root Mean Square Error |
| R² | $1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$ | Coefficient of Determination |

##  Model Configuration

### AGSMNet (Full)

```python
model = AGSMNet(
    in_channels=4,          # OHLC channels
    freq_bins=17,           # From AG-STFT
    time_steps=20,          # From AG-STFT
    mfe_hidden=32,          # MFE first conv
    mfe_out=64,             # MFE output channels
    n_rssg_groups=2,        # Number of RSSG groups
    n_rssb_per_group=3,     # RSSBs per group
    d_state=64,             # SSM state dimension
    predictor_hidden=128,   # Predictor FC dim
    dropout=0.1
)
```

### AGSMNetLite (Lightweight)

```python
model = AGSMNetLite(
    in_channels=4,
    freq_bins=17,
    time_steps=20,
    hidden_dim=64,
    n_mamba_layers=2,
    d_state=32,
    dropout=0.1
)
```

##  Evaluation & Results

### Research Status
The project has successfully achieved several critical engineering and research milestones:
*   **Pipeline Verification**: Validated the complete end-to-end training pipeline from raw LOB data ingestion to Mamba-based inference.
*   **Component Validity**: Unit tests (`tests/test_components.py`) confirm that the custom AG-STFT and 2D-SSM layers mathematically behave as described in the reference paper.
*   **Scalability**: The `AGSMNetLite` variant successfully trains on consumer hardware, demonstrating the efficiency of the Mamba architecture compared to quadratic-complexity Transformers.

### Performance Metrics
The model is evaluated using a rigorous set of metrics to ensure both numerical accuracy and directional correctness:
- **Numerical Accuracy**: MAE, RMSE, and R² Score.
- **Trading Relevance**: Directional Accuracy (DA) measures the percentage of correct trend predictions (Up/Down).
- **Baselines**: Performance is benchmarked against a Naive Predictor (Yesterday's value) to demonstrate true learning.

*Current experiments focus on convergence stability on the expanded LOB dataset.*

##  Roadmap & Future Work

This project is under active development. Current focus areas include:
1.  **Hyperparameter Optimization**: Systematically tuning `window_size`, `alpha` (STFT), and Mamba state dimensions using Ray Tune.
2.  **Extended Dataset**: Scaling training to include 50+ NIFTY 50 stocks to improve generalization.
3.  **Real-time Inference**: Exporting the model to ONNX for low-latency inference in live trading scenarios.
4.  **Transformer vs Mamba**: Conducting a rigorous A/B test against Transformer-based baselines (PatchTST, iTransformer).

For a deep dive into the architectural decisions and implementation details, please see the [**Technical Walkthrough**](walkthrough.md).

##  Component Tests

```python
# Test AG-STFT adaptive windows
from models.ag_stft import AdaptiveGaussianSTFT

ag_stft = AdaptiveGaussianSTFT(alpha=1.0)

# Low frequency = wide window
sigma_low = ag_stft.compute_adaptive_sigma(0.01)  # ~100

# High frequency = narrow window  
sigma_high = ag_stft.compute_adaptive_sigma(0.5)   # ~2

assert sigma_low > sigma_high  # ✓
```

##  References

1. Huang et al. (2025). "AGSMNet: A Novel Approach for Stock Price Prediction Using Adaptive Gaussian STFT and Mamba Architecture"
2. Gu & Dao (2023). "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
3. Gu et al. (2021). "Efficiently Modeling Long Sequences with Structured State Spaces"
4. Hu et al. (2018). "Squeeze-and-Excitation Networks" (Channel Attention)

##  License

MIT License - see [LICENSE](LICENSE) for details.

##  Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

##  Contact

For questions about implementation details, please open an issue.
