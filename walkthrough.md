# AGSMNet Project Walkthrough

## 1. Project Overview
This project implements **AGSMNet** (Adaptive Gaussian STFT + Mamba Network), a state-of-the-art architecture for financial time-series forecasting. It explicitly tackles the challenge of **non-stationary** stock market data by combining time-frequency analysis with selective state-space models.

### Key Innovations Implemented
1.  **Adaptive Gaussian STFT (AG-STFT)**: A custom signal processing layer that dynamically adjusts the window size based on frequency. This allows capturing both transient high-frequency changes (market noise/shocks) and long-term low-frequency trends.
2.  **Mamba / State-Space Models (SSM)**: Instead of Transformers, this project uses Mamba (VSSM), which scales linearly with sequence length ($O(L)$) rather than quadratically ($O(L^2)$). This is crucial for processing high-resolution financial data.
3.  **2D Selective Scan**: Adapted the 1D Mamba scan to 2D spectrograms, scanning in 4 directions (forward, backward, up, down) to capture global context.

## 2. Technical Implementation Details

### Data Pipeline (`utils/dataset.py`)
-   **Preprocessing**: Raw OHLC (Open-High-Low-Close) data is normalized using a **rolling window** approach to handle distribution shifts.
-   **Spectrogram Generation**:
    -   Input: Time series window ($T \times C$).
    -   Transform: AG-STFT applied on-the-fly or cached.
    -   Output: Time-Frequency representation ($F \times T \times C$).

### Model Architecture (`models/`)
-   **`ag_stft.py`**: logic for the adaptive sigma calculation: $\sigma(f) = \alpha / (f + \epsilon)$.
-   **`mamba_simple.py`**: Custom implementation of the RSSB (Residual State-Space Block) and VSSM.
    -   Includes **Channel Attention (SE-Block)** to weight important frequency bands.
    -   Uses `einsum` for efficient multi-dimentional operations.

### Training Loop (`experiments/train.py`)
-   Implemented a custom training loop with:
    -   **Gradient Clipping**: To prevent exploding gradients common in RNNs/SSMs.
    -   **Scheduler**: `ReduceLROnPlateau` to adapt learning rate.
    -   **Metrics**: Tracking RMSE, MAE, R², and Directional Accuracy.
    -   **Checkpointing**: Saves full state (optimizer + model) for resuming.

## 3. Challenges & Solutions

### Challenge: Non-Stationarity
*Problem*: Stock prices shift distribution over time (e.g., prices in 2020 vs 2024). Standard normalization fails.
*Solution*: Implemented **Window-based Z-score normalization**. Each sample is normalized based *only* on its lookback window.

### Challenge: Memory Usage
*Problem*: 2D feature maps with Mamba can consume significant VRAM.
*Solution*: Implemented `AGSMNetLite`, a stripped-down version reducing channels and layers while maintaining the core architectural advantages.

### Challenge: Data Scarcity (OHLC vs LOB)
*Problem*: Daily OHLC data for a single stock yields only ~5,000 points over 20 years, which is insufficient for training deep SSMs without overfitting.
*Solution*: Integrated **Limit Order Book (LOB)** data training support. LOB data provides tick-level granularity (Bid/Ask queues), expanding the dataset from thousands to millions of data points, allowing the Mamba model to learn complex microstructure patterns.

## 4. Engineering Best Practices
This project adheres to professional software engineering standards to ensuring maintainability and scalability:
-   **Modular Design**: Components (AG-STFT, Mamba, Backend) are decoupled into separate modules in `models/` for reusability.
-   **Testing**: Unit tests in `tests/` ensure component correctness using `pytest`.
-   **Configuration Management**: Hyperparameters and model configs are separated from logic.
-   **Type Hinting**: Extensive use of Python type hints for code clarity and IDE support.

## 5. Future Improvements for Production
-   **Hyperparameter Tuning**: Run a sweep (Ray Tune / Optuna) to find optimal window sizes and alpha for STFT.
-   **Ensembling**: Combine AGSMNet with Transformer baselines.
-   **Latency Optimization**: Export to ONNX/TensorRT for real-time inference.
