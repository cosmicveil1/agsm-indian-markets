# AGSMNet Kaggle Setup Instructions

## 1. Upload Data
1. Go to your Kaggle Dataset.
2. Upload the `agsm_code.tar.gz` bundle.
3. **Important**: Also upload your `data.csv` (the 0.6GB LOB file) if you plan to use it.

## 2. Setup Notebook
1. Create a new Notebook on Kaggle.
2. Add your uploaded Dataset to the notebook.
3. Copy the code below into the first cell to extract the code and setup the environment.

```python
# 1. Extract Code
import tarfile
import os

# Adjust payload path if needed (Kaggle mounts datasets at /kaggle/input/...)
# Replace 'your-dataset-name' found in the right panel
dataset_path = '/kaggle/input/your-dataset-name/agsm_code.tar.gz' 

if os.path.exists(dataset_path):
    print("Extracting code...")
    with tarfile.open(dataset_path, "r:gz") as tar:
        tar.extractall("/kaggle/working/")
    print("Done!")
else:
    print("Code bundle not found. Please check the path.")

# 2. Install Dependencies
!pip install -q einops yfinance
!pip install -q causal-conv1d>=1.2.0
!pip install -q mamba-ssm>=1.2.0

# 3. Verify
!ls -R /kaggle/working/
```

## 3. Run Training
```python
%cd /kaggle/working/

# Example: Train on one of the smaller CSVs (OHLC)
!python experiments/train.py --data data/raw/RELIANCE.csv --model lite --epochs 10

# Example: Train on LOB data (if data.csv is uploaded)
# !python experiments/train_lob.py --data /kaggle/input/your-dataset-name/data.csv --model full --epochs 50
```
