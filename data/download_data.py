"""
Download Indian stock market data from Yahoo Finance
"""
import yfinance as yf
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import time

# Indian stocks to analyze
STOCKS = {
    # Indices
    '^NSEI': 'NIFTY_50',
    '^BSESN': 'SENSEX',
    
    # Large Cap
    'RELIANCE.NS': 'RELIANCE',
    'TCS.NS': 'TCS',
    'HDFCBANK.NS': 'HDFCBANK',
    'INFY.NS': 'INFOSYS',
    'ICICIBANK.NS': 'ICICIBANK',
    'HINDUNILVR.NS': 'HUL',
    'ITC.NS': 'ITC',
    'SBIN.NS': 'SBI',
    
    # Mid Cap
    'ADANIENT.NS': 'ADANI_ENT',
    'BAJFINANCE.NS': 'BAJAJ_FINANCE',
}

def download_stock_data(ticker, name, start_date='2010-01-01', end_date=None):
    """
    Download OHLCV data for a stock
    
    Args:
        ticker: Yahoo Finance ticker symbol
        name: Clean name for saving
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD), defaults to today
    
    Returns:
        DataFrame with OHLC data
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Downloading {name} ({ticker})...")
    
    try:
        # Download data
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        # Check if data is valid
        if data.empty:
            print(f"  ⚠️  No data found for {ticker}")
            return None
        
        # Handle multi-index columns (yfinance sometimes does this)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.droplevel(1)
        
        # Keep only OHLC columns
        data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
        
        # Remove any NaN values
        data = data.dropna()
        
        # Ensure all values are numeric
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Remove any rows with NaN after conversion
        data = data.dropna()
        
        # Add some basic info
        data['Returns'] = data['Close'].pct_change()
        data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
        
        # Save to CSV with proper formatting
        output_path = Path('data/raw') / f'{name}.csv'
        data.to_csv(output_path, index=True, float_format='%.2f')
        
        print(f"  ✅ Downloaded {len(data)} rows, saved to {output_path}")
        print(f"  📊 Date range: {data.index[0].strftime('%Y-%m-%d')} to {data.index[-1].strftime('%Y-%m-%d')}")
        
        # Price statistics
        close_min = float(data['Close'].min())
        close_max = float(data['Close'].max())
        print(f"  💰 Price range: ₹{close_min:.2f} - ₹{close_max:.2f}")
        
        return data
        
    except Exception as e:
        print(f"  ❌ Error downloading {ticker}: {str(e)}")
        return None

def main():
    """Download all stocks"""
    print("=" * 60)
    print("📈 Downloading Indian Stock Market Data")
    print("=" * 60)
    
    # Create directory if it doesn't exist
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    for ticker, name in STOCKS.items():
        data = download_stock_data(ticker, name)
        if data is not None:
            results[name] = data
        time.sleep(1)  # Be nice to Yahoo Finance
    
    print("\n" + "=" * 60)
    print(f"✅ Downloaded {len(results)}/{len(STOCKS)} stocks successfully")
    print("=" * 60)
    
    # Show summary
    if results:
        print("\n📊 Summary:")
        for name, data in results.items():
            print(f"  {name:15s}: {len(data):4d} rows, "
                  f"₹{float(data['Close'].iloc[-1]):8.2f} (latest)")
    
    return results

if __name__ == "__main__":
    results = main()