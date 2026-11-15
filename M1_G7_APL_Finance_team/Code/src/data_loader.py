"""
Data Loader: Portfolio Returns for M1 Structure Learning
Loads and prepares standardized time-series data
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_portfolio_data_synthetic(n_assets=10, n_observations=2500, seed=42):
    """
    Generate synthetic portfolio returns (realistic AR(1) process).
    
    For M1, uses synthetic data to avoid Yahoo Finance rate limits.
    In production, replace with yfinance.download().
    
    Args:
        n_assets: number of stocks (default 10)
        n_observations: time steps (default 2500 ~= 10 years)
        seed: random seed for reproducibility
    
    Returns:
        pd.DataFrame: standardized log-returns, shape (n_observations, n_assets)
    """
    np.random.seed(seed)
    
    assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'JPM', 'GS', 'XOM', 'CVX', 'NVDA'][:n_assets]
    
    # Generate AR(1) returns (realistic time-series)
    data = np.zeros((n_observations, n_assets))
    for i in range(n_assets):
        ar_coef = np.random.uniform(0.3, 0.7)  # AR(1) coefficient
        data[0, i] = np.random.normal(0, 1)
        for t in range(1, n_observations):
            data[t, i] = ar_coef * data[t-1, i] + np.random.normal(0, 1)
    
    # Standardize
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    
    # Return as DataFrame
    returns = pd.DataFrame(data, columns=assets)
    returns.index = pd.date_range('2015-01-01', periods=n_observations, freq='D')
    
    return returns


def load_portfolio_data_csv(filepath):
    """
    Load portfolio returns from CSV file.
    
    Expected format:
    - Index: dates (optional)
    - Columns: asset tickers
    - Values: log-returns (standardized)
    
    Args:
        filepath: path to CSV file
    
    Returns:
        pd.DataFrame: returns
    """
    returns = pd.read_csv(filepath, index_col=0, parse_dates=True)
    return returns


def train_test_split_temporal(data, train_frac=0.8):
    """
    Temporal train-test split (not random).
    
    Args:
        data: pd.DataFrame of returns
        train_frac: fraction for training (default 0.8)
    
    Returns:
        data_train, data_test
    """
    n = len(data)
    split_idx = int(n * train_frac)
    
    data_train = data.iloc[:split_idx]
    data_test = data.iloc[split_idx:]
    
    return data_train, data_test
