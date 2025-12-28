import pandas as pd
import numpy as np
import pickle
import torch
import os

class MinMaxNormalizer:
    """
    Normalizes data to range [0, 1].
    Crucial for Neural Networks to converge faster.
    """
    def __init__(self):
        self._min = None
        self._max = None

    def fit(self, data):
        self._min = np.min(data)
        self._max = np.max(data)

    def transform(self, data):
        if self._min is None or self._max is None:
            raise ValueError("Scaler has not been fitted yet.")
        # Avoid division by zero
        if self._max - self._min == 0:
            return np.zeros_like(data)
        return (data - self._min) / (self._max - self._min)

    def inverse_transform(self, data):
        return data * (self._max - self._min) + self._min

def load_traffic_data(file_path: str) -> pd.DataFrame:
    """
    Loads traffic speed data from .h5 file.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_hdf(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Failed to load H5 file: {e}")

def load_adjacency_matrix(file_path: str):
    """
    Loads the sensor graph structure (adjacency matrix).
    """
    if not os.path.exists(file_path):
        # Return None or Identity if file missing (for robust testing)
        return None
        
    with open(file_path, 'rb') as f:
        sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)
    return adj_mx

def create_sequences(data: np.ndarray, seq_len: int, horizon: int):
    """
    Creates input (X) and target (Y) sequences for time-series forecasting.
    X shape: (num_samples, seq_len, num_nodes, num_features)
    Y shape: (num_samples, horizon, num_nodes, num_features)
    """
    X, Y = [], []
    num_samples, num_nodes = data.shape
    
    # We need to add a feature dimension if it's just (Time, Nodes)
    # Target shape: (Time, Nodes, 1)
    if data.ndim == 2:
        data = np.expand_dims(data, axis=-1)
        
    for i in range(num_samples - seq_len - horizon + 1):
        # Input: Past 'seq_len' steps
        x_i = data[i : i + seq_len, :, :]
        # Target: Future 'horizon' steps
        y_i = data[i + seq_len : i + seq_len + horizon, :, :]
        
        X.append(x_i)
        Y.append(y_i)
        
    return np.array(X), np.array(Y)