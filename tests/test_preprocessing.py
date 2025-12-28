import numpy as np
import pytest
import sys
import os

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from src.mobility.preprocessing import (
    load_adjacency_matrix, 
    load_traffic_data, 
    create_sequences, 
    MinMaxNormalizer
)

def test_load_adjacency_matrix():
    # We just check if it returns None when file doesn't exist (robustness)
    # or check normal loading if we had a dummy pickle.
    assert load_adjacency_matrix("non_existent_file.pkl") is None

def test_load_traffic_data():
    # Should raise error for missing file
    with pytest.raises(FileNotFoundError):
        load_traffic_data("non_existent.h5")

def test_minmax_normalization():
    # Create dummy data: [10, 20, 30]
    data = np.array([[10], [20], [30]])
    
    # --- FIX 1: Initialize first, then fit ---
    scaler = MinMaxNormalizer()
    scaler.fit(data)
    
    normalized = scaler.transform(data)
    
    # Min should be 0, Max should be 1
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0
    
    # Check inverse transform
    inversed = scaler.inverse_transform(normalized)
    assert np.allclose(data, inversed)

def test_create_sequences():
    # Data: numbers 0 to 19 (20 total steps)
    # Shape: (20, 1) -> 20 time steps, 1 sensor
    data = np.arange(20).reshape(-1, 1)
    
    input_len = 12
    output_len = 1
    
    X, Y = create_sequences(data, input_len, output_len)
    
    # Expected number of samples = Total - (In + Out) + 1
    # 20 - (12 + 1) + 1 = 8 samples
    expected_samples = 8
    
    # --- FIX 2: Expect 4D shape (Samples, Seq, Nodes, Features) ---
    # Our code adds the feature dimension automatically.
    assert X.shape == (expected_samples, input_len, 1, 1)
    assert Y.shape == (expected_samples, output_len, 1, 1)