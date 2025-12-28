import torch
import pytest
import sys
import os

# Ensure we can import from 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# --- FIX: Point to the correct location 'src.mobility.models' ---
from src.mobility.models.gnn_lstm import ST_GCN_LSTM, ChebConv
from src.mobility.schemas.validation import TrafficSnapshot

# --- Test 1: Does the Graph Convolution layer output the right shape? ---
def test_cheb_conv_shape():
    # Batch=2, Nodes=5, In_Channels=1
    x = torch.randn(2, 5, 1)
    # Laplacian (Nodes x Nodes) = 5x5 identity matrix
    laplacian = torch.eye(5) 
    
    # Initialize Layer (K=2 means looking at immediate neighbors)
    conv = ChebConv(in_channels=1, out_channels=16, K=2)
    output = conv(x, laplacian)
    
    # Expected Output: [Batch, Nodes, Out_Channels]
    assert output.shape == (2, 5, 16), f"Expected (2, 5, 16), got {output.shape}"

# --- Test 2: Does the full Model work end-to-end? ---
def test_st_gcn_lstm_forward():
    batch_size = 4
    num_nodes = 10
    seq_len = 12
    in_channels = 1
    
    # Dummy Input: [Batch, Seq, Nodes, Features]
    x = torch.randn(batch_size, seq_len, num_nodes, in_channels)
    laplacian = torch.eye(num_nodes)
    
    model = ST_GCN_LSTM(num_nodes=num_nodes, in_channels=in_channels, 
                        out_channels=1, lstm_units=32, K=2)
    
    output = model(x, laplacian)
    
    # Expected Output: [Batch, 1] (Predicting 1 value per batch sample)
    assert output.shape == (batch_size, 1), f"Expected ({batch_size}, 1), got {output.shape}"

# --- Test 3: Does Pydantic catch negative speeds? ---
def test_negative_speed_fails():
    data = {
        "timestamp": "2023-12-01T12:00:00",
        "sensor_readings": {"sensor_1": -5.0} # Negative speed!
    }
    # This should raise a ValueError
    with pytest.raises(ValueError):
        TrafficSnapshot(**data)

# --- Test 4: Does Pydantic accept valid data? ---
def test_valid_traffic_snapshot():
    data = {
        "timestamp": "2023-12-01T12:00:00",
        "sensor_readings": {"sensor_1": 65.5, "sensor_2": 40.0}
    }
    snapshot = TrafficSnapshot(**data)
    assert snapshot.sensor_readings["sensor_1"] == 65.5