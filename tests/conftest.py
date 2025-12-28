import pytest
import numpy as np
import pandas as pd
import pickle
import os

@pytest.fixture(scope="session")
def mock_data_dir(tmp_path_factory):
    """
    Creates a temporary directory for storing mock data files during tests.
    """
    return tmp_path_factory.mktemp("data")

@pytest.fixture(scope="session")
def mock_adj_matrix(mock_data_dir):
    """
    Creates a dummy adjacency matrix (sensor graph) and saves it as a .pkl file.
    Structure: (sensor_ids, sensor_id_to_ind, adjacency_matrix)
    """
    num_sensors = 5
    ids = [str(i) for i in range(num_sensors)]
    sensor_map = {id: i for i, id in enumerate(ids)}
    # Random adjacency matrix (5x5)
    adj_mx = np.random.rand(num_sensors, num_sensors)
    
    file_path = mock_data_dir / "adj_mx.pkl"
    
    with open(file_path, "wb") as f:
        pickle.dump((ids, sensor_map, adj_mx), f, protocol=2)
        
    return file_path

@pytest.fixture(scope="session")
def mock_traffic_data(mock_data_dir):
    """
    Creates dummy traffic speed data and saves it as an .h5 file.
    Shape: (100 time steps, 5 sensors)
    """
    num_sensors = 5
    num_samples = 100
    
    # Random speeds between 0 and 70 mph
    data = np.random.rand(num_samples, num_sensors) * 70
    df = pd.DataFrame(data, columns=[str(i) for i in range(num_sensors)])
    
    file_path = mock_data_dir / "metr-la.h5"
    
    # Save as HDF5
    df.to_hdf(file_path, key='df')
    
    return file_path