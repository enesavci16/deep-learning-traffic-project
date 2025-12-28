import pickle
import numpy as np
import torch
import logging

logger = logging.getLogger(__name__)

def load_adj_matrix(pkl_filename: str):
    """
    Loads the adjacency matrix from the sensor pickle file.
    """
    try:
        with open(pkl_filename, 'rb') as f:
            # The original file contains: (sensor_ids, sensor_id_to_ind_map, adjacency_matrix)
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f, encoding='latin1')
        return sensor_ids, adj_mx
    except Exception as e:
        logger.error(f"Failed to load adjacency matrix: {e}")
        raise

def calculate_scaled_laplacian(adj_mx: np.ndarray) -> torch.Tensor:
    """
    Transforms the raw Adjacency Matrix (A) into the Scaled Laplacian (L_tilde)
    required by the ChebConv layer in STGCN.
    """
    # 1. Handle potential infinite/NaN values
    adj_mx[np.isinf(adj_mx)] = 0
    adj_mx[np.isnan(adj_mx)] = 0

    # 2. Calculate Degree Matrix (D)
    degree = np.sum(adj_mx, axis=1)
    degree[degree == 0] = 1e-5  # Prevent division by zero

    # 3. Calculate Normalized Laplacian: L = I - D^(-1/2) * A * D^(-1/2)
    d_inv_sqrt = np.power(degree, -0.5)
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    
    adj_normalized = np.dot(np.dot(d_mat_inv_sqrt, adj_mx), d_mat_inv_sqrt)
    L = np.eye(adj_mx.shape[0]) - adj_normalized

    # 4. Scale for Chebyshev: L_tilde = (2L / lambda_max) - I
    # We assume lambda_max approx 2.0 for simplicity in this dataset
    lambda_max = 2.0
    L_tilde = (2 * L / lambda_max) - np.eye(adj_mx.shape[0])

    return torch.FloatTensor(L_tilde)