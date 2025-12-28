import numpy as np
import pandas as pd
import pickle
import os
import sys

# Standard way to compute graph structure from data if map is missing:
# Calculate Pearson correlation between all sensor pairs.
# If correlation > 0.5, draw an edge.

def generate_adjacency_matrix():
    # 1. Setup Paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # src/mobility/utils
    DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '../data'))
    
    H5_PATH = os.path.join(DATA_DIR, 'metr-la.h5')
    OUTPUT_PKL_PATH = os.path.join(DATA_DIR, 'adj_METR-LA.pkl')

    print(f"Reading data from: {H5_PATH}")
    if not os.path.exists(H5_PATH):
        print("❌ Error: metr-la.h5 not found!")
        return

    # 2. Load Data
    df = pd.read_hdf(H5_PATH)
    data = df.values
    sensor_ids = df.columns.astype(str).tolist()
    
    print(f"Data Loaded. Shape: {data.shape} (Time x Sensors)")
    print("Calculating correlation matrix (this may take 1-2 minutes)...")

    # 3. Calculate Correlation
    # We transpose to (Sensors x Time) so corrcoef works on sensors
    corr_matrix = np.corrcoef(data.T)
    
    # 4. Thresholding to create Adjacency
    # We only keep strong connections (> 0.4 correlation)
    # The original METR-LA graph is based on distance, but correlation is a good proxy.
    threshold = 0.4
    adj_mx = np.zeros_like(corr_matrix)
    adj_mx[corr_matrix > threshold] = 1
    
    # Remove self-loops (optional, but standard for some GCNs)
    np.fill_diagonal(adj_mx, 0)
    
    # 5. Save in the expected format: (sensor_ids, map, matrix)
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    
    print(f"Graph generated! Found {np.sum(adj_mx)} connections.")
    
    with open(OUTPUT_PKL_PATH, 'wb') as f:
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
        
    print(f"✅ Saved adjacency matrix to: {OUTPUT_PKL_PATH}")

if __name__ == "__main__":
    generate_adjacency_matrix()