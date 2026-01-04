import numpy as np
import pandas as pd
import pickle
import os
import sys

# Standard way to compute graph structure from data if map is missing:
# Calculate Pearson correlation between all sensor pairs.
# If correlation > 0.4, draw an edge.

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

    print(f"Data Loaded. Shape: {data.shape}")

    # --- DÜZELTME BAŞLANGICI (DATA LEAKAGE PREVENTION) ---
    # Verinin sadece eğitim kısmını (ilk %70) alarak korelasyon hesaplıyoruz.
    # Böylece model test setindeki (gelecekteki) trafik davranışlarını "önceden görmemiş" oluyor.
    TRAIN_RATIO = 0.7 
    num_train = int(data.shape[0] * TRAIN_RATIO)
    train_data = data[:num_train, :]

    print(f"Using only training data (first {num_train} steps) for graph generation to avoid Data Leakage.")
    # --- DÜZELTME BİTİŞİ ---

    print("Calculating correlation matrix (this may take 1-2 minutes)...")

    # 3. Calculate Correlation
    # We transpose (.T) so corrcoef works on sensors (rows) instead of time steps
    # corr_matrix shape: (num_sensors, num_sensors)
    corr_matrix = np.corrcoef(train_data.T)
    
    # 4. Thresholding to create Adjacency
    # We only keep strong connections (> 0.4 correlation)
    # The original METR-LA graph is based on distance, but correlation is a good proxy.
    threshold = 0.4
    adj_mx = np.zeros_like(corr_matrix)
    adj_mx[corr_matrix > threshold] = 1
    
    # Remove self-loops (diagonal = 0)
    # Chebyshev Conv formula usually adds Identity matrix later or handles x term explicitly.
    np.fill_diagonal(adj_mx, 0)
    
    # 5. Save in the expected format: (sensor_ids, map, matrix)
    sensor_id_to_ind = {sensor_id: i for i, sensor_id in enumerate(sensor_ids)}
    
    print(f"Graph generated! Found {np.sum(adj_mx)} connections.")
    
    with open(OUTPUT_PKL_PATH, 'wb') as f:
        # Saving as a list to match standard METR-LA format
        pickle.dump([sensor_ids, sensor_id_to_ind, adj_mx], f)
        
    print(f"✅ Saved adjacency matrix to: {OUTPUT_PKL_PATH}")

if __name__ == "__main__":
    generate_adjacency_matrix()