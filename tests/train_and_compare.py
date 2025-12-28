import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import pickle
from torch.utils.data import DataLoader, TensorDataset

# Import your sophisticated model
# Make sure you created src/model/gnn_lstm.py as discussed!
from src.model.gnn_lstm import ST_GCN_LSTM 

# --- 1. Define Baseline Model (Simple LSTM) ---
class SimpleLSTM(nn.Module):
    """
    A basic LSTM that treats all sensors as a flat vector.
    Used as a baseline to prove your GNN is better.
    """
    def __init__(self, num_nodes, in_channels, out_channels, hidden_size):
        super(SimpleLSTM, self).__init__()
        # Flatten input: nodes * features
        self.lstm = nn.LSTM(input_size=num_nodes * in_channels, 
                            hidden_size=hidden_size, 
                            batch_first=True)
        self.fc = nn.Linear(hidden_size, out_channels) # Predicts global average speed
    
    def forward(self, x, laplacian=None):
        # x shape: [Batch, Seq, Nodes, Feat]
        b, s, n, f = x.shape
        x_flat = x.view(b, s, n * f) # Flatten nodes into features
        out, _ = self.lstm(x_flat)
        return self.fc(out[:, -1, :]) # Take last step

# --- 2. Data Loading & Preprocessing ---
def load_data(data_path="data/metr-la.h5", adj_path="data/adj_METR-LA.pkl"):
    print(f"Loading data from {data_path}...")
    
    # 1. Load Speed Data
    try:
        df = pd.read_hdf(data_path)
    except FileNotFoundError:
        print(f"ERROR: {data_path} not found. Please move 'metr-la.h5' to Code/data/")
        exit(1)
        
    # 2. Load Adjacency Matrix (for GNN)
    if os.path.exists(adj_path):
        with open(adj_path, 'rb') as f:
            sensor_ids, sensor_id_to_ind, adj_mx = pickle.load(f)
        laplacian = torch.tensor(adj_mx, dtype=torch.float32)
    else:
        print("Warning: Adjacency matrix not found. Using Identity matrix (Model will be less accurate).")
        laplacian = torch.eye(df.shape[1])

    return df, laplacian

def create_sequences(data, seq_len=12, horizon=1):
    """
    Creates (X, Y) pairs. 
    X = 1 hour of history (12 steps)
    Y = Next 5 mins average speed (1 step)
    """
    data_array = data.values # Convert to numpy
    X, Y = [], []
    
    # Use subset for faster testing if needed, e.g., data_array[:5000]
    # For final project, use full data_array
    limit = 2000 # LIMITING TO 2000 STEPS FOR FAST DEMO. REMOVE THIS FOR REAL TRAINING!
    print(f"Generating sequences from first {limit} rows...")
    
    for i in range(limit - seq_len - horizon):
        X.append(data_array[i : i+seq_len]) 
        # Target: We predict the Global Average Speed for simplicity in this demo
        # (Real app might predict all sensors, but let's keep it simple for comparison)
        Y.append(np.mean(data_array[i+seq_len : i+seq_len+horizon]))

    X = np.array(X)[..., np.newaxis] # Add feature dim: [Batch, Seq, Nodes, 1]
    Y = np.array(Y)[..., np.newaxis] # [Batch, 1]
    
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# --- 3. Training Loop ---
def train_and_evaluate(model_name, model, train_loader, test_loader, laplacian, epochs=3):
    print(f"\n--- Training {model_name} ---")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            
            # GNN needs laplacian, LSTM ignores it
            pred = model(x, laplacian)
            
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1}: Train Loss (MSE) = {total_loss/len(train_loader):.4f}")

    # Evaluation
    model.eval()
    total_mae = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x, laplacian)
            total_mae += torch.mean(torch.abs(pred - y)).item()
            
    final_mae = total_mae / len(test_loader)
    print(f">>> {model_name} Final MAE (Mean Absolute Error): {final_mae:.4f}")
    return final_mae

# --- 4. Main Execution ---
def main():
    # A. Setup
    SEQ_LEN = 12
    BATCH_SIZE = 32
    NODES = 207 
    
    # B. Prepare Data
    df, laplacian = load_data()
    X, Y = create_sequences(df, SEQ_LEN)
    
    # Split Train/Test (80/20)
    split_idx = int(len(X) * 0.8)
    train_data = TensorDataset(X[:split_idx], Y[:split_idx])
    test_data = TensorDataset(X[split_idx:], Y[split_idx:])
    
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    
    # C. Run Experiment
    
    # 1. Baseline LSTM
    lstm = SimpleLSTM(num_nodes=NODES, in_channels=1, out_channels=1, hidden_size=64)
    mae_lstm = train_and_evaluate("Baseline LSTM", lstm, train_loader, test_loader, laplacian)
    
    # 2. Your ST-GCN-LSTM
    gnn = ST_GCN_LSTM(num_nodes=NODES, in_channels=1, out_channels=1, lstm_units=64, K=2)
    mae_gnn = train_and_evaluate("ST-GCN-LSTM", gnn, train_loader, test_loader, laplacian)
    
    # D. Compare and Save
    print("\n" + "="*30)
    print(f"COMPARISON RESULTS:")
    print(f"Simple LSTM MAE: {mae_lstm:.4f}")
    print(f"ST-GCN+LSTM MAE: {mae_gnn:.4f}")
    
    os.makedirs("models", exist_ok=True)
    if mae_gnn < mae_lstm:
        print("\nWINNER: ST-GCN-LSTM (Lower error is better)")
        print("Saving GNN model to 'models/best_model.pth'...")
        torch.save(gnn.state_dict(), "models/best_model.pth")
    else:
        print("\nWINNER: Simple LSTM (Surprisingly!)")
        print("Saving LSTM model to 'models/best_model.pth'...")
        torch.save(lstm.state_dict(), "models/best_model.pth")
    print("="*30)

if __name__ == "__main__":
    main()