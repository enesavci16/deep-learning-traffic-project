import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import os
from torch.utils.data import DataLoader, TensorDataset

# --- PATH FIX ---
# Add project root to sys.path so we can import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(current_dir, '../../../')) 
sys.path.append(src_dir)

from src.mobility.models.gnn_lstm import ST_GCN_LSTM
from src.mobility.preprocessing import load_traffic_data, load_adjacency_matrix, create_sequences, MinMaxNormalizer

# --- 1. Define Baseline Model (Simple LSTM) ---
class SimpleLSTM(nn.Module):
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

# --- 2. Training Helper ---
def train_and_evaluate(model_name, model, train_loader, test_loader, laplacian, epochs=3):
    print(f"\n--- Training {model_name} ---")
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            optimizer.zero_grad()
            
            # --- SHAPE FIX: Ensure Target matches Model Output ---
            # y comes in as [Batch, 1, Nodes, 1] or [Batch, 1, 1]
            # Model outputs [Batch, 1]
            y = y.view(y.size(0), -1) # Flatten to [Batch, 1]
            
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
            y = y.view(y.size(0), -1) # Flatten target here too
            pred = model(x, laplacian)
            # MAE Calculation
            total_mae += torch.mean(torch.abs(pred - y)).item()
            
    final_mae = total_mae / len(test_loader)
    print(f">>> {model_name} Final MAE: {final_mae:.4f}")
    return final_mae

# --- 3. Main Execution ---
def main():
    # Paths (Relative to this script location)
    # This script is in src/mobility/models/
    # Data is in src/mobility/data/
    BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
    DATA_DIR = os.path.join(BASE_DIR, '../data') # Go up one level to 'mobility', then into 'data'
    
    DATA_PATH = os.path.join(DATA_DIR, "metr-la.h5")
    ADJ_PATH = os.path.join(DATA_DIR, "adj_METR-LA.pkl")
    
    # Models save to Code/models/ (Project Root/models)
    SAVE_DIR = os.path.abspath(os.path.join(BASE_DIR, '../../../models'))
    
    # 1. Load Data
    print(f"Loading Data from {DATA_PATH}...")
    try:
        df = load_traffic_data(DATA_PATH)
        adj_mx = load_adjacency_matrix(ADJ_PATH)
    except Exception as e:
        print(f"Error: {e}")
        print("Please ensure 'metr-la.h5' is in 'Code/src/mobility/data/' folder.")
        return

    # Handle missing Adjacency Matrix
    num_nodes = df.shape[1]
    if adj_mx is None:
        print("Warning: Adjacency matrix not found. Using Identity.")
        laplacian = torch.eye(num_nodes)
    else:
        laplacian = torch.tensor(adj_mx, dtype=torch.float32)

    # 2. Preprocess
    print("Preprocessing...")
    scaler = MinMaxNormalizer()
    scaler.fit(df.values)
    data_norm = scaler.transform(df.values)
    
    # Use subset for fast testing (remove [:2000] for real training if you want)
   # Use 10,000 steps or full data (remove slice)
# Note: Full data will take longer to train!
    X, Y = create_sequences(data_norm[:10000], seq_len=12, horizon=1)
    
    # Average the target Y across nodes (axis 2) to match the model output
    Y = np.mean(Y, axis=2) 
    
    # Split
    split = int(len(X) * 0.8)
    train_data = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32))
    test_data = TensorDataset(torch.tensor(X[split:], dtype=torch.float32), torch.tensor(Y[split:], dtype=torch.float32))
    
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

    # 3. Run Experiments
    
    # Model A: Baseline LSTM
    lstm = SimpleLSTM(num_nodes=num_nodes, in_channels=1, out_channels=1, hidden_size=64)
    mae_lstm = train_and_evaluate("Baseline LSTM", lstm, train_loader, test_loader, laplacian)
    
    # Model B: ST-GCN-LSTM
    gnn = ST_GCN_LSTM(num_nodes=num_nodes, in_channels=1, out_channels=1, lstm_units=64, K=2)
    mae_gnn = train_and_evaluate("ST-GCN-LSTM", gnn, train_loader, test_loader, laplacian)
    
    # 4. Compare & Save
    print("\n" + "="*30)
    print(f"RESULTS:")
    print(f"LSTM MAE: {mae_lstm:.4f}")
    print(f"GNN  MAE: {mae_gnn:.4f}")
    
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    # --- FORCE SAVE GNN FOR CONSUMER APP ---
    gnn_save_path = os.path.join(SAVE_DIR, "gnn_model.pth")
    torch.save(gnn.state_dict(), gnn_save_path)
    print(f"✅ Saved ST-GCN model to {gnn_save_path} (For Consumer App)")

    # Also save the "best" one generally
    best_save_path = os.path.join(SAVE_DIR, "best_model.pth")
    if mae_gnn < mae_lstm:
        print("\n🏆 WINNER: ST-GCN-LSTM")
        torch.save(gnn.state_dict(), best_save_path)
    else:
        print("\n🏆 WINNER: Baseline LSTM")
        torch.save(lstm.state_dict(), best_save_path)
    
    print("="*30)

if __name__ == "__main__":
    main()