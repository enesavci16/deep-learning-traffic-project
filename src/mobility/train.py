import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import pandas as pd
from pathlib import Path

# Import our custom modules
from mobility.models import TrafficLSTM
# Ensure you have these functions in preprocessing.py. 
# If not, I can provide the code for them.
from mobility.preprocessing import load_traffic_data, create_sequences, MinMaxNormalizer

class TrafficTrainer:
    def __init__(self, data_path, seq_len=12, horizon=1, batch_size=64, hidden_dim=64, lr=0.001):
        # Detect GPU (Real-World optimization)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Training on device: {self.device}")
        
        # Hyperparameters
        self.seq_len = seq_len      # Look back 12 steps (1 hour)
        self.horizon = horizon      # Predict 1 step ahead (5 mins)
        self.batch_size = batch_size
        
        # 1. Load Data
        print(f"⏳ Loading data from {data_path}...")
        try:
            # We use the function you already tested
            df = load_traffic_data(data_path)
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise e

        data_values = df.values.astype(np.float32)
        
        # 2. Normalize Data (Critical for LSTM)
        self.scaler = MinMaxNormalizer()
        data_normalized = self.scaler.fit_transform(data_values)
        
        # 3. Create Sequences (Sliding Window)
        # Input: (Samples, 12, Sensors) -> Output: (Samples, 1, Sensors)
        print("✂️ Creating time sequences...")
        X, y = create_sequences(data_normalized, seq_len, horizon)
        
        # Convert to PyTorch Tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).squeeze(1) # Target shape: (Batch, Sensors)
        
        # 4. Split Data (70% Train, 10% Val, 20% Test)
        total_len = len(X_tensor)
        train_size = int(0.7 * total_len)
        val_size = int(0.1 * total_len)
        
        print(f"📊 Dataset Split: Train={train_size}, Val={val_size}, Test={total_len - train_size - val_size}")

        self.train_loader = DataLoader(
            TensorDataset(X_tensor[:train_size], y_tensor[:train_size]), 
            batch_size=batch_size, 
            shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(X_tensor[train_size:train_size+val_size], y_tensor[train_size:train_size+val_size]), 
            batch_size=batch_size
        )
        
        # 5. Initialize Model
        num_sensors = X_tensor.shape[2]
        self.model = TrafficLSTM(num_sensors=num_sensors, hidden_dim=hidden_dim).to(self.device)
        
        # Optimizer & Loss
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=10):
        print(f"🔥 Starting training for {epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Forward pass
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation Step
            val_loss = self.validate()
            avg_train_loss = train_loss / len(self.train_loader)
            
            print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            # Save Best Model (Artifact generation)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), "best_model.pth")
                print("   💾 New best model saved!")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .h5 file")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    args = parser.parse_args()
    
    # FIX: Do not pass 'epochs' here. The constructor doesn't accept it.
    trainer = TrafficTrainer(data_path=args.data) 
    
    # Pass 'epochs' here instead.
    trainer.train(epochs=args.epochs)