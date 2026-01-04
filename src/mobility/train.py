import sys
import os

# --- PATH DÜZELTME ---
# Bu blok, terminali nereden açarsan aç Python'ın 'src' klasörünü bulmasını sağlar.
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../src/mobility
src_dir = os.path.dirname(current_dir) # .../src
sys.path.append(src_dir)
# ---------------------

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import pandas as pd
import logging

# Modelleri ve yardımcı fonksiyonları çağırıyoruz
from mobility.models import TrafficLSTM, STGCN_LSTM
from mobility.preprocessing import load_traffic_data, create_sequences, MinMaxNormalizer
from mobility.utils.graph import load_adj_matrix, calculate_scaled_laplacian

# Logging ayarı
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TrafficTrainer:
    def __init__(self, data_path, adj_path=None, model_type='lstm', seq_len=12, horizon=1, batch_size=64, hidden_dim=64, lr=0.001):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Training on device: {self.device}")
        
        self.model_type = model_type.lower()
        self.seq_len = seq_len
        self.horizon = horizon
        self.batch_size = batch_size
        
        # 1. Veri Yükleme
        logger.info(f"⏳ Loading traffic data from {data_path}...")
        try:
            df = load_traffic_data(data_path)
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise e

        data_values = df.values.astype(np.float32)
        
        # 2. Normalizasyon
        self.scaler = MinMaxNormalizer()
        data_normalized = self.scaler.fit_transform(data_values)
        
        # 3. Sequence Oluşturma
        logger.info("✂️ Creating time sequences...")
        X, y = create_sequences(data_normalized, seq_len, horizon)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).squeeze(1) 
        
        # 4. Veri Setini Bölme (%70 Train, %10 Val, %20 Test)
        total_len = len(X_tensor)
        train_size = int(0.7 * total_len)
        val_size = int(0.1 * total_len)
        
        logger.info(f"📊 Split: Train={train_size}, Val={val_size}, Test={total_len - train_size - val_size}")

        self.train_loader = DataLoader(
            TensorDataset(X_tensor[:train_size], y_tensor[:train_size]), 
            batch_size=batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            TensorDataset(X_tensor[train_size:train_size+val_size], y_tensor[train_size:train_size+val_size]), 
            batch_size=batch_size
        )

        # 5. Graf Yapısını Yükleme (Sadece STGCN için)
        self.laplacian = None
        if self.model_type == 'stgcn':
            if adj_path is None or not os.path.exists(adj_path):
                raise ValueError("❌ STGCN model requires a valid --adj path!")
            
            logger.info(f"🕸️ Loading Graph from {adj_path}...")
            _, adj_mx = load_adj_matrix(adj_path)
            
            L_tensor = calculate_scaled_laplacian(adj_mx)
            self.laplacian = L_tensor.to(self.device)
            logger.info("✅ Laplacian Matrix calculated and moved to GPU.")

        # 6. Model Başlatma
        num_nodes = X_tensor.shape[2]
        input_dim = X_tensor.shape[3]
        
        if self.model_type == 'stgcn':
            logger.info("🧠 Initializing STGCN + LSTM Model...")
            self.model = STGCN_LSTM(
                num_nodes=num_nodes,
                in_features=input_dim,
                hidden_dim=hidden_dim,
                out_dim=1,
                K=3
            ).to(self.device)
            
        elif self.model_type == 'lstm':
            logger.info("🧠 Initializing Baseline LSTM Model...")
            self.model = TrafficLSTM(
                num_sensors=num_nodes,
                hidden_dim=hidden_dim
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def train(self, epochs=10):
        logger.info(f"🔥 Starting training for {epochs} epochs...")
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                # Model tahmini
                outputs = self.model(X_batch, self.laplacian)
                
                if y_batch.dim() == 2:
                    y_batch = y_batch.unsqueeze(-1)
                    
                loss = self.criterion(outputs, y_batch)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
            
            # Validation
            val_loss = self.validate()
            avg_train_loss = train_loss / len(self.train_loader)
            
            logger.info(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_name = f"best_model_{self.model_type}.pth"
                torch.save(self.model.state_dict(), save_name)
                logger.info(f"  💾 New best model saved: {save_name}")

    def validate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch, self.laplacian)
                if y_batch.dim() == 2:
                    y_batch = y_batch.unsqueeze(-1)
                loss = self.criterion(outputs, y_batch)
                val_loss += loss.item()
        return val_loss / len(self.val_loader)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True, help="Path to .h5 file")
    parser.add_argument("--adj", type=str, default=None, help="Path to .pkl adjacency file")
    parser.add_argument("--model", type=str, default="lstm", choices=["lstm", "stgcn"], help="Model type")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs")
    
    args = parser.parse_args()
    
    trainer = TrafficTrainer(
        data_path=args.data, 
        adj_path=args.adj, 
        model_type=args.model
    )
    
    trainer.train(epochs=args.epochs)