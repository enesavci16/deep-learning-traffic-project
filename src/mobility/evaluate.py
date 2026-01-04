import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
import sys
import logging
from math import sqrt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Path düzeltme
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.append(src_dir)

from mobility.models import TrafficLSTM, STGCN_LSTM
from mobility.preprocessing import load_traffic_data, create_sequences, MinMaxNormalizer
from mobility.utils.graph import load_adj_matrix, calculate_scaled_laplacian

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate_model(model_path, data_path, adj_path, model_type, seq_len=12, horizon=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Veriyi Hazırla
    df = load_traffic_data(data_path)
    data_values = df.values.astype(np.float32)
    
    scaler = MinMaxNormalizer()
    data_normalized = scaler.fit_transform(data_values)
    
    X, y = create_sequences(data_normalized, seq_len, horizon)
    
    # Sadece TEST setini al (%20 en son kısım)
    total_len = len(X)
    test_start_index = int(0.8 * total_len)
    
    X_test = X[test_start_index:]
    y_test = y[test_start_index:] 
    
    # Tensor yap
    X_tensor = torch.FloatTensor(X_test) # Device'a henüz atma, batch batch atacağız
    y_tensor = torch.FloatTensor(y_test)

    # 2. Modeli Yükle
    num_nodes = X_tensor.shape[2]
    input_dim = X_tensor.shape[3]
    
    laplacian = None
    if model_type == 'stgcn':
        if adj_path is None:
             raise ValueError("STGCN için --adj parametresi gerekli!")
        _, adj_mx = load_adj_matrix(adj_path)
        laplacian = calculate_scaled_laplacian(adj_mx).to(device)
        model = STGCN_LSTM(num_nodes, input_dim, 64, 1, K=3).to(device)
    else:
        model = TrafficLSTM(num_nodes, 64).to(device)
        
    logger.info(f"📂 Loading weights from {model_path}...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except RuntimeError as e:
        logger.error(f"Model yüklenirken boyut uyuşmazlığı: {e}")
        return
        
    model.eval()
    
    # 3. Tahmin Yap (BATCH PROCESSING - HAFIZA DOSTU YÖNTEM)
    logger.info("🔮 Running inference on TEST set (in batches)...")
    
    # DataLoader kullanarak veriyi parçalara bölüyoruz (Batch Size 64)
    test_loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=False)
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            # batch_y şu an lazım değil ama sıralama bozulmasın diye tutuyoruz
            
            # Tahmin
            preds = model(batch_x, laplacian)
            
            # CPU'ya alıp listeye ekle (GPU hafızasını boşalt)
            all_preds.append(preds.cpu())
            all_targets.append(batch_y) # Zaten CPU'daydı
            
    # Parçaları birleştir
    preds_tensor = torch.cat(all_preds, dim=0)
    y_tensor_full = torch.cat(all_targets, dim=0)

    # 4. Boyutları Düzeltme
    # Hedef Boyut: (Total_Samples, Num_Nodes)
    
    if preds_tensor.dim() == 3:
        preds_tensor = preds_tensor.squeeze(-1)
        
    if y_tensor_full.dim() == 4:
        y_tensor_full = y_tensor_full.squeeze(-1).squeeze(1)
    elif y_tensor_full.dim() == 3:
        y_tensor_full = y_tensor_full.squeeze(-1)
        
    # Numpy'a çevir
    preds_np = preds_tensor.numpy()
    y_np = y_tensor_full.numpy()
    
    # Inverse Transform (Normalizasyonu geri al)
    preds_real = scaler.inverse_transform(preds_np)
    y_real = scaler.inverse_transform(y_np)

    # 5. Metrikleri Hesapla
    mae = mean_absolute_error(y_real, preds_real)
    rmse = sqrt(mean_squared_error(y_real, preds_real))
    
    logger.info(f"📊 --- RESULTS FOR {model_type.upper()} ---")
    logger.info(f"   MAE  : {mae:.4f}")
    logger.info(f"   RMSE : {rmse:.4f}")
    
    return mae, rmse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--adj", default=None)
    args = parser.parse_args()
    
    print("\n" + "="*40)
    print("   TRAFFIC PREDICTION BENCHMARK")
    print("="*40 + "\n")
    
    # 1. Evaluate LSTM
    if os.path.exists("best_model_lstm.pth"):
        evaluate_model("best_model_lstm.pth", args.data, None, "lstm")
    else:
        print("⚠️ best_model_lstm.pth not found. Skipping LSTM.")

    print("\n" + "-"*40 + "\n")

    # 2. Evaluate STGCN
    if os.path.exists("best_model_stgcn.pth"):
        evaluate_model("best_model_stgcn.pth", args.data, args.adj, "stgcn")
    else:
        print("⚠️ best_model_stgcn.pth not found. Skipping STGCN.")