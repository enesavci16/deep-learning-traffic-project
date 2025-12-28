import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# MODEL 1: Baseline LSTM (from project_lstm_.ipynb)
# ==========================================
class BaselineLSTM(nn.Module):
    """
    Standard LSTM model for time-series prediction.
    This serves as your baseline to compare against the GNN performance.
    """
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, output_dim: int = 1, num_layers: int = 1, dropout: float = 0.0):
        super(BaselineLSTM, self).__init__()
        
        # LSTM Layer
        # batch_first=True expects input shape: [Batch, Time Steps, Features]
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim, 
            num_layers=num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Fully Connected Layer for prediction
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        """
        x shape: [Batch, Time Steps, Nodes, Features] -> We treat nodes as batch samples here
        or [Batch, Time Steps, Features] depending on how you construct the loader.
        """
        # LSTM output: (batch, time, hidden)
        out, _ = self.lstm(x)
        
        # We only care about the last time step for prediction
        out = out[:, -1, :] 
        
        # Final prediction
        out = self.fc(out)
        return out


# ==========================================
# MODEL 2: STGCN + LSTM (from graph+lstm.ipynb)
# ==========================================
class ChebConv(nn.Module):
    """
    Spatio-Temporal Graph Convolutional layer using Chebyshev Polynomials.
    Captures spatial dependencies across the traffic network.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int):
        super(ChebConv, self).__init__()
        self.K = K
        # Learnable Weights for each hop (K)
        # Shape: [K, In_Channels, Out_Channels]
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for stability
        nn.init.xavier_uniform_(self.Theta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        x: [Batch, Nodes, In_Channels]
        laplacian: [Nodes, Nodes]
        """
        batch_size, num_nodes, in_channels = x.size()
        cheb_polys = [x]

        # 1st Order (K=1): L * x
        if self.K > 1:
            cheb_polys.append(torch.matmul(laplacian, x))

        # K-th Order Recurrence: Tk(L) = 2 * L * T{k-1} - T{k-2}
        for k in range(2, self.K):
            l_tk = torch.matmul(laplacian, cheb_polys[k-1])
            tk = 2 * l_tk - cheb_polys[k-2]
            cheb_polys.append(tk)

        # Stack into [K, Batch, Nodes, In_Channels]
        cheb_poly_tensor = torch.stack(cheb_polys, dim=0)
        
        # Einstein summation: Sum over K orders (T_k * Theta_k)
        output = torch.einsum('k b n i, k i o -> b n o', cheb_poly_tensor, self.Theta)
        
        return output + self.bias

class STGCN_LSTM(nn.Module):
    """
    Hybrid Spatio-Temporal model.
    1. Uses Graph Convolution (ChebConv) to understand road connections (Spatial).
    2. Uses LSTM to understand traffic flow over time (Temporal).
    """
    def __init__(self, num_nodes: int, in_features: int, hidden_dim: int, out_dim: int, K: int, dropout: float = 0.5):
        super(STGCN_LSTM, self).__init__()
        
        # 1. Spatial Block
        self.spatial_layer = ChebConv(in_features, hidden_dim, K=K)
        
        # 2. Temporal Block
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Output Block
        self.regressor = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        x shape: [Batch, Time_Steps, Nodes, Features]
        laplacian shape: [Nodes, Nodes]
        """
        batch_size, time_steps, num_nodes, features = x.size()

        # --- Spatial Processing (Frame by Frame) ---
        # Flatten time into batch to apply GCN to each time step independently
        x_reshaped = x.view(batch_size * time_steps, num_nodes, features)
        
        # Apply Graph Convolution
        spatial_out = torch.relu(self.spatial_layer(x_reshaped, laplacian))
        
        # Reshape back to sequence: [Batch, Time, Nodes, Hidden]
        spatial_out = spatial_out.view(batch_size, time_steps, num_nodes, -1)

        # --- Temporal Processing (LSTM) ---
        # We need to treat each Node as a separate sequence for the LSTM
        # Permute to: [Batch, Nodes, Time, Hidden]
        spatial_out = spatial_out.permute(0, 2, 1, 3) 
        
        # Merge Batch and Nodes -> [Batch * Nodes, Time, Hidden]
        lstm_input = spatial_out.reshape(batch_size * num_nodes, time_steps, -1)
        
        # Run LSTM
        # out: [Batch * Nodes, Time, Hidden], h_n: [1, Batch * Nodes, Hidden]
        _, (h_n, _) = self.lstm(lstm_input)
        
        # Take the last hidden state (the most recent memory)
        last_hidden = h_n[-1] 

        # --- Prediction ---
        output = self.regressor(last_hidden)
        
        # Reshape back to [Batch, Nodes, Out_Dim]
        return output.view(batch_size, num_nodes, -1)