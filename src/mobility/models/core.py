import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. BASELINE MODEL: TrafficLSTM
# (Sadece Zamansal İlişkiyi Öğrenir)
# ==========================================
class TrafficLSTM(nn.Module):
    """
    LSTM Model for Traffic Prediction (Baseline Model).
    As defined in the project proposal, this model uses historical data
    to predict future traffic speeds.
    It treats all sensors as a flat vector, ignoring the graph structure.
    """
    def __init__(self, num_sensors: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(TrafficLSTM, self).__init__()
        
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # input_size = num_sensors (We treat all sensors as features at one time step)
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Fully Connected Layer to map hidden state to output (prediction)
        self.fc = nn.Linear(hidden_dim, num_sensors)

    def forward(self, x, laplacian=None):
        """
        x shape: [batch_size, seq_len, num_nodes, features] OR [batch, seq, num_nodes]
        laplacian: Not used here (kept for compatibility with training loop)
        """
        # Data Shape Correction:
        # If input is 4D [Batch, Seq, Nodes, Features], squeeze it to [Batch, Seq, Nodes]
        # because standard LSTM expects features (nodes) in the last dimension.
        if x.dim() == 4:
            x = x.squeeze(-1)
        
        # Initialize hidden states
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out shape: [batch, seq_len, hidden_dim]
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the last time step (Many-to-One architecture)
        last_time_step = out[:, -1, :]
        
        # Decode the hidden state to get speed values
        prediction = self.fc(last_time_step)
        
        # Output shape: [Batch, Nodes]
        # We reshape to [Batch, Nodes, 1] to match STGCN output format for loss calculation
        return prediction.unsqueeze(-1)


# ==========================================
# 2. PROPOSED MODEL: STGCN_LSTM
# (Hem Zamansal Hem Uzamsal İlişkiyi Öğrenir)
# ==========================================
class ChebConv(nn.Module):
    """
    Spatio-Temporal Graph Convolutional layer using Chebyshev Polynomials.
    """
    def __init__(self, in_channels: int, out_channels: int, K: int):
        super(ChebConv, self).__init__()
        self.K = K
        self.Theta = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.Theta)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        # x: [Batch, Nodes, In_Channels]
        # laplacian: [Nodes, Nodes]
        
        cheb_polys = [x]
        if self.K > 1:
            cheb_polys.append(torch.matmul(laplacian, x))

        for k in range(2, self.K):
            l_tk = torch.matmul(laplacian, cheb_polys[k-1])
            tk = 2 * l_tk - cheb_polys[k-2]
            cheb_polys.append(tk)

        cheb_poly_tensor = torch.stack(cheb_polys, dim=0)
        # Efficient Einstein Summation for Graph Convolution
        output = torch.einsum('k b n i, k i o -> b n o', cheb_poly_tensor, self.Theta)
        return output + self.bias

class STGCN_LSTM(nn.Module):
    """
    Hybrid Model: ChebNet (Spatial) + LSTM (Temporal)
    This is the proposed model in the PhD thesis.
    """
    def __init__(self, num_nodes: int, in_features: int, hidden_dim: int, out_dim: int, K: int, dropout: float = 0.5):
        super(STGCN_LSTM, self).__init__()
        
        # 1. Spatial Block (Extracts features from neighbors)
        self.spatial_layer = ChebConv(in_features, hidden_dim, K=K)
        
        # 2. Temporal Block (Learns time patterns per node)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        
        # 3. Output Block
        self.regressor = nn.Linear(hidden_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, laplacian: torch.Tensor) -> torch.Tensor:
        """
        x shape: [Batch, Time_Steps, Nodes, Features]
        """
        batch_size, time_steps, num_nodes, features = x.size()

        # --- Spatial Processing ---
        # Flatten time to apply GCN to each timestamp independently
        x_reshaped = x.view(batch_size * time_steps, num_nodes, features)
        
        # GCN Operation
        spatial_out = torch.relu(self.spatial_layer(x_reshaped, laplacian))
        
        # Restore Sequence: [Batch, Time, Nodes, Hidden]
        spatial_out = spatial_out.view(batch_size, time_steps, num_nodes, -1)

        # --- Temporal Processing ---
        # Permute for LSTM: We want [Batch * Nodes, Time, Hidden]
        # Critical Step: Treating each node as a separate time-series sequence
        spatial_out = spatial_out.permute(0, 2, 1, 3) 
        lstm_input = spatial_out.reshape(batch_size * num_nodes, time_steps, -1)
        
        _, (h_n, _) = self.lstm(lstm_input)
        last_hidden = h_n[-1] 

        # --- Prediction ---
        output = self.regressor(last_hidden)
        
        # Output shape: [Batch, Nodes, Out_Dim]
        return output.view(batch_size, num_nodes, -1)