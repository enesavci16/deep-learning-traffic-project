import torch
import torch.nn as nn
import torch.nn.functional as F

class ChebConv(nn.Module):
    """
    Standard Graph Convolution using Chebyshev Polynomials.
    """
    def __init__(self, in_channels, out_channels, K):
        super(ChebConv, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Learnable Weights: One weight matrix per Chebyshev order k
        self.weights = nn.Parameter(torch.FloatTensor(K, in_channels, out_channels))
        self.bias = nn.Parameter(torch.FloatTensor(out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.zeros_(self.bias)

    def forward(self, x, laplacian):
        """
        x: [batch, seq_len, num_nodes, in_channels]
        laplacian: [num_nodes, num_nodes]
        """
        # If input has sequence dimension, merge it with batch for spatial processing
        if x.dim() == 4:
            batch_size, seq_len, num_nodes, in_channels = x.size()
            x = x.view(batch_size * seq_len, num_nodes, in_channels)
            restore_shape = True
        else:
            batch_size, num_nodes, in_channels = x.size()
            restore_shape = False

        # 1. Chebyshev Recurrence
        # T_0(L) * x = x
        # T_1(L) * x = L * x
        # T_k(L) * x = 2 * L * T_{k-1} - T_{k-2}
        
        # Initialize output accumulator
        # Shape: [Batch*Seq, Nodes, Out]
        output = torch.zeros(x.size(0), num_nodes, self.out_channels).to(x.device)
        
        # T_0(x) = x
        T_0 = x # [Batch*Seq, Nodes, In]
        
        # Compute term k=0
        # (Batch*Seq, Nodes, In) x (In, Out) -> (Batch*Seq, Nodes, Out)
        output += torch.matmul(T_0, self.weights[0])
        
        if self.K > 1:
            # T_1(x) = L * x
            # (Nodes, Nodes) x (Batch*Seq, Nodes, In) 
            # We need to permute for matmul: (Batch*Seq, In, Nodes) is harder, 
            # let's reshape L to broadcast or iterate.
            
            # Simple approach: L * X for each sample. 
            # Optimized: (Batch*Seq, Nodes, In) -> permute -> (In, Batch*Seq, Nodes)
            # This is complex to implement from scratch efficiently without PyG.
            # Let's stick to the simplest working math for the project:
            
            # L is (Nodes, Nodes), x is (Batch*Seq, Nodes, In)
            # Result = (Batch*Seq, Nodes, In)
            T_1 = torch.matmul(laplacian, T_0)
            output += torch.matmul(T_1, self.weights[1])
            
        # Recurrence for K >= 2
        T_prev = T_1
        T_prev_prev = T_0
        
        for k in range(2, self.K):
            # T_k = 2 * L * T_{k-1} - T_{k-2}
            term1 = 2 * torch.matmul(laplacian, T_prev)
            T_k = term1 - T_prev_prev
            
            output += torch.matmul(T_k, self.weights[k])
            
            T_prev_prev = T_prev
            T_prev = T_k

        output = output + self.bias
        
        if restore_shape:
            output = output.view(batch_size, seq_len, num_nodes, self.out_channels)
            
        return output

class ST_GCN_LSTM(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, lstm_units, K, lstm_layers=1):
        super(ST_GCN_LSTM, self).__init__()
        
        # 1. Spatial Block (Graph Convolution)
        self.cheb_conv = ChebConv(in_channels, 64, K)
        
        # 2. Temporal Block (LSTM)
        # Input: 64 spatial features per node
        # We process the graph average (or flatten) to feed LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_units, 
                            num_layers=lstm_layers, batch_first=True)
        
        # 3. Output Layer
        self.fc = nn.Linear(lstm_units, out_channels)

    def forward(self, x, laplacian):
        """
        x shape: [batch, seq_len, num_nodes, in_channels]
        """
        batch_size, seq_len, num_nodes, in_features = x.size()
        
        # A. Apply GCN to extract spatial features
        # Out: [batch, seq_len, num_nodes, 64]
        spatial_features = self.cheb_conv(x, laplacian)
        spatial_features = F.relu(spatial_features)
        
        # B. Pool features to get a "Whole City Traffic State" vector
        # Average over all nodes: [batch, seq_len, 64]
        # (This simplifies the problem to predicting global state, 
        # but makes it compatible with standard LSTM easily)
        x_pooled = torch.mean(spatial_features, dim=2)
        
        # C. Temporal Processing (LSTM)
        # Out: [batch, seq_len, lstm_units]
        lstm_out, _ = self.lstm(x_pooled)
        
        # D. Prediction (Take last time step)
        last_step = lstm_out[:, -1, :]
        prediction = self.fc(last_step)
        
        return prediction