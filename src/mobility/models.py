import torch
import torch.nn as nn

class TrafficLSTM(nn.Module):
    """
    LSTM Model for Traffic Prediction (Baseline Model).
    As defined in the project proposal, this model uses historical data
    to predict future traffic speeds.
    """
    def __init__(self, num_sensors: int, hidden_dim: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(TrafficLSTM, self).__init__()
        
        self.num_sensors = num_sensors
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM Layer
        # input_size = num_sensors (We treat all sensors as features at one time step)
        # batch_first=True ensures input format is (Batch, Sequence, Features)
        self.lstm = nn.LSTM(
            input_size=num_sensors,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )
        
        # Fully Connected Layer to map hidden state to output (prediction)
        # We map the hidden dimension back to the number of sensors
        self.fc = nn.Linear(hidden_dim, num_sensors)

    def forward(self, x):
        # x shape: [batch_size, seq_len, num_sensors]
        
        # Initialize hidden states (Short-term memory) and Cell states (Long-term memory)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        # Forward propagate LSTM
        # out shape: [batch, seq_len, hidden_dim]
        out, _ = self.lstm(x, (h0, c0))
        
        # We only care about the last time step to predict the future (Many-to-One architecture)
        # This aligns with the goal of using 12 past steps to predict the next 1 step [cite: 46]
        last_time_step = out[:, -1, :]
        
        # Decode the hidden state of the last time step to get speed values
        prediction = self.fc(last_time_step)
        
        return prediction