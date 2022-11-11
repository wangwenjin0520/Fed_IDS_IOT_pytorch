import torch
from torch import nn


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1, num_layers=1, dropout=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_layer_size,
                            num_layers=num_layers,
                            dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = len(input_seq)
        lstm_out, hidden_cell_n = self.lstm(input_seq.view(1, batch_size, -1), None)
        predictions = self.linear(lstm_out[-1, :, :])
        return lstm_out[-1, :, :], predictions
