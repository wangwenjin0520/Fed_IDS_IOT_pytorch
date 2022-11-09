import torch
from torch import nn


class GRU(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, output_size=1, dropout=0, num_layers=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_layer_size = hidden_layer_size
        self.gru = nn.GRU(input_size=input_size,
                          hidden_size=hidden_layer_size,
                          num_layers=num_layers,
                          dropout=dropout)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        batch_size = len(input_seq)
        # hidden_cell = torch.zeros(self.num_layers, batch_size, self.hidden_layer_size)
        gru_out, hidden_cell_n = self.gru(input_seq.view(1, batch_size, -1), None)
        predictions = self.linear(gru_out[-1, :, :])
        return gru_out[-1, :, :], predictions, hidden_cell_n
