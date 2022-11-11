from torch import nn


class CNN(nn.Module):
    def __init__(self, input_size=1, output_size=1):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_size, out_channels=128, kernel_size=(1,)),
            nn.ReLU(),
            nn.MaxPool1d(1),
            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=(1,)),
            nn.ReLU(),
            nn.MaxPool1d(1),
        )
        self.linear = nn.Linear(256, output_size)

    def forward(self, input_seq):
        # CNN
        input = input_seq.unsqueeze(-1)
        cnn_out = self.cnn(input)
        predictions = self.linear(cnn_out[:, :, -1])
        return cnn_out[:, :, -1], predictions
