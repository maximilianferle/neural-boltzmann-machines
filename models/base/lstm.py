import torch
import torch.nn as nn
from torchinfo import summary
from torch.nn.utils.rnn import pack_padded_sequence


class LSTMPredictor(nn.Module):
    def __init__(self, n_features, lstm_units):
        super(LSTMPredictor, self).__init__()
        self.lstm = nn.LSTM(n_features, lstm_units, batch_first=True)
        self.linear = nn.Linear(lstm_units, n_features)

    def forward(self, x):
        out, (last_hidden, _) = self.lstm(x)
        return self.linear(last_hidden.squeeze())


if __name__ == "__main__":
    # Example usage:
    n_features = 10
    lstm_units = 32

    model = LSTMPredictor(n_features=n_features, lstm_units=lstm_units)
    summary(model)

    # Create some dummy input data with batch_size=3 and max sequence_length=5
    dummy_input = torch.randn(3, 6, n_features)

    dummy_input = pack_padded_sequence(dummy_input, lengths=[3, 4, 6], batch_first=True, enforce_sorted=False)

    # Get the model output
    output = model(dummy_input)
    pass
