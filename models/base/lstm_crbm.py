import torch
import torch.nn as nn
from models.base.crbm import CRBM
from torchinfo import summary
from torch.nn.utils.rnn import pack_padded_sequence


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)

    def forward(self, x):
        out, (last_hidden, _) = self.lstm(x)
        return last_hidden.squeeze()


class LSTMCRBM(nn.Module):

    def __init__(self,
                 n_features: int,
                 lstm_units: int,
                 crbm_hidden: int,
                 ):
        super().__init__()

        self.lstm = LSTM(input_size=n_features, hidden_size=lstm_units)
        self.crbm = CRBM(nx=lstm_units,
                         ny=n_features,
                         nh=crbm_hidden)

    def _sample(self,
                x,
                mc_steps: int = 32,
                denoise: bool = False,
                ):
        x = self.lstm(x)
        y_model, bias, precision, weights = self.crbm._sample(x=x, mc_steps=mc_steps, denoise=denoise)
        return y_model, bias, precision, weights

    def _sample_multi_step(self,
                           x,
                           mc_steps: int = 32,
                           denoise: bool = False,
                           steps: int = None,
                           ):
        assert steps, "Variable `steps` cannot be None when calling `_sample_multi_step()`."
        # TODO

    @torch.no_grad()
    def sample(self,
               x,
               mc_steps: int = 32,
               denoise: bool = False,
               steps: int = None,
               ):
        if steps is None:
            y_model, *_ = self._sample(x=x, mc_steps=mc_steps, denoise=denoise)
        else:
            y_model, *_ = self._sample_multi_step(x=x, mc_steps=mc_steps, denoise=denoise, steps=steps)
        return y_model

    def forward(self, x):
        return self.sample(x=x, mc_steps=32, denoise=True)

    def compute_loss(self, y, x, mc_steps, denoise: bool = False):
        y_model, bias, precision, weights = self._sample(x=x, mc_steps=mc_steps, denoise=denoise)
        return self.crbm._compute_loss(y=y, y_model=y_model, bias=bias, precision=precision, weights=weights)


if __name__ == "__main__":
    # Example usage:
    n_features = 10
    lstm_units = 32
    crbm_hidden = 16

    model = LSTMCRBM(n_features=n_features, lstm_units=lstm_units, crbm_hidden=crbm_hidden)
    summary(model)

    # Create some dummy input data with batch_size=3 and max sequence_length=5
    dummy_input = torch.randn(3, 6, n_features)

    dummy_input = pack_padded_sequence(dummy_input, lengths=[3, 4, 6], batch_first=True, enforce_sorted=False)

    # Get the model output
    output = model(dummy_input)
