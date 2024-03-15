import math

import torch
from torch import nn
from torch.nn.functional import mse_loss, cosine_similarity
from torchinfo import summary


def logcosh(x):
    return x - math.log(2.0) + torch.nn.functional.softplus(-2.0 * x)


def visible_times_weights(visible, weights):
    return torch.bmm(visible.unsqueeze(dim=1), weights).squeeze(dim=1)


def hidden_times_weights(hidden, weights):
    return torch.bmm(weights, hidden.unsqueeze(dim=2)).squeeze(dim=2)


class Bias(nn.Module):

    def __init__(self, nx, ny):
        super().__init__()

        net = nn.Linear(nx, ny)
        self.net = net

    def forward(self, x):
        return self.net(x)


class Precision(nn.Module):
    def __init__(self, nx, ny, pmin=1e-3, pmax=1e3):
        super().__init__()

        self.net = nn.Linear(nx, ny)

        self.lpmin = math.log(pmin)
        self.lpmax = math.log(pmax)

    def forward(self, x):
        return self.net(x).clip(self.lpmin, self.lpmax).exp()


class Weights(nn.Module):

    def __init__(self, nx, ny, nh):
        super().__init__()
        self._ny = ny
        self._nh = nh
        self._norm = math.sqrt(self._ny)
        self.net = nn.Linear(nx, ny * nh)

    def forward(self, x):
        return self.net(x).reshape((-1, self._ny, self._nh)) / self._norm


class CRBM(nn.Module):
    def __init__(self, nx, ny, nh):
        super().__init__()

        self.bias_net = Bias(nx, ny)
        self.precision_net = Precision(nx, ny)
        self.weights_net = Weights(nx, ny, nh)

    @staticmethod
    def _free_energy(y, bias, precision, weights):
        diff = y - bias
        self_energy = 0.5 * (diff * precision * diff).sum(dim=-1)
        phi = logcosh(visible_times_weights(diff, weights)).sum(dim=-1)
        free_energy = self_energy - phi
        return free_energy

    @torch.no_grad()
    def _sample_hid(self, y, bias, precision, weights):
        diff = y - bias
        logits = visible_times_weights(diff, weights)
        proba = torch.sigmoid(2.0 * logits)
        sample = 2.0 * torch.bernoulli(proba) - 1.0
        return sample

    @torch.no_grad()
    def _sample_vis(self, h, bias, precision, weights, denoise=False):
        field = hidden_times_weights(h, weights)
        y = bias + field / precision
        if not denoise:
            noise = torch.randn_like(bias)
            y = y + noise / precision.sqrt()
        return y

    def _compute_mse(self, y_flat, y_model, bias, precision, weights):
        y_true = y_flat.clone()
        y_true.requires_grad = True
        y_pred = y_model.clone()
        y_pred.requires_grad = True

        return torch.pow(y_true - y_pred, 2).mean()

    def _compute_loss(self, y, y_model, bias, precision, weights):
        y_flat = torch.flatten(y, start_dim=1)
        pos_phase = self._free_energy(y_flat, bias, precision, weights)
        neg_phase = self._free_energy(y_model, bias, precision, weights)
        CD_loss = (pos_phase - neg_phase).mean()

        # This is a supplemental loss to understand how the bias net is learning
        diff = y_flat - bias
        bias_mse = (diff * diff).mean()

        # This is a supplemental loss to understand how the precision net is learning
        # It is more natural to interpret as a variance instead of as precision
        variance = 1 / precision
        y_variance = torch.pow((y_flat - y_flat.mean()), 2)
        diff = y_variance - variance
        variance_mse = (diff * diff).mean()

        pred_mse = self._compute_mse(y_flat, y_model, bias, precision, weights)
        cos_sim = cosine_similarity(y_flat, y_model).mean()

        return {
            "CD_loss": CD_loss,
            "MSE_bias_loss": bias_mse,
            "MSE_variance_loss": variance_mse,
            "MSE_pred": pred_mse,
            "cosine_similarity": cos_sim,
        }

    def compute_loss(self, y, x, mc_steps):
        y_model, bias, precision, weights = self._sample(x=x, mc_steps=mc_steps)
        return self._compute_loss(y=y, y_model=y_model, bias=bias, precision=precision, weights=weights)

    def _sample(self, x, mc_steps, denoise=False):
        nsteps = max(1, mc_steps)

        bias = self.bias_net(x)
        precision = self.precision_net(x)
        weights = self.weights_net(x)

        y_model = bias.clone()
        for _ in range(nsteps):
            h_model = self._sample_hid(y_model, bias, precision, weights)
            y_model = self._sample_vis(h_model, bias, precision, weights)

        if denoise:
            y_model = self._sample_vis(h_model, bias, precision, weights, denoise=True)

        return y_model, bias, precision, weights

    @torch.no_grad()
    def sample(self, x, mc_steps, denoise=False):
        y_model, *_ = self._sample(x=x, mc_steps=mc_steps, denoise=denoise)
        return y_model


if __name__ == "__main__":
    model = CRBM(32, 10, 16)
    summary(model)
