import numpy as np
import torch
import torch.nn as nn

FLAT_SIZE = 16 * 6 * 10
HIDDEN_SIZE = 128

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma0=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.mu_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.sigma_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.mu_bias = nn.Parameter(torch.empty(out_features))
        self.sigma_bias = nn.Parameter(torch.empty(out_features))

        self.register_buffer("eps_in", torch.zeros(in_features))
        self.register_buffer("eps_out", torch.zeros(out_features))

        self.reset_parameters(sigma0)

    def reset_parameters(self, sigma0):
        bound = 1 / np.sqrt(self.in_features)
        self.mu_weight.data.uniform_(-bound, bound)
        self.mu_bias.data.uniform_(-bound, bound)
        self.sigma_weight.data.fill_(sigma0 / np.sqrt(self.in_features))
        self.sigma_bias.data.fill_(sigma0 / np.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size, device=self.mu_weight.device)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        self.eps_in = self._scale_noise(self.in_features)
        self.eps_out = self._scale_noise(self.out_features)

    def forward(self, x):
        if self.training:
            self.reset_noise()
            weight = (
                self.mu_weight
                + self.sigma_weight * torch.ger(self.eps_out, self.eps_in)
            )
            bias = self.mu_bias + self.sigma_bias * self.eps_out
        else:
            weight = self.mu_weight
            bias = self.mu_bias
        return nn.functional.linear(x, weight, bias)

class NoisyDuelQNet(nn.Module):
    def __init__(self, n_actions: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(8),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            NoisyLinear(FLAT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.value_stream = NoisyLinear(HIDDEN_SIZE, 1)
        self.adv_stream = NoisyLinear(HIDDEN_SIZE, n_actions)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, FLAT_SIZE)
        x = self.fc(x)
        value = self.value_stream(x)
        adv = self.adv_stream(x)
        q = value + (adv - adv.mean(dim=1, keepdim=True))
        return q
