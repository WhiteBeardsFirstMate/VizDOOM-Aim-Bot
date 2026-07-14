import torch
import torch.nn as nn

FLAT_SIZE = 16 * 6 * 10
HIDDEN_SIZE = 128

class A2CNet(nn.Module):
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
            nn.Linear(FLAT_SIZE, HIDDEN_SIZE),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(HIDDEN_SIZE, n_actions)
        self.value_head = nn.Linear(HIDDEN_SIZE, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, FLAT_SIZE)
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x).reshape(-1, 1)
        return logits, value
