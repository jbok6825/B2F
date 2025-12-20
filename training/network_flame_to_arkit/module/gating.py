import torch
import torch.nn as nn
import torch.nn.functional as F


class Gating(nn.Module):
    """
    Mixture-of-experts gating network.
    """

    def __init__(self, device, input_dim, num_experts, rng=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.w0 = nn.Parameter(torch.empty(input_dim, 128))
        self.w1 = nn.Parameter(torch.empty(128, 128))
        self.w2 = nn.Parameter(torch.empty(128, num_experts))
        self.b0 = nn.Parameter(torch.empty(1, 128))
        self.b1 = nn.Parameter(torch.empty(1, 128))
        self.b2 = nn.Parameter(torch.empty(1, num_experts))
        self.bn0 = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, x):
        h0 = torch.matmul(x, self.w0) + self.b0
        h0 = F.relu(self.bn0(h0))
        h1 = torch.matmul(h0, self.w1) + self.b1
        h1 = F.relu(self.bn1(h1))
        logits = torch.matmul(h1, self.w2) + self.b2
        return F.softmax(logits, dim=-1)
