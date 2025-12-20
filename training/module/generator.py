import torch
import torch.nn as nn
import torch.nn.functional as F


class MotionGenerator(nn.Module):
    """
    Expert-specific MLPs blended by gating coefficients.
    Shapes mirror the legacy FLAME→ARKit mapper checkpoints.
    """

    def __init__(self, device, input_dim, output_dim, rng=None):
        super().__init__()
        self.device = device
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Expert-specific weights: [E, in, hidden] etc.
        self.w0 = nn.Parameter(torch.empty(8, input_dim, 128))
        self.w1 = nn.Parameter(torch.empty(8, 128, 128))
        self.w2 = nn.Parameter(torch.empty(8, 128, output_dim))
        self.b0 = nn.Parameter(torch.empty(8, 128))
        self.b1 = nn.Parameter(torch.empty(8, 128))
        self.b2 = nn.Parameter(torch.empty(8, output_dim))

        self.bn0 = nn.BatchNorm1d(128)
        self.bn1 = nn.BatchNorm1d(128)

    def forward(self, generator_input, blend_coeff, batch_size):
        # generator_input: [B, input_dim]; blend_coeff: [B, E]
        # Expand inputs for each expert
        # hidden0: [E, B, hidden]
        hidden0 = torch.einsum("bi,eih->ebh", generator_input, self.w0) + self.b0.unsqueeze(1)
        hidden0 = F.relu(self.bn0(hidden0.reshape(-1, hidden0.shape[-1])).reshape_as(hidden0))

        hidden1 = torch.einsum("ebh,ehj->ebj", hidden0, self.w1) + self.b1.unsqueeze(1)
        hidden1 = F.relu(self.bn1(hidden1.reshape(-1, hidden1.shape[-1])).reshape_as(hidden1))

        out = torch.einsum("ebj,ejk->ebk", hidden1, self.w2) + self.b2.unsqueeze(1)
        # Weight by blend coefficients and sum experts
        out = torch.einsum("be,ebk->bk", blend_coeff, out)
        return out
