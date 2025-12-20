import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import math

'''
latent_dim은 한 frame의 feature
output_dim은 한 frame의 feature
'''
class ContentEncoder(nn.Module):
    def __init__(self, latent_dim, output_dim, positional_encoding = True, use_vqvae=False):
        super(ContentEncoder, self).__init__()

        self.positional_encoding = positional_encoding
        if self.positional_encoding == True:
            
            self.pos_encoder = PositionalEncoding(latent_dim)
        seqTransEncoderLayer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=4,
            dim_feedforward=256,
            dropout=0.2,
            activation="gelu"
        )
        self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer, num_layers=8)
        


        self.output_projection = nn.Linear(latent_dim, output_dim)

    def forward(self, input,):
        input = input.transpose(0, 1)
        if self.positional_encoding == True:
            input = self.pos_encoder(input)
        output = self.seqEncoder(input)
        output = output.transpose(0, 1)


        # 기본 구조
        output = self.output_projection(output)
        output = F.normalize(output, p=2, dim=-1)
        return output

    def quantize(self, projected):
        flattened = projected.view(-1, self.code_dim)

        # 거리 계산 시 스케일 조정
        distances = torch.cdist(flattened, self.embedding.weight * 5, p=2)
        indices = distances.argmin(dim=1)
        quantized = self.embedding(indices).view(projected.shape)

        # Straight-through gradient for quantization
        quantized = projected + (quantized - projected).detach()

        return quantized


    def compute_vq_loss(self, quantized, projected, commitment_weight=1.0):
        # Commitment Loss (L2 distance)
        commitment_loss = commitment_weight * F.mse_loss(quantized.detach(), projected)

        # Straight-through gradient for quantization
        codebook_loss = 0.5 * F.mse_loss(quantized, projected.detach())


        # Total VQ Loss
        return commitment_loss + codebook_loss


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=20000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]