import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F

import gc
from training.network.configs import *
import torch.optim as optim
from training.network.utils import *


class StyleEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, type="attn", use_vae=True, use_normalize=True,
                 vq_latent_dim=12, vq_num_categories=16):  # 추가된 인자
        super(StyleEncoder, self).__init__()
        self.device = DEVICE
        self.use_vae = use_vae
        self.use_normalize = use_normalize

        self.vq_latent_dim = vq_latent_dim
        self.vq_num_categories = vq_num_categories
        self.style_embedding_size = output_size  # 그냥 latent vector 크기로 간주

        # VQ-VAE일 경우 출력 차원 조정
        encoder_output_size = (vq_latent_dim * vq_num_categories) if use_vae else output_size

        if type == "gru":
            self.encoder = StyleEncoderGRU(input_size, hidden_size, encoder_output_size)
        elif type == "attn":
            self.encoder = StyleEncoderAttn(input_size, hidden_size, encoder_output_size)

    def forward(self, input = None, temperature: float =1e-6, hard: bool = False, logits = None):
          # [B, D * K] if VQ-VAE
        if self.use_vae:
            if logits == None:
                encoder_output = self.encoder(input)
            # encoder output을 categorical distribution으로 reshape
                logits = encoder_output.view(-1, self.vq_latent_dim, self.vq_num_categories)  # [B, D, K]
            # Gumbel-softmax trick (VQ-VAE 스타일)
            style_embedding = F.gumbel_softmax(logits, tau=temperature, hard=hard, dim=-1)  # [B, D, K]
            style_embedding = style_embedding.view(style_embedding.size(0), -1)  # [B, D*K]


            return style_embedding, logits, None  # VQ-VAE에서는 logvar 대신 logits 리턴
        else:
            encoder_output = self.encoder(input)
            # 기존 방식 유지
            if self.use_normalize:
                encoder_output = F.normalize(encoder_output, p=2, dim=1)
            return encoder_output, None, None
        
    def kl_gumbel_softmax_uniform(self, logits):
        """
        logits: [B, D, K]  (D = self.vq_latent_dim, K = self.vq_num_categories)
        Computes KL(q || Uniform), where q = softmax(logits)
        """
        q = F.softmax(logits, dim=-1)  # [B, D, K]
        log_q = F.log_softmax(logits, dim=-1)  # [B, D, K]
        
        # log(1/K)
        log_uniform = math.log(1.0 / self.vq_num_categories)

        # KL = sum_i q_i (log q_i - log (1/K)) = sum_i q_i log q_i + log K
        kl_per_sample = (q * log_q).sum(dim=-1) + log_uniform * -1  # [B, D]
        kl_mean = kl_per_sample.mean()  # 평균 내기

        return kl_mean




class StyleEncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size, style_embedding_size):
        super(StyleEncoderGRU, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm1D(
                input_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            # AvgPoolNorm1D(kernel_size=2),
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
        )
        self.rnn_layer = nn.GRU(hidden_size, hidden_size, 1, batch_first=True, bidirectional=True)
        self.projection_layer = LinearNorm(
            hidden_size * 2, style_embedding_size, w_init_gain="linear"
        )

    def forward(self, input):
        input = self.convs(input)
        output, _ = self.rnn_layer(input)
        style_embedding = self.projection_layer(output[:, -1])
        return style_embedding


class StyleEncoderAttn(nn.Module):
    """ Style Encoder Module:
        - Positional Encoding
        - Nf x FFT Blocks
        - Linear Projection Layer
    """

    def __init__(self, input_size, hidden_size, style_embedding_size):
        super(StyleEncoderAttn, self).__init__()

        # positional encoding
        self.pos_enc = PositionalEncoding(style_embedding_size)

        self.convs = nn.Sequential(
            ConvNorm1D(
                input_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            nn.LayerNorm(hidden_size),
            nn.Dropout(0.2),
            ConvNorm1D(
                hidden_size,
                style_embedding_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            nn.LayerNorm(style_embedding_size),
            nn.Dropout(0.2),
        )
        # FFT blocks
        blocks = []
        for _ in range(1):
            blocks.append(FFTBlock(style_embedding_size))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, input):
        """ Forward function of Prosody Encoder:
            frames_energy = (B, T_max)
            frames_pitch = (B, T_max)
            mel_specs = (B, nb_mels, T_max)
            speaker_ids = (B, )
            output_lengths = (B, )
        """
        output_lengths = torch.as_tensor(
            len(input) * [input.shape[1]], device=input.device, dtype=torch.int32
        )
        # compute positional encoding
        pos = self.pos_enc(output_lengths.unsqueeze(1)).to(input.device)  # (B, T_max, hidden_embed_dim)
        # pass through convs
        outputs = self.convs(input)  # (B, T_max, hidden_embed_dim)

        # create mask
        mask = ~get_mask_from_lengths(output_lengths)  # (B, T_max)
        # add encodings and mask tensor
        outputs = outputs + pos  # (B, T_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, T_max, hidden_embed_dim)
        # pass through FFT blocks
        for _, block in enumerate(self.blocks):
            outputs = block(outputs, None, mask)  # (B, T_max, hidden_embed_dim)
        # average pooling on the whole time sequence
        style_embedding = torch.sum(outputs, dim=1) / output_lengths.unsqueeze(
            1
        )  # (B, hidden_embed_dim)

        return style_embedding


# ===============================================
#                   Sub-modules
# ===============================================
class LinearNorm(nn.Module):
    """ Linear Norm Module:
        - Linear Layer
    """

    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = nn.Linear(in_dim, out_dim, bias=bias)
        nn.init.xavier_uniform_(self.linear_layer.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Linear Norm
            x = (*, in_dim)
        """
        x = self.linear_layer(x)  # (*, out_dim)

        return x


class PositionalEncoding(nn.Module):
    """ Positional Encoding Module:
        - Sinusoidal Positional Embedding
    """

    def __init__(self, embed_dim, max_len=20000, timestep=10000.0):
        super(PositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2).float() * (-np.log(timestep) / self.embed_dim)
        )  # (embed_dim // 2, )
        self.pos_enc = torch.FloatTensor(max_len, self.embed_dim).zero_()  # (max_len, embed_dim)
        self.pos_enc[:, 0::2] = torch.sin(pos * div_term)
        self.pos_enc[:, 1::2] = torch.cos(pos * div_term)

    def forward(self, x):
        """ Forward function of Positional Encoding:
            x = (B, N) -- Long or Int tensor
        """
        # initialize tensor
        nb_frames_max = torch.max(torch.cumsum(x, dim=1))
        pos_emb = torch.FloatTensor(
            x.size(0), nb_frames_max, self.embed_dim
        ).zero_()  # (B, nb_frames_max, embed_dim)
        # pos_emb = pos_emb.cuda(x.device, non_blocking=True).float()  # (B, nb_frames_max, embed_dim)

        # TODO: Check if we can remove the for loops
        for line_idx in range(x.size(0)):
            pos_idx = []
            for column_idx in range(x.size(1)):
                idx = x[line_idx, column_idx]
                pos_idx.extend([i for i in range(idx)])
            emb = self.pos_enc[pos_idx]  # (nb_frames, embed_dim)
            pos_emb[line_idx, : emb.size(0), :] = emb

        return pos_emb


class FFTBlock(nn.Module):
    """ FFT Block Module:
        - Multi-Head Attention
        - Position Wise Convolutional Feed-Forward
        - FiLM conditioning (if film_params is not None)
    """

    def __init__(self, hidden_size):
        super(FFTBlock, self).__init__()
        self.attention = MultiHeadAttention(hidden_size)
        self.feed_forward = PositionWiseConvFF(hidden_size)

    def forward(self, x, film_params, mask):
        """ Forward function of FFT Block:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
            mask = (B, L_max)
        """
        # attend
        attn_outputs, _ = self.attention(
            x, x, x, key_padding_mask=mask
        )  # (B, L_max, hidden_embed_dim)
        attn_outputs = attn_outputs.masked_fill(
            mask.unsqueeze(2), 0
        )  # (B, L_max, hidden_embed_dim)
        # feed-forward pass
        outputs = self.feed_forward(attn_outputs, film_params)  # (B, L_max, hidden_embed_dim)
        outputs = outputs.masked_fill(mask.unsqueeze(2), 0)  # (B, L_max, hidden_embed_dim)

        return outputs


class MultiHeadAttention(nn.Module):
    """ Multi-Head Attention Module:
        - Multi-Head Attention
            A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, L. Kaiser and I. Polosukhin
            "Attention is all you need",
            in NeurIPS, 2017.
        - Dropout
        - Residual Connection 
        - Layer Normalization
    """

    def __init__(self, hidden_size):
        super(MultiHeadAttention, self).__init__()
        self.multi_head_attention = nn.MultiheadAttention(hidden_size, 4, 0.1)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        """ Forward function of Multi-Head Attention:
            query = (B, L_max, hidden_embed_dim)
            key = (B, T_max, hidden_embed_dim)
            value = (B, T_max, hidden_embed_dim)
            key_padding_mask = (B, T_max) if not None
            attn_mask = (L_max, T_max) if not None
        """
        # compute multi-head attention
        # attn_outputs = (L_max, B, hidden_embed_dim)
        # attn_weights = (B, L_max, T_max)
        attn_outputs, attn_weights = self.multi_head_attention(
            query.transpose(0, 1),
            key.transpose(0, 1),
            value.transpose(0, 1),
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
        )
        attn_outputs = attn_outputs.transpose(0, 1)  # (B, L_max, hidden_embed_dim)
        # apply dropout
        attn_outputs = self.dropout(attn_outputs)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        attn_outputs = self.layer_norm(attn_outputs + query)  # (B, L_max, hidden_embed_dim)

        return attn_outputs, attn_weights


class PositionWiseConvFF(nn.Module):
    """ Position Wise Convolutional Feed-Forward Module:
        - 2x Conv 1D with ReLU
        - Dropout
        - Residual Connection 
        - Layer Normalization
        - FiLM conditioning (if film_params is not None)
    """

    def __init__(self, hidden_size):
        super(PositionWiseConvFF, self).__init__()
        self.convs = nn.Sequential(
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="relu",
            ),
            nn.ReLU(),
            ConvNorm1D(
                hidden_size,
                hidden_size,
                kernel_size=3,
                stride=1,
                padding=int((3 - 1) / 2),
                dilation=1,
                w_init_gain="linear",
            ),
            nn.Dropout(0.1),
        )
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, x, film_params):
        """ Forward function of PositionWiseConvFF:
            x = (B, L_max, hidden_embed_dim)
            film_params = (B, nb_film_params)
        """
        # pass through convs
        outputs = self.convs(x)  # (B, L_max, hidden_embed_dim)
        # add residual connection and perform layer normalization
        outputs = self.layer_norm(outputs + x)  # (B, L_max, hidden_embed_dim)
        # add FiLM transformation
        if film_params is not None:
            nb_gammas = int(film_params.size(1) / 2)
            assert nb_gammas == outputs.size(2)
            gammas = film_params[:, :nb_gammas].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            betas = film_params[:, nb_gammas:].unsqueeze(1)  # (B, 1, hidden_embed_dim)
            outputs = gammas * outputs + betas  # (B, L_max, hidden_embed_dim)

        return outputs


class ConvNorm1D(nn.Module):
    """ Conv Norm 1D Module:
        - Conv 1D
    """

    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=1,
            stride=1,
            padding=None,
            dilation=1,
            bias=True,
            w_init_gain="linear",
    ):
        super(ConvNorm1D, self).__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        nn.init.xavier_uniform_(self.conv.weight, gain=nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        """ Forward function of Conv Norm 1D
            x = (B, L, in_channels)
        """
        if x.dim() != 3:
            raise RuntimeError(f"ConvNorm1D expected 3D input (B, L, C), got shape {tuple(x.shape)}")
        if x.shape[-1] != self.conv.in_channels:
            raise RuntimeError(
                f"ConvNorm1D channel mismatch: got C={x.shape[-1]}, expected {self.conv.in_channels}; "
                f"x shape={tuple(x.shape)}"
            )
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.conv(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x


class AvgPoolNorm1D(nn.Module):
    def __init__(
            self, kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True
    ):
        super(AvgPoolNorm1D, self).__init__()
        self.avgpool1d = nn.AvgPool1d(kernel_size, stride, padding, ceil_mode, count_include_pad)

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, in_channels, L)
        x = self.avgpool1d(x)  # (B, out_channels, L)
        x = x.transpose(1, 2)  # (B, L, out_channels)

        return x


def get_mask_from_lengths(lengths):
    """ Create a masked tensor from given lengths

    :param lengths:     torch.tensor of size (B, ) -- lengths of each example

    :return mask: torch.tensor of size (B, max_length) -- the masked tensor
    """
    max_len = torch.max(lengths)
    # ids = torch.arange(0, max_len).cuda(lengths.device, non_blocking=True).long()
    ids = torch.arange(0, max_len).long().to(lengths.device)
    mask = (ids < lengths.unsqueeze(1)).bool().to(lengths.device)
    return mask
