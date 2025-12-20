import torch.nn as nn
import torch
from training.network.utils import *

class TransformerDecoder(nn.Module):
    def __init__(self, 
                 device,
                 feature_dim,
                 dim_feedforward = 256,
                 n_head = 4,  
                #  max_seq_len = 30*10,
                 max_seq_len = 3000,
                 period = 30,
                 batch_size = 256
                 ):

        super(TransformerDecoder, self).__init__()
        self.device = device
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=n_head, dim_feedforward=dim_feedforward,
                                                   batch_first=True)

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        self.biased_mask = init_biased_mask(n_head=4, max_seq_len=max_seq_len, period=period)


    ''' 
    hidden states shape (batch_size, frame_length, feature_dim)
    memory shape (batch_size, frame_length, feature_dim)
    output_shape (batch_size, frame_length, featrue_dim)
    '''
        
    def forward(self, hidden_states, memory):

        batch_size = hidden_states.size(0)
        seq_len = hidden_states.size(1)

        tgt_mask = self.biased_mask[:, :seq_len, :seq_len].clone().detach().to(device=self.device)
        tgt_mask = tgt_mask.unsqueeze(0).repeat(batch_size, 1, 1, 1).view(-1, seq_len, seq_len)


        
        memory_mask = enc_dec_mask(self.device, hidden_states.shape[1], hidden_states.shape[1])
        
        decoder_output = self.transformer_decoder(hidden_states, memory, tgt_mask=tgt_mask,
                                            memory_mask=memory_mask)
        
        return decoder_output