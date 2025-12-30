import math


import torch
import torch.nn as nn

from vision.model.attention import TransformerBlock


# sinusoidal positional embeds
class SinusoidalPosEmb(nn.Module):
    '''
    Sinusoidal Positional Embedding Module.
    Args:
        dim (int): Dimensionality of the positional embeddings.
    Returns:
        torch.Tensor: Positional embeddings of shape (batch_size, dim).
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class Decoder(nn.Module):
    def __init__(self, num_emb, hidden_size=128, num_layers=3, num_heads=4, 
                 mlp_dropout=0.1, att_dropout=0.1):
        '''
        Transformer Decoder that generates sequences by attending to encoder outputs.
        Args:
            num_emb (int): Size of the vocabulary for token embeddings.
            hidden_size (int): Dimensionality of input and output features.
            num_layers (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            mlp_dropout (float): Dropout rate for the feed-forward network.
            att_dropout (float): Dropout rate for attention layers.
        Returns:
            torch.Tensor: Output logits of shape (batch_size, seq_length, num_emb).
        '''
        super(Decoder, self).__init__()
        
        # Create and Initialize an embedding layer for tokens
        self.embedding = nn.Embedding(num_emb, hidden_size)
        self.embedding.weight.data = 0.001 * self.embedding.weight.data

        # Initialize sinusoidal positional embeddings
        self.pos_emb = SinusoidalPosEmb(hidden_size)
        
        # Create multiple transformer blocks as layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_dropout=mlp_dropout, att_dropout=att_dropout,
                             decoder=True) for _ in range(num_layers)
        ])
                
        # Define a linear layer for output prediction
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.fc_out = nn.Linear(hidden_size, num_emb)
        
    def forward(self, input_seq, encoder_output, input_padding_mask=None, 
                encoder_padding_mask=None):        
        # Embed the input sequence
        input_embs = self.embedding(input_seq)
        bs, l, h = input_embs.shape

        # Add positional embeddings to the input embeddings
        seq_indx = torch.arange(l, device=input_seq.device)
        pos_emb = self.pos_emb(seq_indx).reshape(1, l, h).expand(bs, l, h)
        embs = input_embs + pos_emb
        
        # Pass the embeddings through each transformer block
        for block in self.blocks:
            embs = block(embs, 
                           input_key_mask=input_padding_mask, 
                           cross_key_mask=encoder_padding_mask, 
                           kv_cross=encoder_output)
        
        return self.fc_out(self.layer_norm(embs))