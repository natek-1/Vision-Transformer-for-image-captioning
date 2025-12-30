import torch
import torch.nn as nn

class AttentionBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, masking=True, dropout=0.1):
        '''
        Multi-Head Attention Block with optional causal masking.
        Args:
            hidden_size (int): Dimensionality of input and output features. 
            num_heads (int): Number of attention heads.
            masking (bool): If True, applies causal masking (for decoder).
            dropout (float): Dropout rate for attention weights.
        '''
        super(AttentionBlock, self).__init__()
        self.masking = masking
        self.multihead_attn = nn.MultiheadAttention(
            hidden_size, 
            num_heads=num_heads, 
            batch_first=True, 
            dropout=dropout # Fixed parameter name
        )

    def forward(self, x_in, kv_in, key_mask=None):
        mask = None
        if self.masking:
            # Query length x Key length
            q_len = x_in.size(1)
            k_len = kv_in.size(1)
            mask = torch.triu(torch.ones(q_len, k_len, device=x_in.device), 1).bool()
            
        # Returns (output, weights) - we take [0]
        return self.multihead_attn(x_in, kv_in, kv_in, attn_mask=mask, 
                                   key_padding_mask=key_mask)[0]
        
class TransformerBlock(nn.Module):
    def __init__(self, hidden_size=128, num_heads=4, decoder=False, masking=True, mlp_dropout=0.1, att_dropout=0.1):
        '''
        Transformer Block with self-attention, optional cross-attention (for decoder),
        and feed-forward network.
        Args:
            hidden_size (int): Dimensionality of input and output features.
            num_heads (int): Number of attention heads.
            decoder (bool): If True, includes cross-attention layer.
            masking (bool): If True, applies causal masking in self-attention.
            mlp_dropout (float): Dropout rate for the feed-forward network.
            att_dropout (float): Dropout rate for attention layers.
        '''
        super(TransformerBlock, self).__init__()
        self.decoder = decoder

        # Pre-Norm approach
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn1 = AttentionBlock(hidden_size, num_heads, masking, att_dropout)
        
        if self.decoder:
            self.norm2 = nn.LayerNorm(hidden_size)
            self.attn2 = AttentionBlock(hidden_size, num_heads, False, att_dropout)
        
        self.norm_mlp = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(mlp_dropout)
        )
                
    def forward(self, x, input_key_mask=None, cross_key_mask=None, kv_cross=None):
        # Self-Attention
        res = x
        x = self.norm1(x)
        x = self.attn1(x, x, key_mask=input_key_mask)
        x = x + res

        # Cross-Attention with pre-norm
        if self.decoder and kv_cross is not None:
            res = x
            x = self.norm2(x)
            x = self.attn2(x, kv_cross, key_mask=cross_key_mask)
            x = x + res

        # Feed Forward with pre-norm
        res = x
        x = self.norm_mlp(x)
        x = self.mlp(x)
        x = x + res
        
        return x