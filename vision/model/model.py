import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTModel



class PatchEmbedding(nn.Module):
    '''
    Turns a 2D input image into a 1D sequence learnable embedding vector.

    Args:
        in_channels (int): Number of color channels for the input images. Defaults to 3.
        patch_size (int): Size of patches to convert input image into. Defaults to 16.
        embedding_dim (int): Size of embedding to turn image into. Defaults to 768.
    '''
    def __init__(self, in_channels = 3, patch_size=16, embedding_dim=768):
        super().__init__()
        self.patcher = nn.Conv2d(in_channels=in_channels, out_channels=embedding_dim,
                                 kernel_size=patch_size, stride=patch_size, padding=0)
    
    def forward(self, x):
        # input of shape (batch_size, color_channel, height, width)
        x = self.patcher(x) # (batch_size, embedding_dim, height//patch_size, width//patch_size)
        x = x.flatten(2) # (batch_size, embedding_dim, (height * width)//(patch_size)**2)
        return x.permute(0, 2, 1) #(batch_size,(height * width)//(patch_size)**2), embedding_dim)
    
class EncoderBlock(nn.Module):
    '''
    Encoder block that returns a representation of the image patches.
    
    Args:
        embedding_dim (int): size of the embedding for each image patch. Defaults 768.
        num_heads (int): Number of head in the attention layer. Defaults 12.
        mlp_size (int): Size for the feed forward portion of the encoder. Defaults 3072.
        dropout (float): Amount of dropout in attention and mlp layer. Default 0.1
    '''
    def __init__(self, embeding_dim = 768, num_heads = 12, mlp_size= 3072, dropout=0.1):
        super().__init__()
        
        self.layer_norm1 = nn.LayerNorm(normalized_shape=embeding_dim)
        self.attention = nn.MultiheadAttention(embed_dim=embeding_dim, num_heads=num_heads, dropout=dropout,
                                               batch_first=True)
        
        self.layer_norm2 = nn.LayerNorm(normalized_shape=embeding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embeding_dim, mlp_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_size, embeding_dim),
            nn.Dropout(dropout)   
        )
    
    def forward(self, x):
        norm_x = self.layer_norm1(x)
        x = x + self.attention(norm_x, norm_x, norm_x, need_weights=False)[0]
        
        norm_x = self.layer_norm2(x)
        return x + self.mlp(norm_x)

class ViT(nn.Module):
    '''
    Creates a Vision Transformer architecture with ViT-Base hyperparameters by default with no classification token and layer.
    '''
    def __init__(self, img_size= 224, in_channels=3, patch_size=16, num_blocks = 12,
                 embed_dim = 768, mlp_size = 3072, num_heads = 12, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, f"Image size must be divisible by patch size, image size: {img_size}, patch size: {patch_size}."
        self.num_patches = (img_size // patch_size) ** 2
        
        self.class_embedding = nn.Parameter(data=torch.randn(1, 1, embed_dim)) ##
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches+1, embed_dim)) ##
        self.embed_dropout = nn.Dropout(dropout)
        
        self.patch_embedding = PatchEmbedding(in_channels=in_channels, patch_size=patch_size, embedding_dim=embed_dim)
        
        self.transformer_encoder = nn.Sequential(*[EncoderBlock(embeding_dim=embed_dim, num_heads=num_heads,
                                                                mlp_size=mlp_size, dropout=dropout) for _ in range(num_blocks)])
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # x is an image of shape (batch_size, color_chanel, height, width)
        batch_size = x.shape[0] ##
        class_token = self.class_embedding.expand(batch_size, -1, -1) ## (batch_size, 1, n_embed_dim)
        x = self.patch_embedding(x) # -> (batch_size, embed_dim, num_patches)
        x = torch.cat((class_token, x), dim=1) ##
        x = x + self.position_embedding
        x = self.embed_dropout(x)
        x = self.transformer_encoder(x) # -> (batch_size, embed_dim, num_patches)
        return self.classifier(x[:,0,:]) # take only the classification token
    

class CasualSelfAttention(nn.Module):
    
    def __init__(self, n_embed=768, n_head=12, block_size=45):
        super().__init__()
        assert n_embed % n_head == 0
        # we will use this to get the key, value, pair for all heads only using one batch
        self.c_attn = nn.Linear(n_embed, 3*n_embed)
        
        # output
        self.c_proj = nn.Linear(n_embed, n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = n_head
        self.n_embed = n_embed
        
        # not really a bias, used for masking
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size).type(torch.int)
                             , persistent=False)
        
    
    def forward(self, x, att_mask=None):
        # x of shape (batch_size, seq_len, embed_dim)
        # att_mask of shape (batch_size, seq_len)
        batch_size, seq_len, embed_dim = x.size()
        
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embed, dim=2) # split on last dimention
        
        q = q.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2) #(b, nh, seq_len, hs)
        k = k.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2) #(b, nh, seq_len, hs)
        v = v.view(batch_size, seq_len, self.n_head, embed_dim // self.n_head).transpose(1, 2) #(b, nh, seq_len, hs)
        
        att = (q @ k.transpose(-2, -1)) * (1/ math.sqrt(k.size(-1)))  #(batch_size, h, seq_len, seq_len)
        
        mask = self.bias[:,:, :seq_len,:seq_len]
        if att_mask is not None:
            mask = self.bias[:,:, :seq_len,:seq_len] & att_mask.unsqueeze(1).unsqueeze(1).int() # batch_size, 1, seq_len, seq_len
            
        att = att.masked_fill(mask == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v # (b, nh, seq_len, seq_len) x (b, nh, seq_len, hs) -> (b, nh, seq_len, hs)
        y = y.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        return self.c_proj(y)

class MLP(nn.Module):
    
    def __init__(self, n_embed):
        super().__init__()
        
        self.c_fc = nn.Linear(n_embed, 4*n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * n_embed, n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    
    def __init__(self, n_embed=768, n_head=12, block_size=45):
        super().__init__()

        self.ln_1 = nn.LayerNorm(n_embed)
        self.attn = CasualSelfAttention(n_embed=n_embed, n_head=n_head, block_size=block_size)
        self.ln_2 = nn.LayerNorm(n_embed)
        self.mlp = MLP(n_embed)
    
    def forward(self, x, att_mask = None):
        x = x + self.attn(self.ln_1(x), att_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    
    def __init__(self, vocab_size = 50257, n_embed = 768, block_size=45, n_layer=12, n_head=12):
        super().__init__()
        self.n_layer = n_layer
        self.block_size = block_size
        
        
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(vocab_size, n_embed), # typical(token) embedding 
                wpe = nn.Embedding(block_size, n_embed), # positional emebeding
                h = nn.ModuleList(Block(n_embed, n_head, block_size) for _ in range(n_layer)),
                ln_f = nn.LayerNorm(n_embed)
            )
        )
        self.lm_head = nn.Linear(n_embed, vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.n_layer) ** -0.5 # std deviation initialization to get the to finish with std of 1. We multiply by two because each layer has 2 residual blocks
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            
        
    
    def forward(self, idx: torch.Tensor, encoder_output: torch.Tensor, attn_mask: torch.Tensor = None):
        # we get idx of shape batch_size, seq_len
        # encoder output of shape batch_size, n_embed
        # attn_mask is of shape batch_size, seq_len
        batch_size, seq_len = idx.shape
        assert seq_len <= self.block_size, f"Cannot process a sequence length of {seq_len}, the context length is of size {self.block_size}, which is smaller"
        seq_len += 1
        
        token_embed = self.transformer.wte(idx) # shape( batch_size, seq_len, n_embed)
        token_embed = torch.cat([encoder_output.unsqueeze(1), token_embed], dim=1)
        
        pos = torch.arange(0, seq_len, dtype=torch.long, device=idx.device)
        pos_embed = self.transformer.wpe(pos) # shape (seq_len, n_emebd)
        x = pos_embed + token_embed # broadcasting occurs resulting in output of shape batch_size, seq_len, n_embed
        
        if attn_mask is not None:
            cls_attention = torch.ones((batch_size, 1), device=attn_mask.device)
            attn_mask = torch.cat([cls_attention, attn_mask], dim=1) # now attn_mask matches the new seq_len.
        
        # go forward in the decoder layer
        for block in self.transformer.h:
            x = block(x, attn_mask)
            
        # last layer normalization
        x = self.transformer.ln_f(x)
        
        logits = self.lm_head(x) # result of shape batch_size, seq_len, vocab_size
        return logits

class Captioner(nn.Module):
    
    def __init__(self,vocab_size, max_seq_len, padding_idx, img_size= 224, in_channels=3, patch_size=16, num_blocks = 12,
                 embed_dim = 768, mlp_size = 3072, num_heads = 12, dropout=0.1):
        super().__init__()
        self.vit = ViT(
            img_size=img_size, in_channels=in_channels, patch_size=patch_size, embed_dim=embed_dim,
            num_blocks=num_blocks, mlp_size=mlp_size, num_heads=num_heads, dropout=dropout)
        vit_name = "google/vit-base-patch16-224-in21k"
        self.vit = ViTModel.from_pretrained(vit_name)
        for param in self.vit.parameters():
            param.requires_grad = False
        
        self.gpt2 = GPT(
            vocab_size=vocab_size, max_seq_len=max_seq_len, padding_idx=padding_idx, embed_dim=embed_dim,
            mlp_size=mlp_size, num_layers=num_blocks, num_heads=num_heads, dropout=dropout)
        
    def forward(self, image, caption, attn_mask):
        encoder_out = self.vit(image)
        return self.gpt2(caption, encoder_out, attn_mask)


