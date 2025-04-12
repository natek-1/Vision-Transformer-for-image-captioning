import math
from dataclasses import dataclass\

import torch
from torch import nn
from transformers import ViTFeatureExtractor, ViTModel
import torch.nn.functional as F



@dataclass
class GPTConfig:
    
    block_size: int = 1024 # max sequence length
    vocab_size: int = 50257 # size of vocabulary
    n_layer: int = 12 # number of layers
    n_head: int = 12 # number of attention heads
    n_embed: int = 768 # embedding dimension

class CasualSelfAttention(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embed % config.n_head == 0
        self.config = config
        # we will use this to get the key, value, pair for all heads only using one batch
        self.c_attn = nn.Linear(config.n_embed, 3*config.n_embed)
        
        # output
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        #regularization
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        
        # not really a bias, used for masking
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size).type(torch.int)
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
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        
        self.config = config
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1
    
    def forward(self, x):
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm(config.n_embed)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed)
        self.mlp = MLP(config)
    
    def forward(self, x, att_mask = None):
        x = x + self.attn(self.ln_1(x), att_mask)
        x = x + self.mlp(self.ln_2(x))
        return x
    
class GPT(nn.Module):
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        
        self.transformer = nn.ModuleDict(
            dict(
                wte = nn.Embedding(config.vocab_size, config.n_embed), # typical(token) embedding 
                wpe = nn.Embedding(config.block_size, config.n_embed), # positional emebeding
                h = nn.ModuleList(Block(config) for _ in range(config.n_layer)),
                ln_f = nn.LayerNorm(config.n_embed)
            )
        )
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)
        
        # weight sharing scheme
        self.transformer.wte.weight = self.lm_head.weight
        
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5 # std deviation initialization to get the to finish with std of 1. We multiply by two because each layer has 2 residual blocks
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
        assert seq_len <= self.config.block_size, f"Cannot process a sequence length of {seq_len}, the context length is of size {self.config.block_size}, which is smaller"
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
    
    @classmethod
    def from_pretrained(cls, model_type):
        """Load pretrained GPT-2 model weight from hugging face"""
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
        
        from transformers import GPT2LMHeadModel
        print("Loading weights from pretrained gpt: ", model_type)
        
        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embed=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embed=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embed=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embed=1600), # 1558M params
        }[model_type]
        
        config_args['vocab_size'] = 50257
        config_args['block_size'] = 1024
        
        
        config = GPTConfig(**config_args)
        model = GPT(config)
        state_dict = model.state_dict()
        state_dict_keys = state_dict.keys()
        
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        state_dict_hf = model_hf.state_dict()
        state_dict_keys_hf = state_dict_hf.keys()

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        
        assert len(state_dict_keys_hf) == len(state_dict_keys), f"mismatched keys: {len(state_dict_keys_hf)} != {len(state_dict_keys)}"
        for k in state_dict_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert state_dict_hf[k].shape[::-1] == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert state_dict_hf[k].shape == state_dict[k].shape
                with torch.no_grad():
                    state_dict[k].copy_(state_dict_hf[k])

        return model
    
    
class Captioner(nn.Module):
    
    def __init__(self):
        super().__init__()
        vit_name = "google/vit-base-patch16-224-in21k"
        #vit_name = "google/vit-base-patch16-224"
        self.vit = ViTModel.from_pretrained(vit_name)
        for param in self.vit.parameters():
            param.requires_grad = False
            
        model_name = "gpt2"
        self.gpt2 = GPT.from_pretrained(model_name)
        #self.gpt2.config.pad_token_id = self.gpt2.config.eos_token_id
        
        
    def forward(self, image, caption, attn_mask=None):
        # image of shape (batch_size, color_chanel, hieght, width)
        # caption is of shape (batch_size, seq_len)
        # attn_mask is of shape (batch_size, seq_len)
        encoder_out = self.vit(image)
        encoder_out = encoder_out.pooler_output # (batch_size, embed_dim)
        
        return self.gpt2(caption, encoder_out, attn_mask)
    
    @torch.no_grad
    def generate(self, image, max_new_tokens=45, temperature=0.1, top_k=50, end_token=vocab.tokenizer.eos_token):
        caption = []
        encoder_out = self.vit(image).pooler_output # (batch_size, embed_dim)
        start_output = torch.tensor([[]]*image.shape[0], device=image.device, dtype=torch.long)
        for _ in range(max_new_tokens):
            logits = self.gpt2(start_output, encoder_out)
            logits = logits[:,-1,:]/temperature# (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_prob, topk_indeces = torch.topk(probs, top_k, dim=-1) #both output are of shape (batch_size, 50)
            ix = torch.multinomial(topk_prob, 1, ) # (batch_size, 1)
            xcol = torch.gather(topk_indeces, -1, ix) # (batch_size, 1)
            start_output = torch.cat([start_output, xcol], dim=-1)
        
        return start_output.tolist()
    

    @torch.no_grad
    def manual_generate(self, image, max_new_tokens=45, temperature=0.1, top_k=50, end_token=vocab.tokenizer.eos_token):
        encoder_out = self.vit(image).pooler_output # (batch_size, embed_dim)
        pos = torch.arange(0, 1, dtype=torch.long, device=image.device)
        pos_embed = self.gpt2.transformer.wpe(pos)
        x = pos_embed.unsqueeze(0) + encoder_out.unsqueeze(1)
        for block in self.gpt2.transformer.h:
            x = block(x)
        logits = self.gpt2.lm_head(self.gpt2.transformer.ln_f(x)).squeeze(1)
        probs = F.softmax(logits, dim=-1)
        topk_prob, topk_indeces = torch.topk(probs, top_k, dim=-1)
        ix = torch.multinomial(topk_prob, 1, )
        xcol = torch.gather(topk_indeces, -1, ix)
        start_output = xcol
        for _ in range(max_new_tokens):
            logits = self.gpt2(start_output, encoder_out)
            logits = logits[:,-1,:]/temperature# (batch_size, vocab_size)
            probs = F.softmax(logits, dim=-1)
            topk_prob, topk_indeces = torch.topk(probs, top_k, dim=-1) #both output are of shape (batch_size, 50)
            ix = torch.multinomial(topk_prob, 1, ) # (batch_size, 1)
            xcol = torch.gather(topk_indeces, -1, ix) # (batch_size, 1)
            start_output = torch.cat([start_output, xcol], dim=-1)
        
        return start_output.tolist()
    
    def greedy(self, image, max_new_tokens=45):
        encoder_out = self.vit(image).pooler_output # (batch_size, embed_dim)
        pos = torch.arange(0, 1, dtype=torch.long, device=image.device)
        pos_embed = self.gpt2.transformer.wpe(pos)
        x = pos_embed.unsqueeze(0) + encoder_out.unsqueeze(1)
        for block in self.gpt2.transformer.h:
            x = block(x)
        logits = self.gpt2.lm_head(self.gpt2.transformer.ln_f(x))
        start_output =logits.argmax(dim=-1) # (batch_size, 1, vocacb_size) -> (batch_size, 1)
        
        for _ in range(max_new_tokens-1):
            logits = self.gpt2(start_output, encoder_out)
            logits = logits[:,-1,:]# (batch_size, vocab_size)
            xcol = logits.argmax(dim=-1, keepdim=True)
            start_output = torch.cat([start_output, xcol], dim=-1)
        
        return start_output.tolist()