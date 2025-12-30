import torch
import torch.nn as nn
import torch.nn.functional as F

from vision.model.encoder import VisionEncoder
from vision.model.decoder import Decoder

class TokenDrop(nn.Module):
    """For a batch of tokens indices, randomly replace a non-specical token.
    
    Args:
        prob (float): probability of dropping a token
        blank_token (int): index for the blank token
        num_special (int): Number of special tokens, assumed to be at the start of the vocab
    Returns:
        torch.Tensor: Tensor of same shape as input with some tokens replaced by blank_token
    """

    def __init__(self, prob=0.1, blank_token=103, special_tokens=[0, 101, 102]):
        self.prob = prob
        self.special_tokens= special_tokens
        self.blank_token = blank_token

    def __call__(self, sample):
        # Randomly sample a bernoulli distribution with p=prob
        # to create a mask where 1 means we will replace that token
        prob_mask = torch.rand(sample.shape, device=sample.device) < self.prob
        
        # only replace if the token is not the eos token
        protected_mask = torch.zeros(sample.shape, dtype=torch.bool, device=sample.device)
        for t in self.special_tokens:
            protected_mask = protected_mask | (sample == t)
        drop_mask = prob_mask & ~protected_mask
        
        sample_out = torch.where(drop_mask, torch.tensor(self.blank_token, device=sample.device), sample)
        return sample_out

class VisionEncoderDecoder(nn.Module):
    def __init__(self, image_size, channels_in, num_emb, patch_size=16, 
                 hidden_size=128, num_layers=(3, 3), num_heads=4, mlp_dropout=0.1, att_dropout=0.1):
        super(VisionEncoderDecoder, self).__init__()
        
        # Create an encoder and decoder with specified parameters
        self.encoder = VisionEncoder(image_size=image_size, channels_in=channels_in, 
                                     patch_size=patch_size, hidden_size=hidden_size, 
                                     num_layers=num_layers[0], num_heads=num_heads, mlp_dropout=mlp_dropout, att_dropout=att_dropout)
        
        self.decoder = Decoder(num_emb=num_emb, hidden_size=hidden_size, 
                               num_layers=num_layers[1], num_heads=num_heads, mlp_dropout=mlp_dropout, att_dropout=att_dropout)

    def forward(self, input_image, target_seq, padding_mask):
        # Generate padding masks for the target sequence
        bool_padding_mask = padding_mask == 0

        encoded_seq = self.encoder(image=input_image)
        decoded_seq = self.decoder(input_seq=target_seq, 
                                   encoder_output=encoded_seq, 
                                   input_padding_mask=bool_padding_mask)
        return decoded_seq

    @torch.inference_mode()
    def generate(self, input_image, sos_token, eos_token, pad_token, max_length=90, temp=0.4):
        device = input_image.device
        batch_size = input_image.size(0)
        
        encoded_seq = self.encoder(image=input_image)
        
        # Initialize sequence with SOS token
        generated_ids = torch.full((batch_size, 1), sos_token, dtype=torch.long, device=device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # Pre-create pad token tensor (avoid creating in loop)
        pad_token_tensor = torch.tensor(pad_token, dtype=torch.long, device=device)
        
        for i in range(max_length):
            # Create padding mask for finished sequences
            # True = should be masked (PyTorch convention for key_padding_mask)
            padding_mask = finished.unsqueeze(1).expand(-1, generated_ids.size(1))
            
            # Get logits for next token
            logits = self.decoder(
                input_seq=generated_ids, 
                encoder_output=encoded_seq,
                input_padding_mask=padding_mask
            )
            
            # Apply temperature and get next token
            if temp == 0.0:
                logits = logits[:, -1, :]
                probs = F.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # [batch, 1]
            else:
                logits = logits[:, -1, :] / temp
                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [batch, 1]
            
            # Replace with pad token for finished sequences
            next_token = torch.where(
                finished.unsqueeze(-1),  # ✅ Match shape [batch, 1]
                pad_token_tensor, 
                next_token
            )
            
            # Append to generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            
            # Update finished status
            finished |= (next_token.squeeze(-1) == eos_token)
            
            # Early stopping
            if finished.all():
                break
        
        return generated_ids




