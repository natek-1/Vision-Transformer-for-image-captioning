import torch
import torch.nn as nn


# Define a decoder module from the gpt-2 architecture
class Decoder(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_size=128, num_layers=3, num_heads=4):
        super(Decoder, self).__init__()
        
        # Create an embedding layer for tokens
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Create multiple decoder layers
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, 
                                                   dim_feedforward=hidden_size * 4, dropout=0.0,
                                                   batch_first=True, norm_first=True)
        # TransformerDecoder will clone the decoder_layer "num_layers" times
        self.decoder_layers = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.register_buffer('tril', torch.tril(torch.ones(max_length, max_length)))
        self.norm = nn.LayerNorm(hidden_size)
                
        # Define a linear layer for output prediction
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
    def forward(self, input_seq, encoder_output, input_padding_mask=None, 
                encoder_padding_mask=None):        
        # Embed the input sequence
        input_embs = self.token_embedding(input_seq)
        batch_size, seq_len, hidden_size = input_embs.shape

        # Add positional embeddings to the input embeddings
        seq_idx = torch.arange(seq_len, device=input_seq.device)
        seq_idx = self.position_embedding(seq_idx)
        seq_idx = seq_idx.unsqueeze(0)
        embs = input_embs + seq_idx
        casual_mask = self.tril[:seq_len, :seq_len] == 0
        
        # Pass the embeddings through each transformer block
        output = self.decoder_layers(tgt=embs, memory=encoder_output, tgt_mask=casual_mask,
                                     tgt_key_padding_mask=input_padding_mask, 
                                     memory_key_padding_mask=encoder_padding_mask)
        
        return self.fc_out(self.norm(output))