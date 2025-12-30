
import torch
import torch.nn as nn
from vision.model.attention import TransformerBlock

def extract_patches(image_tensor, patch_size=16):
    '''
    Extracts non-overlapping patches from the input image tensor.
    Args:
        image_tensor (torch.Tensor): Input image tensor of shape (BS, C, H, W).
        patch_size (int): Size of each square patch.
    Returns:
        torch.Tensor: Tensor of shape (BS, L, H) where L is the number of patches
                      and H is the flattened patch size (C * patch_size * patch_size).
    '''
    # Get the dimensions of the image tensor
    bs, c, _, _ = image_tensor.size()
    
    # Define the Unfold layer with appropriate parameters
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    unfolded = unfold(image_tensor)
    
    # Reshape the unfolded tensor to match the desired output shape
    unfolded = unfolded.transpose(1, 2).reshape(bs, -1, c * patch_size * patch_size)
    
    return unfolded


class VisionEncoder(nn.Module):
    def __init__(self, image_size, channels_in, patch_size=16, hidden_size=128, 
                 num_layers=3, num_heads=4, mlp_dropout=0.1, att_dropout=0.1):
        '''
        Vision Transformer Encoder that processes images by dividing them into patches
        and passing them through transformer blocks.
        Args:
            image_size (int): Height and width of the input images (assumed square).
            channels_in (int): Number of input channels (e.g., 3 for RGB).
            patch_size (int): Size of each square patch.
            hidden_size (int): Dimensionality of input and output features.
            num_layers (int): Number of transformer blocks.
            num_heads (int): Number of attention heads.
            mlp_dropout (float): Dropout rate for the feed-forward network.
            att_dropout (float): Dropout rate for attention layers.
        Returns:
            torch.Tensor: Encoded image features of shape (batch_size, num_patches, hidden_size).
        '''
        super(VisionEncoder, self).__init__()
        
        self.patch_size = patch_size
        self.fc_in = nn.Linear(channels_in * patch_size * patch_size, hidden_size)
        
        seq_length = (image_size // patch_size) ** 2
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, 
                                                      hidden_size).normal_(std=0.02))
        
        # Create multiple transformer blocks as layers
        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, 
                             decoder=False, masking=False,
                             mlp_dropout=mlp_dropout, att_dropout=att_dropout) for _ in range(num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(hidden_size)
                
    def forward(self, image):  

        patch_seq = extract_patches(image, patch_size=self.patch_size)
        x = self.fc_in(patch_seq)

        # Add a unique embedding to each token embedding
        x = x + self.pos_embedding
        
        # Pass the embeddings through each transformer block
        for block in self.blocks:
            x = block(x)
        
        return self.final_ln(x)