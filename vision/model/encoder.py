
import torch
import torch.nn as nn
from transformers import ViTModel

# Define an Vision Encoder module from the vit architecture
class VisionEncoder(nn.Module):
    def __init__(self, hidden_size=128, model_name = 'google/vit-large-patch16-224-in21k'):
        super(VisionEncoder, self).__init__()
        
        self.model = ViTModel.from_pretrained(model_name, torch_dtype=torch.float32)
        self.normalize = nn.Linear(self.model.config.hidden_size, hidden_size)
        self._freeze()

    def _freeze(self) -> None:
        for param in self.model.parameters():
            param.requires_grad = False
                
    def forward(self, image):
        '''
        '''
        outputs = self.model(image).last_hidden_state
        return self.normalize(outputs)