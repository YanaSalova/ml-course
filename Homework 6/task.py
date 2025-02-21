import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super().__init__()
    
    def forward(self, x):
        return None, (None, None)
    
    
# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super().__init__()
        
    def forward(self, z):
        return None
    
# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=3, latent_size=256, down_channels=6, up_channels=12):
        super().__init__()
        self.encoder = None
        self.decoder = None
        
    def forward(self, x):
        return x_pred, kld
    
    def encode(self, x):
        return z
    
    def decode(self, z):
        return x_pred
    
    def save(self):
        pass
    
    def load(self)
        pass