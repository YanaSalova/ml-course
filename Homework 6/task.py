import torch
from torch import nn
from torch.nn import functional as F

#Task 1

class Encoder(nn.Module):
    def __init__(
        self,
        img_size=128,
        latent_size=512,
        start_channels=16,
        downsamplings=5
    ):
        """
        Параметры:
        - img_size: размер входного изображения (высота = ширина).
        - latent_size: размер латентного вектора.
        - start_channels: кол-во каналов после первого свёрточного слоя.
        - downsamplings: кол-во даунсэмплингов (Conv stride=2).
        """
        super().__init__()
        
        self.initial_conv = nn.Conv2d(
            in_channels=3,
            out_channels=start_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )

        layers = []
        channels = start_channels
        for _ in range(downsamplings):
            layers.append(nn.Conv2d(channels, channels * 2, kernel_size=3, stride=2, padding=1))
            layers.append(nn.BatchNorm2d(channels * 2))
            layers.append(nn.ReLU(inplace=True))
            channels *= 2
        self.downsampling_blocks = nn.Sequential(*layers)

        final_size = img_size // (2 ** downsamplings)

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(channels * final_size * final_size, 2 * latent_size)

    def forward(self, x):
        """
        На вход: x, shape=[batch_size, 3, img_size, img_size].
        На выходе: 
          z = mu + eps * sigma,
          (mu, sigma) — тензоры размера [batch_size, latent_size].
        """
        x = self.initial_conv(x)
        x = self.downsampling_blocks(x)
        x = self.flatten(x)

        params = self.linear(x)  
        mu, log_sigma = torch.chunk(params, 2, dim=1)
        sigma = torch.exp(log_sigma)
        eps = torch.randn_like(sigma)
        z = mu + eps * sigma
        
        return z, (mu, sigma)
    


# Task 2
class Decoder(nn.Module):
    def __init__(
        self,
        img_size=128,
        latent_size=512,
        end_channels=16,
        upsamplings=5
    ):
        super().__init__()
        feat_size = img_size // (2 ** upsamplings)

        self.linear = nn.Sequential(
            nn.Linear(latent_size, end_channels * feat_size * feat_size),
            nn.ReLU(inplace=True)
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(end_channels, feat_size, feat_size))
     
        layers = []
        channels = end_channels
        for _ in range(upsamplings):
            out_ch = max(1, channels // 2)
            layers.append(nn.ConvTranspose2d(
                channels, out_ch,
                kernel_size=4, stride=2, padding=1
            ))
            layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            channels = out_ch
        self.upsampling_blocks = nn.Sequential(*layers)
        
        self.final_conv = nn.Conv2d(channels, 3, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Tanh()

    def forward(self, z):
        x = self.linear(z)
        x = self.unflatten(x)
        x = self.upsampling_blocks(x)
        x = self.final_conv(x)
        x = self.activation(x)  
        return x
    


# Task 3
class VAE(nn.Module):
    def __init__(
        self,
        img_size=128,
        downsamplings=3,
        latent_size=256,
        down_channels=6,
        up_channels=12
    ):
        super().__init__()
        
        self.encoder = Encoder(
            img_size=img_size,
            latent_size=latent_size,
            start_channels=down_channels,
            downsamplings=downsamplings
        )
      
        self.decoder = Decoder(
            img_size=img_size,
            latent_size=latent_size,
            end_channels=up_channels,
            upsamplings=downsamplings
        )
        
    def forward(self, x):
        """
        1) Пропускаем x через encoder -> получаем z, (mu, sigma)
        2) Пропускаем z через decoder -> получаем x_pred
        3) Считаем KLD
        """
        z, (mu, sigma) = self.encoder(x)
        x_pred = self.decoder(z)

        kld = 0.5 * torch.sum(
            sigma**2 + mu**2 - torch.log(sigma**2 + 1e-8) - 1, dim=1
        ).mean()

        return x_pred, kld
    
    def encode(self, x):
        """
        Возвращает одно сэмплирование z из энкодера 
        """
        z, _ = self.encoder(x)
        return z
    
    def decode(self, z):
        """
        Декодирует латентный вектор z в изображение
        """
        return self.decoder(z)
    
    def save(self, path="vae_checkpoint.pth"):
        torch.save(self.state_dict(), path)
    
    def load(self, path="vae_checkpoint.pth"):
        self.load_state_dict(torch.load(path))
