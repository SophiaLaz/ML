import torch
from torch import nn
from torch.nn import functional as F


# Task 1

class Encoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, start_channels=16, downsamplings=5):
        super().__init__()
        self.latent_size = latent_size
        small_model = nn.Sequential(*[
            nn.Sequential(
                nn.Conv2d(start_channels * 2 ** i, start_channels * 2 ** (i + 1), 3, 2, 1),
                nn.BatchNorm2d(start_channels * 2 ** (i + 1)),
                nn.ReLU()
            )
            for i in range(downsamplings)
        ])
        self.model = nn.Sequential(
            nn.Conv2d(3, start_channels, 1, 1, 0),
            small_model,
            nn.Flatten(),
            nn.Linear(start_channels * img_size**2 // 2**downsamplings, 2 * latent_size)
        )

    def forward(self, x):
        [mu, sigma] = torch.split(self.model(x), self.latent_size, dim=1)
        sigma = torch.exp(sigma)
        embedding = mu + torch.randn(sigma.shape).cuda() * sigma
        return embedding, (mu, sigma)
    
    
# Task 2

class Decoder(nn.Module):
    def __init__(self, img_size=128, latent_size=512, end_channels=16, upsamplings=5):
        super().__init__()
        small_model = nn.Sequential(*[
            nn.Sequential(
                nn.ConvTranspose2d(end_channels * 2**i, end_channels * 2**(i-1), 4, 2, 1),
                nn.BatchNorm2d(end_channels * 2**(i-1)),
                nn.ReLU()
            )
            for i in range(upsamplings, 0, -1)
        ])
        self.model = nn.Sequential(
            nn.Linear(latent_size, end_channels * img_size ** 2 // 2**upsamplings),
            nn.Unflatten(1, (end_channels * 2 ** upsamplings, img_size // 2**upsamplings, img_size // 2**upsamplings)),
            small_model,
            nn.Conv2d(end_channels, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)
    
# Task 3

class VAE(nn.Module):
    def __init__(self, img_size=128, downsamplings=5, latent_size=256, down_channels=6, up_channels=10):
        super(VAE, self).__init__()
        self.encoder = Encoder(img_size, latent_size, down_channels, downsamplings)
        self.decoder = Decoder(img_size, latent_size, up_channels, downsamplings)

    def forward(self, x):
        z, (mu, sigma) = self.encoder(x)
        x_pred = self.decoder(z)
        kld = 0.5 * (torch.square(sigma) + torch.square(mu) - 2 * torch.log(sigma) - torch.ones(mu.shape).cuda())
        return x_pred, kld

    def encode(self, x):
        res, _ = self.encoder(x)
        return res

    def decode(self, z):
        return self.decoder(z)

    def save(self):
        torch.save(self.state_dict(), __file__[:-7] + 'checkpoint.pth')

    def load(self):
        self.load_state_dict(torch.load(__file__[:-7] + 'checkpoint.pth'))
        self.eval()
