import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim):
        super(Generator, self).__init__()

        def block(in_feat, out_feat):
            layers = [nn.Linear(in_feat, out_feat)]
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(latent_dim, 128),
            *block(128, 256),
            # *block(256, 512),
            # *block(512, 1024),
            nn.Linear(256, latent_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        vector = self.model(x)
        return vector
