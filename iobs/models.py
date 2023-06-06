import torch
from torch import nn


class BaseAE(nn.Module):
    def __init__(self, input_shape, encoder, decoder, bottleneck):
        super().__init__()

        self.input_shape = input_shape
        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.latent_dim = bottleneck.num_features

    def forward(self, features):
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

    def forward_neck(self, features, n_open):
        code = self.encoder(features)
        code = self.bottleneck.forward_neck(code, n_open)
        reconstructed = self.decoder(code)
        return reconstructed

    def forward_mask(self, features, mask):
        code = self.encoder(features)
        code = self.bottleneck.forward_mask(code, mask)
        reconstructed = self.decoder(code)
        return reconstructed

    def forward_all(self, features):
        batch_size = len(features)
        code = self.encoder(features)
        code = self.bottleneck.forward_all(code)
        code = torch.flatten(code, end_dim=1)
        reconstructed = self.decoder(code)
        reconstructed = torch.unflatten(
            reconstructed, dim=0, sizes=(batch_size, -1))
        return reconstructed
