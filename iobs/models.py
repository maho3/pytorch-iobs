import torch
from torch import nn, Tensor
from .layers import IOBLayer


class BaseAE(nn.Module):
    def __init__(self,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 bottleneck: IOBLayer
                 ) -> None:
        """Basic autoencoder model with support for IOB forward functions

        Args:
            encoder (nn.Module): An encoder module which outputs a dense layer
            decoder (nn.Module): A decoder module which accepts a dense layer
            bottleneck (nn.Module): An IOBLayer module
        """
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.bottleneck = bottleneck
        self.latent_dim = bottleneck.num_features

    def forward(self, features: Tensor) -> Tensor:
        """Usual forward autoencoding with a fully-open bottleneck."""
        code = self.encoder(features)
        reconstructed = self.decoder(code)
        return reconstructed

    def forward_mask(self, features: Tensor, mask: Tensor) -> Tensor:
        """Apply a custom mask to the latent vector. Mask must be of shape
        (self.latent_dim) or (batch_size, self.latent_dim).
        """
        code = self.encoder(features)
        code = self.bottleneck.forward_mask(code, mask)
        reconstructed = self.decoder(code)
        return reconstructed

    def forward_neck(self, features: Tensor, n_open: int) -> Tensor:
        """Forward autoencoding with a bottleneck of width n_open."""
        code = self.encoder(features)
        code = self.bottleneck.forward_neck(code, n_open)
        reconstructed = self.decoder(code)
        return reconstructed

    def forward_all(self, features: Tensor) -> Tensor:
        """Pass through the batch for all possible bottleneck widths.
        Creates a new axis in the output batch to store different bottleneck
        configurations. For example, if the input is of shape
        (batch_size, *in_shape), then this function will output a Tensor
        of shape (batch_size, self.latent_dim+1, *in_shape), where the first
        index of the second axis is equivalent to n_open=0 and the last
        index is equivalent to n_open=num_features.
        """
        batch_size = len(features)
        code = self.encoder(features)
        code = self.bottleneck.forward_all(code)
        code = torch.flatten(code, end_dim=1)
        reconstructed = self.decoder(code)
        reconstructed = torch.unflatten(
            reconstructed, dim=0, sizes=(batch_size, -1))
        return reconstructed
