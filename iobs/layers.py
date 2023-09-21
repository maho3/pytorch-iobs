import torch
from torch import nn, Tensor


class IOBLayer(nn.Module):
    def __init__(self, num_features: int, device=None) -> None:
        """Pytorch module which applies IOB masking on latent variables.
        Currently only supports dense connections with inputs of shape
        (batch_size, num_features).

        Args:
            num_features (int): Number of features to expect in the input.
                Equivalent to maximum bottleneck width.
            device (_type_, optional): Device on which this module will
                be stored. Can be set to None, 'cpu', 'gpu', or 'mps'.
                Defaults to None.
        """
        super().__init__()
        self.num_features = num_features
        self.range = torch.arange(1, 1+self.num_features).to(device)

    def forward(self, input: Tensor) -> Tensor:
        """Dummy forward function which just passes all features through
        without masking. Equivalent to fully open bottleneck or
        nn.Identity.
        """
        return input

    def forward_mask(self, input: Tensor, mask: Tensor) -> Tensor:
        """Apply a custom mask to the input. Mask must be of shape
        (num_features) or (batch_size, num_features).
        """
        return input*mask

    def forward_neck(self, input: Tensor, n_open: int) -> Tensor:
        """Pass though only the first n_open nodes for the whole batch.
        Equivalent to a bottleneck of width n_open.
        """
        mask = self.range.le(n_open)
        return self.forward_mask(input, mask)

    def forward_all(self, input: Tensor) -> Tensor:
        """Pass through the batch for all possible bottleneck widths.
        Creates a new axis in the output batch to store different bottleneck
        configurations. For example, if the input is of shape
        (batch_size, num_features), then this function will output a Tensor
        of shape (batch_size, num_features+1, num_features), where the first
        index of the second axis is equivalent to n_open=0 and the last
        index is equivalent to n_open=num_features.
        """
        input = input.unsqueeze(1)
        input = input.expand(-1, self.num_features+1, self.num_features)
        return torch.tril(input, diagonal=-1)  # accounting for fully-closed


class StochasticIOBLayer(IOBLayer):
    def __init__(self, num_features: int, dist='uniform', device=None,
                 min_open=0, **dist_kwargs) -> None:
        super().__init__(num_features, device)

        self.min_open = 0
        self.dist = dist
        self.dist_kwargs = dist_kwargs

    def _sample_n_open(self) -> int:
        """Returns a random sampling of the distribution specified in
        self.dist.

        Returns:
            int: The number of open bottleneck connections for this forward
                pass of the layer.
        """
        if self.dist == 'uniform':
            sample = torch.randint(0, self.num_features-self.min_open)
        elif self.dist == 'geometric':
            sample = torch.distributions.Geometric(**self.dist_kwargs).sample()
        elif self.dist == 'poisson':
            sample = torch.distributions.Poisson(**self.dist_kwargs).sample()
        else:
            raise NotImplementedError(
                f"Distribution '{self.dist}' is not currently supported for "
                "object StochasticIOBLayer.")

        return min(
            self.num_features,
            self.min_open + sample
        )

    def forward(self, input: Tensor) -> Tensor:
        """Randomly sample the number of open connections using the
        pre-specified distribution.
        """
        return super().forward_neck(input, self._sample_n_open())
