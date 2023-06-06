import torch
from torch import nn, Tensor


class IOBLayer(nn.Module):
    def __init__(self, num_features: int, device=None) -> None:
        super().__init__()
        self.num_features = num_features
        self.range = torch.arange(1, 1+self.num_features).to(device)

    def forward(self, input: Tensor) -> Tensor:
        return input

    def forward_mask(self, input, mask) -> Tensor:
        return input*mask

    def forward_neck(self, input, n_open) -> Tensor:
        mask = self.range.le(n_open)
        return self.forward_mask(input, mask)

    def forward_all(self, input) -> Tensor:
        input = input.unsqueeze(1)
        input = input.expand(-1, self.num_features+1, self.num_features)
        return torch.tril(input, diagonal=-1)  # accounting for fully-closed


class StochasticIOBLayer(IOBLayer):
    def __init__(self, num_features: int, dist='uniform', device=None,
                 **dist_kwargs) -> None:
        super().__init__(num_features, device)

        self.min_open = 0
        self.dist = dist
        self.distargs = dist_kwargs

    def _sample_n_open(self) -> int:
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
        return super().forward_neck(input, self._sample_n_open())
