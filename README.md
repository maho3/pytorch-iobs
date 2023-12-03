# Pytorch IOBs
[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](https://github.com/maho3/pytorch-iobs/blob/main/LICENSE)
[![tests](https://github.com/maho3/pytorch-iobs/actions/workflows/iob-tests.yml/badge.svg)](https://github.com/maho3/pytorch-iobs/actions/workflows/iob-tests.yml)

A lightweight implementation of Information-Ordered Bottlenecks (IOBs) in PyTorch. **For theory details, see the [paper](https://arxiv.org/abs/2305.11213).**

## Tutorial
The IOB implementation is extremely simple and designed to natively wrap existing `torch.nn` layers. The core of our implementation is the [`IOBLayer`](iobs/layers.py#L5) object, which allows one to mimic a bottleneck of variable width by selectively masking portions of its input.

The `IOBLayer` can be used in PyTorch's functional API. Consider the following 2-layer linear autoencoder.
```python
from torch import nn
from iobs.layers import IOBLayer

class AutoEncoder(nn.Module):
    def __init__(self, data_dim):
        super().__init__()
        self.encoder = nn.Linear(data_dim, 8)
        self.decoder = nn.Linear(8, data_dim)
        self.bottleneck = IOBLayer(8)

    def forward(self, batch_features, n_open):
        code = self.encoder(batch_features)
        code = self.bottleneck.forward_neck(code, n_open)
        reconstructed = self.decoder(code)
        return reconstructed
```
Here, the `forward_neck` function only allows information to pass through the first `n_open` nodes of the previous hidden layer, by masking the other inputs. However, it outputs a fixed dimensional tensor, allowing functional consistency with the downstream layers. As an example, running the above `forward` function with `n_open=5` is equivalent to an autoencoder with a bottleneck width of 5 nodes.

## Functionality
The `IOBLayer` has four differentiable methods to pass forward information.
* `IOBLayer.forward(input)` passes all the information through all the nodes and is equivalent to an Identity matrix multiplication.
* `IOBLayer.forward_neck(input, n_open)` only passes information through the first `n_open` nodes.
* `IOBLayer.forward_mask(input, mask)` allows one to pass a custom mask which will be applied to the latents. For a batch size of 1, this is functionally equivalent to `input*mask`.
* `IOBLayer.forward_all(input)` passes information through all possible `n_open` bottlenecks, and aggregates all configurations into a new batch dimension. For example, if `input.shape=(64,8)`, wherein the max width of the IOBLayer is 8, then the output of `IOBLayer.forward_all(input)` is of shape `(64,9,8)`, where each element of the second axis represents a different `n_open` (including `n_open=0`). This can be later flattened into a larger batch (e.g. `(64*9,8)`) and passed as a regular tensor to the downstream architecture.

## Installation
The installation of pytorch-iobs is designed to be a lightweight wrapper on top of pytorch. The only dependencies are recent versions of `numpy`, `torch`, and `tqdm`. To install, simply clone the repository and use `pip install`.
```bash
git clone git@github.com:maho3/pytorch-iobs.git
cd pytorch-iobs
pip install -e .
```

## Examples
See the practical examples in `notebooks/` for demonstrations on using IOBLayers during training and inference.