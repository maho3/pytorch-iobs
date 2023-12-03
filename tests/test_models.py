import torch
import torch.nn as nn
from iobs.models import BaseAE
from iobs.layers import IOBLayer


def test_base_ae_init():
    # Test BaseAE initialization
    num_features = 16
    latent_dim = 10
    encoder = nn.Linear(num_features, latent_dim)
    decoder = nn.Linear(latent_dim, num_features)
    bottleneck = IOBLayer(latent_dim)
    model = BaseAE(encoder, decoder, bottleneck)
    assert model.encoder == encoder
    assert model.decoder == decoder
    assert model.bottleneck == bottleneck


def test_base_ae_forward():
    # Test forward function of BaseAE
    num_features = 16
    latent_dim = 10
    encoder = nn.Linear(num_features, latent_dim)
    decoder = nn.Linear(latent_dim, num_features)
    bottleneck = IOBLayer(latent_dim)
    model = BaseAE(encoder, decoder, bottleneck)
    input = torch.randn(10, num_features)
    output = model.forward(input)
    assert output.shape == input.shape


def test_base_ae_forward_mask():
    # Test forward_mask function of BaseAE
    num_features = 16
    latent_dim = 5
    encoder = nn.Linear(num_features, latent_dim)
    decoder = nn.Linear(latent_dim, num_features)
    bottleneck = IOBLayer(latent_dim)
    model = BaseAE(encoder, decoder, bottleneck)
    input = torch.randn(10, num_features)
    mask = torch.tensor([1, 0, 1, 0, 1])
    output_masked = model.forward_mask(input, mask)
    assert output_masked.shape == input.shape


def test_base_ae_forward_neck():
    # Test forward_neck function of BaseAE
    num_features = 16
    latent_dim = 10
    encoder = nn.Linear(num_features, latent_dim)
    decoder = nn.Linear(latent_dim, num_features)
    bottleneck = IOBLayer(latent_dim)
    model = BaseAE(encoder, decoder, bottleneck)
    input = torch.randn(10, num_features)
    n_open = 3
    output_neck = model.forward_neck(input, n_open)
    assert output_neck.shape == input.shape


def test_base_ae_forward_all():
    # Test forward_all function of BaseAE
    num_features = 16
    latent_dim = 10
    encoder = nn.Linear(num_features, latent_dim)
    decoder = nn.Linear(latent_dim, num_features)
    bottleneck = IOBLayer(latent_dim)
    model = BaseAE(encoder, decoder, bottleneck)
    input = torch.randn(10, num_features)
    output_all = model.forward_all(input)
    assert output_all.shape == (input.shape[0], latent_dim+1, input.shape[1])
