import torch
from iobs.layers import IOBLayer


def test_iob_init():
    # Test IOBLayer initialization
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iob_layer = IOBLayer(num_features, device)
    assert iob_layer.num_features == num_features
    assert iob_layer.device == device


def test_iob_forward():
    # Test forward function
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iob_layer = IOBLayer(num_features, device)
    input = torch.randn(10, num_features)
    output = iob_layer.forward(input)
    assert torch.allclose(output, input)


def test_iob_forward_mask():
    # Test forward_mask function
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iob_layer = IOBLayer(num_features, device)
    input = torch.randn(10, num_features)
    mask = torch.tensor([1, 0, 1, 0, 1])
    output_masked = iob_layer.forward_mask(input, mask)
    assert torch.allclose(output_masked, input * mask)


def test_iob_forward_neck():
    # Test forward_neck function
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iob_layer = IOBLayer(num_features, device)
    input = torch.randn(10, num_features)
    n_open = 3
    output_neck = iob_layer.forward_neck(input, n_open)
    expected_output_neck = input * torch.tensor([1, 1, 1, 0, 0])
    assert torch.allclose(output_neck, expected_output_neck)


def test_iob_forward_all():
    # Test forward_all function
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    iob_layer = IOBLayer(num_features, device)
    input = torch.randn(10, num_features)
    output_all = iob_layer.forward_all(input)
    expected_output_all = torch.tril(
        input.unsqueeze(1).expand(-1, num_features+1, num_features),
        diagonal=-1)
    assert torch.allclose(output_all, expected_output_all)
