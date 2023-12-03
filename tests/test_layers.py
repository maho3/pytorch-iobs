import torch
from iobs.layers import IOBLayer, StochasticIOBLayer


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


def test_stochastic_iob_init():
    # Test StochasticIOBLayer initialization
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    min_open = 2
    stochastic_iob_layer = StochasticIOBLayer(
        num_features, min_open=min_open, device=device)
    assert stochastic_iob_layer.num_features == num_features
    assert stochastic_iob_layer.device == device
    assert stochastic_iob_layer.min_open == min_open


def test_stochastic_iob_sample_n_open():
    # Test _sample_n_open function of StochasticIOBLayer
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    min_open = 2
    stochastic_iob_layer = StochasticIOBLayer(
        num_features, min_open=min_open, device=device)

    # Test uniform distribution
    stochastic_iob_layer.dist = 'uniform'
    for _ in range(10):
        n_open = stochastic_iob_layer._sample_n_open()
        assert min_open <= n_open <= num_features

    # Test geometric distribution
    stochastic_iob_layer.dist = 'geometric'
    stochastic_iob_layer.dist_kwargs = {'probs': 0.5}
    for _ in range(10):
        n_open = stochastic_iob_layer._sample_n_open()
        assert min_open <= n_open <= num_features

    # Test poisson distribution
    stochastic_iob_layer.dist = 'poisson'
    stochastic_iob_layer.dist_kwargs = {'rate': 2}
    for _ in range(10):
        n_open = stochastic_iob_layer._sample_n_open()
        assert min_open <= n_open <= num_features


def test_stochastic_iob_forward_neck():
    # Test forward_neck function of StochasticIOBLayer
    num_features = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    min_open = 2
    stochastic_iob_layer = StochasticIOBLayer(
        num_features, min_open=min_open, device=device)
    input = torch.randn(10, num_features)
    n_open = stochastic_iob_layer._sample_n_open()
    output_neck = stochastic_iob_layer.forward_neck(input, n_open)
    expected_output_neck = input * \
        torch.cat([torch.ones(n_open), torch.zeros(num_features - n_open)])
    assert torch.allclose(output_neck, expected_output_neck)
