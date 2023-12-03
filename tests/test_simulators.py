import numpy as np
from iobs.simulators import make_n_disk


def test_make_n_disk():
    # Test case 1: Check if the output shape is correct
    n_samples = 10
    n_disks = 2
    dim = 32
    size_range = (2, 5)
    overlap = True
    verbose = False

    output = make_n_disk(n_samples, n_disks, dim, size_range, overlap, verbose)
    assert output.shape == (n_samples, dim, dim, 1)

    # Test case 2: Check if the output min and max values are correct
    output_min = 0
    output_max = 1
    assert np.min(output) == output_min
    assert np.max(output) == output_max

    # Test case 3: Check that we can make non-overlapping disks
    overlap = False
    output = make_n_disk(n_samples, n_disks, dim, size_range, overlap, verbose)
    assert output.shape == (n_samples, dim, dim, 1)
