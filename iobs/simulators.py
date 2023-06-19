import numpy as np
import tqdm


def make_n_disk(n_samples: int, n_disks: int, dim=32, size_range=(2, 5),
                overlap=True, verbose=False) -> np.ndarray:
    """Generate an n-Disk dataset

    Args:
        n_samples (int): The number of images to sample.
        n_disks (int): The number of disks per image.
        dim (int, optional): The dimensionalisty of each image. Defaults to 32.
        size_range (tuple, optional): The range of possible radii (in pixels)
            of each disk. Defaults to (2, 5).
        overlap (bool, optional): Whether the disks are allowed to overlap
            with each other. If False, then we perform an interative search
            over all possible configurations until either we find one with
            no overlap or we reach the iteration limit (set at 1000).
            Defaults to True.
        verbose (bool, optional): Whether to use tqdm to print a progress bar.
            Defaults to False.

    Raises:
        RuntimeError: If the iteration limit is reached while searching for
            non-overlapping disk configurations. Will only occur if overlap
            is True.

    Returns:
        np.ndarray: Array of shape (n_samples, dim, dim) containing the whole
            dataset of image samples.
    """
    tol = 1000  # how many attempts to avoid overlapping
    xmesh, ymesh = np.meshgrid(np.arange(dim), np.arange(dim))
    data = np.zeros((n_samples, dim, dim, 1))
    for i in tqdm.tqdm(range(n_samples), disable=not verbose):
        rs, xs, ys = [], [], []  # record positions and sizes
        for _ in range(n_disks):
            for _ in range(tol):
                # sample positions and size
                r = np.random.uniform(*size_range)
                x, y = np.random.uniform(r, dim-r, size=2)
                # check if overlap with any previous disks
                if not overlap:
                    rerun = False
                    for i in range(len(rs)):
                        if np.sqrt((x-xs[i])**2 + (y-ys[i])**2) <= r+rs[i]:
                            rerun = True
                            break
                    if rerun:
                        continue
                # store
                rs += [r]
                xs += [x]
                ys += [y]
                break
            else:
                raise RuntimeError(
                    f"Couldn't place {n_disks} disks without overlapping "
                    "in {tol} tries.")
            # fill in disk in image
            rmesh = ((xmesh - x) ** 2 + (ymesh - y) ** 2)
            mask = rmesh < r**2
            data[i, mask] = 1
    return data
