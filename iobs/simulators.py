import numpy as np
import tqdm


def make_n_disk(n_samples: int, n_disks: int, dim=32, size_range=(2, 5),
                overlap=True, verbose=False) -> np.ndarray:

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
