"""EuroSAT loading, a leakage-safe split, and train-only normalization.

EuroSAT RGB is a folder of 27,000 uint8 64x64 JPEGs in 10 class folders, served
by `torchvision.datasets.EuroSAT` (an `ImageFolder`). The download is the
HuggingFace mirror (~90 MB), not the flaky DFKI host.

Design choices that the runbook explains and the owner must be able to defend:
- The train/val/test split is **stratified per class** and seeded, so it is
  reproducible and every class keeps the same proportion in each split.
- Normalization statistics are computed on the **training split only**. Using
  the whole dataset would leak validation/test pixel statistics into training.
- `tiny_subset` returns random tensors so tests never touch the real download.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

NUM_CLASSES = 10
SEED = 42
IMG_PX = 64
ROOT = "data/eurosat"
FRACS = (0.7, 0.15, 0.15)  # train / val / test

# EuroSAT class folders in ImageFolder (alphabetical) order, for reference and
# for labelling the confusion matrix.
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
    "Pasture", "PermanentCrop", "Residential", "River", "SeaLake",
]


def split_indices(labels, seed=SEED, fracs=FRACS):
    """Stratified, reproducible train/val/test index split.

    Splits each class separately so the proportions in `fracs` hold per class
    (no class is starved in any split) and the three index sets are disjoint and
    cover every sample. Returns three sorted int arrays.
    """
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    f_tr, f_va, _ = fracs
    tr, va, te = [], [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n = idx.size
        n_tr = int(round(f_tr * n))
        n_va = int(round(f_va * n))
        tr.append(idx[:n_tr])
        va.append(idx[n_tr:n_tr + n_va])
        te.append(idx[n_tr + n_va:])
    return (np.sort(np.concatenate(tr)),
            np.sort(np.concatenate(va)),
            np.sort(np.concatenate(te)))


def tiny_subset(n=64, seed=0):
    """A synthetic dataset of random images and labels, for fast tests.

    Mirrors the real samples' shape (3, 64, 64) so models and the training loop
    can be exercised without downloading EuroSAT.
    """
    g = torch.Generator().manual_seed(seed)
    images = torch.rand(n, 3, IMG_PX, IMG_PX, generator=g)
    targets = torch.randint(0, NUM_CLASSES, (n,), generator=g)
    return TensorDataset(images, targets)


def load_eurosat(root=ROOT, transform=None, download=False):
    """The torchvision EuroSAT dataset (ImageFolder). Imported lazily so the
    module stays importable, and tests stay fast, without torchvision side
    effects."""
    from torchvision.datasets import EuroSAT
    return EuroSAT(root=root, transform=transform, download=download)


def compute_norm_stats(dataset, indices, batch_size=256):
    """Per-channel mean and std over the given indices only (the train split).

    `dataset` must yield tensors in [0, 1] (i.e. built with ToTensor). Returns
    two length-3 float tensors.
    """
    loader = DataLoader(Subset(dataset, list(indices)), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    n = 0
    s = torch.zeros(3)
    ss = torch.zeros(3)
    for x, _ in loader:
        x = x.reshape(x.shape[0], 3, -1)
        n += x.shape[0] * x.shape[2]
        s += x.sum(dim=(0, 2))
        ss += (x ** 2).sum(dim=(0, 2))
    mean = s / n
    std = torch.sqrt(ss / n - mean ** 2)
    return mean, std


def make_loaders(root=ROOT, seed=SEED, batch_size=64, download=False,
                 augment=True, num_workers=2):
    """Build train/val/test DataLoaders with a leakage-safe split and
    train-only normalization.

    Light flips are the only augmentation on the train split: overhead imagery
    has no canonical up/down or left/right, so horizontal and vertical flips are
    label-preserving. Val and test get normalization only.
    """
    from torchvision import transforms

    base = load_eurosat(root=root, download=download)
    labels = np.array(base.targets)
    tr_idx, va_idx, te_idx = split_indices(labels, seed=seed)

    to_tensor = transforms.ToTensor()
    stats_ds = load_eurosat(root=root, transform=to_tensor)
    mean, std = compute_norm_stats(stats_ds, tr_idx)
    normalize = transforms.Normalize(mean.tolist(), std.tolist())

    eval_tf = transforms.Compose([to_tensor, normalize])
    if augment:
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            to_tensor,
            normalize,
        ])
    else:
        train_tf = eval_tf

    train_ds = Subset(load_eurosat(root=root, transform=train_tf), tr_idx)
    val_ds = Subset(load_eurosat(root=root, transform=eval_tf), va_idx)
    test_ds = Subset(load_eurosat(root=root, transform=eval_tf), te_idx)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers)
    stats = {"mean": mean.tolist(), "std": std.tolist(),
             "n_train": int(tr_idx.size), "n_val": int(va_idx.size),
             "n_test": int(te_idx.size)}
    return train_loader, val_loader, test_loader, base.classes, stats
