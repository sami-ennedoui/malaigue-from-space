"""Smoke tests for the EuroSAT trained-CNN module.

These never download the dataset and never train to convergence: they run on
synthetic random tensors or a tiny in-memory subset and finish in well under a
second. Anything that needs the real EuroSAT download is gated with the same
`skipif` pattern as tests/test_embed.py.
"""
import os

import numpy as np
import pytest
import torch

from malaigue.eurosat import data, eval as ev, model, train_torch, viz

DATA_DIR = "data/eurosat/eurosat/2750"
data_ready = os.path.isdir(DATA_DIR)


def test_split_is_deterministic_and_disjoint():
    labels = np.repeat(np.arange(data.NUM_CLASSES), 30)  # 300 samples, 30 per class
    tr1, va1, te1 = data.split_indices(labels, seed=0)
    tr2, va2, te2 = data.split_indices(labels, seed=0)
    # same seed -> identical indices
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2) and np.array_equal(te1, te2)
    # the three splits partition the dataset with no leakage
    allidx = np.concatenate([tr1, va1, te1])
    assert np.array_equal(np.sort(allidx), np.arange(labels.size))
    assert set(tr1) & set(va1) == set()
    assert set(tr1) & set(te1) == set()
    assert set(va1) & set(te1) == set()


def test_split_changes_with_seed():
    labels = np.repeat(np.arange(data.NUM_CLASSES), 30)
    tr1, _, _ = data.split_indices(labels, seed=0)
    tr2, _, _ = data.split_indices(labels, seed=1)
    assert not np.array_equal(tr1, tr2)


def test_split_is_stratified():
    labels = np.repeat(np.arange(data.NUM_CLASSES), 30)
    tr, va, te = data.split_indices(labels, seed=0, fracs=(0.7, 0.15, 0.15))
    # every class is represented in every split, in roughly the requested proportion
    for split, frac in ((tr, 0.7), (va, 0.15), (te, 0.15)):
        counts = np.bincount(labels[split], minlength=data.NUM_CLASSES)
        assert (counts > 0).all()
        assert abs(split.size - frac * labels.size) <= data.NUM_CLASSES


def test_tiny_subset_shapes():
    ds = data.tiny_subset(n=20, seed=0)
    assert len(ds) == 20
    x, y = ds[0]
    assert tuple(x.shape) == (3, 64, 64)
    assert 0 <= int(y) < data.NUM_CLASSES


def test_smallcnn_forward_shape():
    net = model.SmallCNN(num_classes=data.NUM_CLASSES)
    out = net(torch.randn(4, 3, 64, 64))
    assert out.shape == (4, data.NUM_CLASSES)


def test_smallcnn_param_budget():
    net = model.SmallCNN(num_classes=data.NUM_CLASSES)
    n = sum(p.numel() for p in net.parameters())
    assert 300_000 <= n <= 1_500_000, f"SmallCNN has {n} params, outside the intended budget"


def test_training_reduces_loss_on_random_data():
    torch.manual_seed(0)
    ds = data.tiny_subset(n=64, seed=0)
    loader = torch.utils.data.DataLoader(ds, batch_size=16, shuffle=True)
    net = model.SmallCNN(num_classes=data.NUM_CLASSES)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    first = train_torch.train_one_epoch(net, loader, opt, crit, device="cpu")
    last = train_torch.train_one_epoch(net, loader, opt, crit, device="cpu")
    assert last < first, f"loss did not decrease: {first:.3f} -> {last:.3f}"


def test_confusion_matrix_shape_and_total():
    ds = data.tiny_subset(n=24, seed=1)
    loader = torch.utils.data.DataLoader(ds, batch_size=8)
    net = model.SmallCNN(num_classes=data.NUM_CLASSES)
    cm = ev.confusion_matrix(net, loader, device="cpu", num_classes=data.NUM_CLASSES)
    assert cm.shape == (data.NUM_CLASSES, data.NUM_CLASSES)
    assert cm.sum() == 24


def test_plot_training_curves_writes_file(tmp_path):
    metrics = {"scratch": {
        "best_epoch": 2, "test_acc": 0.9,
        "history": [
            {"epoch": 1, "train_loss": 1.0, "val_loss": 1.1, "val_acc": 0.5},
            {"epoch": 2, "train_loss": 0.8, "val_loss": 0.9, "val_acc": 0.6},
        ],
        "per_class_acc": {f"c{i}": 0.9 for i in range(data.NUM_CLASSES)},
    }}
    out = tmp_path / "curves.png"
    viz.plot_training_curves(metrics, str(out))
    assert out.exists() and out.stat().st_size > 0


@pytest.mark.skipif(not data_ready, reason="EuroSAT dataset not downloaded")
def test_real_split_covers_dataset():
    labels = np.array(data.load_eurosat().targets)
    tr, va, te = data.split_indices(labels, seed=data.SEED)
    assert tr.size + va.size + te.size == labels.size
    assert tr.size > va.size and tr.size > te.size
