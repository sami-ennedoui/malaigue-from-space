"""Figures for the EuroSAT report.

Everything here reads from `outputs/eurosat/metrics.json` and the downloaded
dataset, so the figures regenerate from any run rather than being baked in.

    uv run python -m malaigue.eurosat.viz

Produces, in outputs/eurosat/:
- training_curves.png  train/val loss and validation accuracy across epochs
- per_class_acc.png    per-class test accuracy against the overall accuracy
- data_samples.png     one real chip per class, to show what the model sees
"""
import glob
import json
import os

import numpy as np

OUT_DIR = "outputs/eurosat"
METRICS = os.path.join(OUT_DIR, "metrics.json")
DATA_ROOT = "data/eurosat/eurosat/2750"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_training_curves(metrics, path=os.path.join(OUT_DIR, "training_curves.png")):
    """Loss and validation accuracy across epochs.

    The loss panel is the overfitting check: train and val loss should both
    fall; a val loss that turns up while train loss keeps falling is overfitting.
    The accuracy panel marks the epoch whose weights were kept.
    """
    plt = _plt()
    h = metrics["scratch"]["history"]
    ep = [r["epoch"] for r in h]
    best = metrics["scratch"]["best_epoch"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(ep, [r["train_loss"] for r in h], "-o", ms=3, label="train loss")
    ax1.plot(ep, [r["val_loss"] for r in h], "-o", ms=3, label="validation loss")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("cross-entropy loss")
    ax1.set_title("Training and validation loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    va = [r["val_acc"] for r in h]
    ax2.plot(ep, va, "-o", ms=3, color="green", label="validation accuracy")
    ax2.axvline(best, ls="--", color="gray", label=f"best epoch ({best})")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("accuracy")
    ax2.set_ylim(min(va) - 0.03, 1.0)
    ax2.set_title("Validation accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle("SmallCNN trained from scratch on EuroSAT")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_per_class(metrics, path=os.path.join(OUT_DIR, "per_class_acc.png")):
    """Per-class test accuracy, sorted, against the overall accuracy line."""
    plt = _plt()
    pc = metrics["scratch"]["per_class_acc"]
    order = sorted(pc, key=pc.get)
    vals = [pc[n] for n in order]
    test_acc = metrics["scratch"]["test_acc"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(order, vals, color="steelblue")
    ax.axvline(test_acc, color="crimson", ls="--", label=f"overall {test_acc:.3f}")
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("test accuracy (per-class recall)")
    ax.set_title("Per-class test accuracy, SmallCNN")
    ax.legend(loc="lower right")
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_data_samples(root=DATA_ROOT, path=os.path.join(OUT_DIR, "data_samples.png"),
                      seed=0):
    """A 2x5 grid of one real chip per class, to show the input the model sees."""
    plt = _plt()
    from PIL import Image

    classes = sorted(d for d in os.listdir(root)
                     if os.path.isdir(os.path.join(root, d)))
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(2, 5, figsize=(11, 5))
    for ax, cls in zip(axes.ravel(), classes):
        files = sorted(glob.glob(os.path.join(root, cls, "*.jpg")))
        img = Image.open(files[int(rng.integers(len(files)))])
        ax.imshow(img)
        ax.set_title(cls, fontsize=9)
        ax.axis("off")
    fig.suptitle("EuroSAT RGB, one 64x64 Sentinel-2 chip per class")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main():
    with open(METRICS) as f:
        metrics = json.load(f)
    p1 = plot_training_curves(metrics)
    p2 = plot_per_class(metrics)
    p3 = plot_data_samples()
    print("wrote:")
    for p in (p1, p2, p3):
        print(" ", p)


if __name__ == "__main__":
    main()
