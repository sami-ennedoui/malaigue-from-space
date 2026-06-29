"""Honest test-set evaluation of the from-scratch SmallCNN.

Loads the best-on-validation checkpoint and runs the test split exactly once,
producing test accuracy, per-class accuracy, and a confusion-matrix figure. The
confusion matrix is where the real, spectrally close confusions show up
(Highway/River, AnnualCrop/PermanentCrop, Pasture/HerbaceousVegetation).

    uv run python -m malaigue.eurosat.eval
"""
import os

import numpy as np
import torch

from malaigue.eurosat import data, model as M
from malaigue.eurosat.train_torch import SCRATCH_CKPT, update_metrics

CONFUSION_PNG = "outputs/eurosat/confusion_matrix.png"


@torch.no_grad()
def confusion_matrix(model, loader, device, num_classes):
    """Integer confusion matrix; rows are true classes, columns predictions."""
    model.eval()
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for x, y in loader:
        pred = model(x.to(device)).argmax(1).cpu().numpy()
        for t, p in zip(y.numpy(), pred):
            cm[t, p] += 1
    return cm


def per_class_accuracy(cm, classes):
    """Recall per class: correctly predicted / true count, as a dict."""
    out = {}
    totals = cm.sum(axis=1)
    for c, name in enumerate(classes):
        out[name] = round(float(cm[c, c] / totals[c]), 4) if totals[c] else None
    return out


def plot_confusion(cm, classes, path=CONFUSION_PNG):
    """Row-normalized confusion heatmap with raw counts annotated."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)))
    ax.set_yticks(range(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(classes, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("EuroSAT SmallCNN confusion matrix (test split)")
    for i in range(len(classes)):
        for j in range(len(classes)):
            if cm[i, j]:
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="row-normalized")
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def main(root=data.ROOT, device="cpu"):
    if not os.path.exists(SCRATCH_CKPT):
        raise SystemExit(f"no checkpoint at {SCRATCH_CKPT}; run train_torch first")
    ckpt = torch.load(SCRATCH_CKPT, map_location=device)
    classes = ckpt["classes"]
    net = M.SmallCNN(num_classes=ckpt["num_classes"]).to(device)
    net.load_state_dict(ckpt["state_dict"])

    _, _, test_loader, _, _ = data.make_loaders(
        root=root, seed=ckpt["seed"], batch_size=128, augment=False)
    cm = confusion_matrix(net, test_loader, device, len(classes))
    test_acc = float(np.trace(cm) / cm.sum())
    per_class = per_class_accuracy(cm, classes)
    path = plot_confusion(cm, classes)

    # merge test metrics into the scratch section written by train_torch
    import json
    with open("outputs/eurosat/metrics.json") as f:
        blob = json.load(f)
    section = blob.get("scratch", {})
    section["test_acc"] = round(test_acc, 4)
    section["per_class_acc"] = per_class
    section["confusion_png"] = path
    section["confusion_matrix"] = cm.tolist()
    update_metrics("scratch", section)

    print(f"test accuracy: {test_acc:.4f}")
    print("per-class accuracy:")
    for name, acc in per_class.items():
        print(f"  {name:22s} {acc}")
    print(f"confusion matrix figure -> {path}")
    return test_acc, per_class


if __name__ == "__main__":
    main()
