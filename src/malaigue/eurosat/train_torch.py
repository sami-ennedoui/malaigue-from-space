"""Training entry points for the from-scratch CNN and the frozen-feature probe.

Run for real (never from pytest):

    uv run python -m malaigue.eurosat.train_torch --model scratch --epochs 20
    uv run python -m malaigue.eurosat.train_torch --model probe

`--model scratch` trains `SmallCNN` from scratch and saves the best-on-validation
checkpoint plus the training history. It never touches the test split; that is
left to `eval.py`, so the test set is seen exactly once.

`--model probe` is the transfer baseline: it extracts frozen ResNet18 features in
one pass and trains a linear head on them. It reports its own test accuracy
inline because it exists only to provide a comparison number; it is not "training
a CNN".
"""
import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

from malaigue.eurosat import data, model as M

OUT_DIR = "outputs/eurosat"
METRICS = os.path.join(OUT_DIR, "metrics.json")
SCRATCH_CKPT = os.path.join(OUT_DIR, "smallcnn.pt")


def train_one_epoch(model, loader, optimizer, criterion, device):
    """One pass over `loader` with gradient updates. Returns mean train loss."""
    model.train()
    total, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        optimizer.step()
        total += loss.item() * x.size(0)
        n += x.size(0)
    return total / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Mean loss and accuracy over `loader`, no gradient updates."""
    model.eval()
    total, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        total += criterion(out, y).item() * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        n += x.size(0)
    return total / n, correct / n


def update_metrics(key, value, path=METRICS):
    """Merge one top-level key into the metrics JSON, creating it if needed."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {}
    if os.path.exists(path):
        with open(path) as f:
            blob = json.load(f)
    blob[key] = value
    with open(path, "w") as f:
        json.dump(blob, f, indent=2)
    return blob


def train_scratch(epochs=20, batch_size=64, seed=data.SEED, lr=1e-3,
                  weight_decay=1e-4, root=data.ROOT, download=False, device="cpu"):
    """Train SmallCNN from scratch, keep the best-on-validation checkpoint."""
    torch.manual_seed(seed)
    train_loader, val_loader, _, classes, stats = data.make_loaders(
        root=root, seed=seed, batch_size=batch_size, download=download)
    net = M.SmallCNN(num_classes=len(classes)).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.CrossEntropyLoss()

    history, best_acc, best_epoch = [], 0.0, 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tl = train_one_epoch(net, train_loader, opt, crit, device)
        vl, va = evaluate(net, val_loader, crit, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": round(tl, 4),
                        "val_loss": round(vl, 4), "val_acc": round(va, 4),
                        "seconds": round(dt, 1)})
        print(f"epoch {epoch:2d}/{epochs}  train_loss={tl:.4f}  "
              f"val_loss={vl:.4f}  val_acc={va:.4f}  ({dt:.0f}s)")
        if va > best_acc:
            best_acc, best_epoch = va, epoch
            torch.save({"state_dict": net.state_dict(), "classes": classes,
                        "stats": stats, "seed": seed, "num_classes": len(classes)},
                       SCRATCH_CKPT)

    section = {
        "model": "SmallCNN (from scratch)",
        "params": int(n_params),
        "epochs_run": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "optimizer": "Adam",
        "augmentation": "random h/v flip",
        "best_val_acc": round(best_acc, 4),
        "best_epoch": best_epoch,
        "epoch_seconds_mean": round(float(np.mean([h["seconds"] for h in history])), 1),
        "history": history,
        "checkpoint": SCRATCH_CKPT,
    }
    update_metrics("dataset", {"seed": seed, "classes": classes, **stats})
    update_metrics("scratch", section)
    print(f"\nbest val acc {best_acc:.4f} at epoch {best_epoch}; "
          f"checkpoint -> {SCRATCH_CKPT}")
    print("now run eval to get honest test metrics + confusion matrix:")
    print("  uv run python -m malaigue.eurosat.eval")
    return section


def _extract_features(backbone, loader, device):
    """Run the frozen backbone over a loader, returning (features, labels)."""
    feats, labels = [], []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            feats.append(backbone(x.to(device)).cpu())
            labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_probe(batch_size=64, seed=data.SEED, lr=1e-3, head_epochs=30,
                root=data.ROOT, download=False, device="cpu"):
    """Transfer baseline: frozen ResNet18 features + a trained linear head.

    Extracts features once for all three splits, then trains a linear classifier
    on the train features with validation early-stopping, and reports test
    accuracy. This is a probe, not a trained CNN.
    """
    torch.manual_seed(seed)
    backbone, feat_dim, preprocess = M.pretrained_backbone()
    backbone.to(device)

    base = data.load_eurosat(root=root, download=download)
    labels = np.array(base.targets)
    classes = base.classes
    tr_idx, va_idx, te_idx = data.split_indices(labels, seed=seed)
    ds = data.load_eurosat(root=root, transform=preprocess)

    def loader(idx):
        return DataLoader(Subset(ds, list(idx)), batch_size=batch_size,
                          shuffle=False, num_workers=2)

    print("extracting frozen ResNet18 features (one pass over the dataset)...")
    t0 = time.time()
    xtr, ytr = _extract_features(backbone, loader(tr_idx), device)
    xva, yva = _extract_features(backbone, loader(va_idx), device)
    xte, yte = _extract_features(backbone, loader(te_idx), device)
    feat_seconds = time.time() - t0
    print(f"feature pass done in {feat_seconds:.0f}s")

    head = nn.Linear(feat_dim, len(classes)).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.CrossEntropyLoss()
    tr_loader = DataLoader(TensorDataset(xtr, ytr), batch_size=256, shuffle=True)
    va_loader = DataLoader(TensorDataset(xva, yva), batch_size=512)

    best_acc, best_state = 0.0, None
    for epoch in range(1, head_epochs + 1):
        train_one_epoch(head, tr_loader, opt, crit, device)
        _, va = evaluate(head, va_loader, crit, device)
        if va > best_acc:
            best_acc = va
            best_state = {k: v.clone() for k, v in head.state_dict().items()}
    head.load_state_dict(best_state)

    # the probe's test pass: cheap, on cached features
    te_loader = DataLoader(TensorDataset(xte, yte), batch_size=512)
    _, test_acc = evaluate(head, te_loader, crit, device)
    preds = head(xte.to(device)).argmax(1).cpu().numpy()
    per_class = _per_class_acc(yte.numpy(), preds, classes)

    section = {
        "model": "ResNet18 (ImageNet) frozen + linear probe",
        "note": "baseline for comparison, NOT a trained CNN",
        "feature_dim": feat_dim,
        "head_epochs": head_epochs,
        "feature_pass_seconds": round(feat_seconds, 1),
        "best_val_acc": round(best_acc, 4),
        "test_acc": round(test_acc, 4),
        "per_class_acc": per_class,
    }
    update_metrics("probe", section)
    print(f"\nprobe best val acc {best_acc:.4f} | test acc {test_acc:.4f}")
    return section


def _per_class_acc(y_true, y_pred, classes):
    out = {}
    for c, name in enumerate(classes):
        m = y_true == c
        out[name] = round(float((y_pred[m] == c).mean()), 4) if m.any() else None
    return out


def main():
    ap = argparse.ArgumentParser(description="Train EuroSAT models")
    ap.add_argument("--model", choices=["scratch", "probe"], default="scratch")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=data.SEED)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--download", action="store_true",
                    help="download EuroSAT if not present")
    args = ap.parse_args()
    if args.model == "scratch":
        train_scratch(epochs=args.epochs, batch_size=args.batch_size,
                      seed=args.seed, lr=args.lr, download=args.download)
    else:
        train_probe(batch_size=args.batch_size, seed=args.seed, lr=args.lr,
                    download=args.download)


if __name__ == "__main__":
    main()
