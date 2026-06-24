"""Clay v1.5 as a frozen Sentinel-2 feature extractor.

Confirmed against the v1.5 checkpoint (Task 8 spike):
- model_size "large", patch_size 8, embedding dim 1024.
- A 256x256 chip yields a 32x32 patch grid (~80 m cells at 10 m).
- encoder.forward(datacube) returns a tuple; out[0] is (1, 1+1024, 1024) (class token + patches),
  out[1] is the (1, 1024) mean-pooled chip embedding.
- We load only the model.encoder.* weights (the teacher and projection head are training-only).
- Band order, means, stds and wavelengths come from Clay's metadata.yaml (sentinel-2-l2a), which
  lines up one-to-one with ingest.CLAY_S2_BANDS.
"""
import datetime as dt
import numpy as np
import torch
import yaml

EMBED_DIM = 1024
CHIP_PX = 256
PATCH_PX = 8


def _s2_stats(metadata_path):
    md = yaml.safe_load(open(metadata_path))
    s2 = md["sentinel-2-l2a"]
    order = s2["band_order"]
    mean = np.array([s2["bands"]["mean"][b] for b in order], dtype="float32")
    std = np.array([s2["bands"]["std"][b] for b in order], dtype="float32")
    waves = [float(s2["bands"]["wavelength"][b]) for b in order]
    return mean, std, waves


class ClayExtractor:
    """Holds the frozen Clay encoder and the normalization it needs."""

    def __init__(self, encoder, mean, std, waves, device):
        self.encoder = encoder
        self.mean = mean
        self.std = std
        self.waves = waves
        self.device = device


def load_clay(ckpt, metadata_path, device="cpu"):
    """Load Clay v1.5, keeping only the encoder, as a frozen feature extractor."""
    from claymodel.module import ClayMAEModule
    module = ClayMAEModule(
        model_size="large", mask_ratio=0.0, patch_size=PATCH_PX,
        shuffle=False, metadata_path=metadata_path,
    )
    full = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)["state_dict"]
    enc_sd = {k[len("model.encoder."):]: v for k, v in full.items() if k.startswith("model.encoder.")}
    module.model.encoder.load_state_dict(enc_sd, strict=False)
    module.eval().to(device)
    mean, std, waves = _s2_stats(metadata_path)
    return ClayExtractor(module.model.encoder, mean, std, waves, device)


def normalize_chip(stack, mean, std):
    """(x - mean) / std per band. stack is (bands, H, W)."""
    return (stack - mean[:, None, None]) / std[:, None, None]


def _datacube(model, chip, date, latlon):
    norm = normalize_chip(chip.astype("float32"), model.mean, model.std)
    pixels = torch.from_numpy(norm).unsqueeze(0).to(model.device)
    d = dt.date.fromisoformat(str(date)[:10])
    week = d.isocalendar().week
    hour = 12  # Sentinel-2 over France passes mid-morning; a fixed midday is fine
    time = torch.tensor(
        [[np.sin(2 * np.pi * week / 52), np.cos(2 * np.pi * week / 52),
          np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)]],
        dtype=torch.float32, device=model.device)
    lat, lon = latlon
    latlon_t = torch.tensor(
        [[np.sin(np.deg2rad(lat)), np.cos(np.deg2rad(lat)),
          np.sin(np.deg2rad(lon)), np.cos(np.deg2rad(lon))]],
        dtype=torch.float32, device=model.device)
    waves = torch.tensor(model.waves, dtype=torch.float32, device=model.device)
    gsd = torch.tensor(10.0, device=model.device)
    return {"pixels": pixels, "time": time, "latlon": latlon_t, "waves": waves, "gsd": gsd}


def patch_embeddings(model, chip, date, latlon):
    """Per-patch embeddings reshaped to (Hp, Wp, 1024); drops the class token."""
    cube = _datacube(model, chip, date, latlon)
    with torch.no_grad():
        out = model.encoder(cube)
    patches = out[0][0, 1:, :]  # drop the leading class token -> (num_patches, 1024)
    side = int(round(patches.shape[0] ** 0.5))
    return patches[: side * side].reshape(side, side, EMBED_DIM).cpu().numpy()


def chip_embedding(model, chip, date, latlon):
    """Single 1024-d mean-pooled embedding for a chip."""
    cube = _datacube(model, chip, date, latlon)
    with torch.no_grad():
        out = model.encoder(cube)
    return out[1][0].cpu().numpy()


def lagoon_embedding(model, ds, water_mask, date, latlon):
    """One 1024-d embedding for the lagoon: mean of patch embeddings over water.
    Centers one chip on the lagoon and pools the patches whose footprint is water."""
    from malaigue import ingest
    stack = np.stack([ds[b].values for b in ingest.CLAY_S2_BANDS]).astype("float32")
    stack = _center_crop(stack, CHIP_PX)
    mask = _center_crop(water_mask[None].astype("float32"), CHIP_PX)[0] > 0.5
    pe = patch_embeddings(model, stack, date, latlon)
    pm = _downsample_mask(mask, pe.shape[:2])
    sel = pe[pm]
    if sel.shape[0] == 0:
        return pe.reshape(-1, EMBED_DIM).mean(0)
    return sel.mean(0)


def _center_crop(arr, size):
    _, h, w = arr.shape
    top = max((h - size) // 2, 0)
    left = max((w - size) // 2, 0)
    out = arr[:, top:top + size, left:left + size]
    if out.shape[1] < size or out.shape[2] < size:
        out = np.pad(out, ((0, 0), (0, size - out.shape[1]), (0, size - out.shape[2])))
    return out


def _downsample_mask(mask, target_hw):
    th, tw = target_hw
    fh, fw = mask.shape[0] // th, mask.shape[1] // tw
    if fh == 0 or fw == 0:
        return np.ones(target_hw, dtype=bool)
    return mask[: th * fh, : tw * fw].reshape(th, fh, tw, fw).mean((1, 3)) > 0.5
