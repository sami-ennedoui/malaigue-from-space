import importlib.util
import os

import numpy as np
import pytest

from malaigue import embed

CKPT = "data/clay/v1.5/clay-v1.5.ckpt"
META = "data/clay/metadata.yaml"
clay_ready = importlib.util.find_spec("claymodel") is not None and os.path.exists(CKPT)


def test_normalize_chip_centers_values():
    stack = np.ones((10, 8, 8), dtype="float32") * 0.2
    mean = np.full(10, 0.2, dtype="float32")
    std = np.full(10, 0.1, dtype="float32")
    out = embed.normalize_chip(stack, mean, std)
    assert np.allclose(out, 0.0)
    assert out.shape == (10, 8, 8)


@pytest.mark.skipif(not clay_ready, reason="claymodel or checkpoint not available")
def test_patch_embeddings_shape():
    model = embed.load_clay(CKPT, META, device="cpu")
    chip = (np.random.rand(10, 256, 256) * 3000).astype("float32")
    pe = embed.patch_embeddings(model, chip, date="2018-07-07", latlon=(43.42, 3.62))
    assert pe.ndim == 3 and pe.shape[2] == 1024
    assert pe.shape[0] == pe.shape[1] == 32  # 256 / patch_size(8)
