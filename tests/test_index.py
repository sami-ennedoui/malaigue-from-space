import numpy as np
import xarray as xr
from malaigue import index


def _toy_ds():
    red = xr.DataArray(np.array([[0.10, 0.10], [0.20, 0.20]]), dims=("y", "x"))
    rededge1 = xr.DataArray(np.array([[0.30, 0.30], [0.20, 0.20]]), dims=("y", "x"))
    return xr.Dataset({"red": red, "rededge1": rededge1})


def test_ndci_formula():
    ds = _toy_ds()
    out = index.ndci(ds)
    # (0.30-0.10)/(0.30+0.10) = 0.5 ; (0.20-0.20)/0.40 = 0.0
    assert np.isclose(out.values[0, 0], 0.5)
    assert np.isclose(out.values[1, 0], 0.0)


def test_apply_water_mask_sets_land_nan():
    ds = _toy_ds()
    nd = index.ndci(ds)
    mask = np.array([[True, False], [True, False]])
    masked = index.apply_water_mask(nd, mask)
    assert np.isnan(masked.values[0, 1])
    assert not np.isnan(masked.values[0, 0])
