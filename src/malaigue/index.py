"""Physical water-quality proxies from Sentinel-2 surface reflectance."""
import numpy as np
import pandas as pd
import xarray as xr


def ndci(ds):
    """Normalized Difference Chlorophyll Index = (B05 - B04) / (B05 + B04)."""
    b04, b05 = ds["B04"], ds["B05"]
    return (b05 - b04) / (b05 + b04)


def turbidity_red(ds):
    """Simple turbidity proxy: red-band (B04) surface reflectance."""
    return ds["B04"]


def apply_water_mask(da, mask):
    """Set non-water pixels to NaN. mask is a boolean array, True over water."""
    if not isinstance(mask, xr.DataArray):
        mask = xr.DataArray(mask, dims=da.dims)
    return da.where(mask)


def sector_stats(da, sectors_gdf):
    """Per-sector summary statistics of a masked DataArray.
    sectors_gdf must be in the same CRS as da and carry a 'name' column."""
    from malaigue import geo
    transform = da.rio.transform()
    shape = (da.rio.height, da.rio.width)
    rows = []
    for _, row in sectors_gdf.iterrows():
        m = geo.rasterize_mask(row.geometry, transform, shape, da.rio.crs)
        vals = da.values[m]
        vals = vals[~np.isnan(vals)]
        rows.append({
            "name": row["name"],
            "mean": float(np.mean(vals)) if vals.size else np.nan,
            "median": float(np.median(vals)) if vals.size else np.nan,
            "p90": float(np.percentile(vals, 90)) if vals.size else np.nan,
            "count": int(vals.size),
        })
    return pd.DataFrame(rows)
