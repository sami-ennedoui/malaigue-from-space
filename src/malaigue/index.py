"""Physical water-quality proxies from Sentinel-2 surface reflectance."""
import numpy as np
import pandas as pd
import xarray as xr


def ndci(ds):
    """Normalized Difference Chlorophyll Index = (rededge1 - red) / (rededge1 + red)."""
    red, rededge1 = ds["red"], ds["rededge1"]
    return (rededge1 - red) / (rededge1 + red)


def turbidity_red(ds):
    """Simple turbidity proxy: red-band surface reflectance."""
    return ds["red"]


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
