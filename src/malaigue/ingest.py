"""Sentinel-2 L2A access via the Planetary Computer STAC API."""
import time

import odc.stac
import pandas as pd
import planetary_computer
import pystac_client
import rioxarray  # noqa: F401  registers the .rio accessor on xarray objects

# Clay v1.5 Sentinel-2 band order (10 bands, the 10 m and 20 m reflectance bands).
CLAY_S2_BANDS = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12"]

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"


def _client():
    return pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)


def _with_retry(fn, tries=5, delay=3):
    """Retry a network call a few times. The STAC API and the laptop sleeping cause
    transient timeouts on long runs."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001  transient network/API errors
            last = exc
            time.sleep(delay * (i + 1))
    raise last


def _search_items(bbox, start, end, max_cloud):
    def run():
        search = _client().search(
            collections=[COLLECTION], bbox=list(bbox),
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
        )
        return list(search.items())
    return _with_retry(run)


def list_clear_dates(bbox, start, end, max_cloud=20):
    """Low-cloud Sentinel-2 acquisitions over bbox, sorted by date ascending."""
    rows = []
    for it in _search_items(bbox, start, end, max_cloud):
        rows.append({
            "date": pd.to_datetime(it.properties["datetime"]).date(),
            "item_id": it.id,
            "cloud": float(it.properties["eo:cloud_cover"]),
        })
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def _signed_item(item_id):
    def run():
        search = _client().search(collections=[COLLECTION], ids=[item_id])
        return next(search.items())
    return _with_retry(run)


def load_scene(item_id, bbox, bands=None, resolution=10):
    """Load a band stack clipped to bbox (EPSG:4326) as an xarray.Dataset in UTM."""
    bands = bands or CLAY_S2_BANDS
    item = _signed_item(item_id)
    ds = odc.stac.load(
        [item], bands=bands, bbox=list(bbox),
        resolution=resolution, chunks={},
    )
    return ds.isel(time=0) if "time" in ds.dims else ds
