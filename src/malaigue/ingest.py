"""Sentinel-2 L2A access via the Planetary Computer STAC API."""
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


def _search_items(bbox, start, end, max_cloud):
    search = _client().search(
        collections=[COLLECTION], bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud}},
    )
    return list(search.items())


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
    search = _client().search(collections=[COLLECTION], ids=[item_id])
    return next(search.items())


def load_scene(item_id, bbox, bands=None, resolution=10):
    """Load a band stack clipped to bbox (EPSG:4326) as an xarray.Dataset in UTM."""
    bands = bands or CLAY_S2_BANDS
    item = _signed_item(item_id)
    ds = odc.stac.load(
        [item], bands=bands, bbox=list(bbox),
        resolution=resolution, chunks={},
    )
    return ds.isel(time=0) if "time" in ds.dims else ds
