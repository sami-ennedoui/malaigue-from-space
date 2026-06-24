"""Sentinel-2 L2A access via the Planetary Computer STAC API."""
import pandas as pd
import planetary_computer
import pystac_client

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
