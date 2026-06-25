"""Geospatial primitives for the Thau lagoon AOI."""
import geopandas as gpd
from rasterio.features import rasterize
from shapely.geometry import Polygon

SCENE_CRS = "EPSG:32631"  # UTM zone 31N, the Sentinel-2 tile CRS over Thau


def aoi_bbox_4326():
    """(minx, miny, maxx, maxy) bounding box around the etang de Thau in EPSG:4326."""
    return (3.52, 43.35, 3.73, 43.47)


def lagoon_polygon_4326():
    """Approximate outline of the lagoon water body. Refine from OSM at build time
    (Overpass: relation 'Etang de Thau'); these vertices are a usable coarse start."""
    return Polygon([
        (3.535, 43.385), (3.565, 43.435), (3.61, 43.455),
        (3.67, 43.445), (3.705, 43.415), (3.66, 43.375),
        (3.60, 43.365), (3.555, 43.37), (3.535, 43.385),
    ])


def sectors_4326():
    """Named sub-sectors used for validation overlays. Coarse polygons: the two
    crisis sectors (Bouzigues NE shore, Meze NW shore) plus the open main basin."""
    bouzigues = Polygon([(3.64, 43.44), (3.675, 43.45), (3.69, 43.435),
                         (3.66, 43.425), (3.64, 43.44)])
    meze = Polygon([(3.58, 43.43), (3.61, 43.45), (3.625, 43.435),
                    (3.595, 43.418), (3.58, 43.43)])
    main_basin = Polygon([(3.55, 43.39), (3.60, 43.43), (3.655, 43.42),
                          (3.63, 43.385), (3.575, 43.378), (3.55, 43.39)])
    return gpd.GeoDataFrame(
        {"name": ["bouzigues", "meze", "main_basin"]},
        geometry=[bouzigues, meze, main_basin],
        crs="EPSG:4326",
    )


def reproject_gdf(gdf, dst_crs):
    """Reproject a GeoDataFrame to dst_crs."""
    return gdf.to_crs(dst_crs)


def rasterize_mask(geom, transform, shape, crs):
    """Boolean mask (True inside geom) on a raster grid given by transform+shape.
    geom is assumed already in the raster CRS; crs is kept for interface clarity."""
    arr = rasterize(
        [(geom, 1)], out_shape=shape, transform=transform,
        fill=0, dtype="uint8", all_touched=False,
    )
    return arr.astype(bool)


def chip_windows(bounds, chip_px=256, res=10):
    """Non-overlapping (col_off, row_off, width, height) windows tiling bounds
    with chip_px square chips at the given resolution in metres."""
    minx, miny, maxx, maxy = bounds
    n_cols = int((maxx - minx) // (chip_px * res))
    n_rows = int((maxy - miny) // (chip_px * res))
    windows = []
    for r in range(max(n_rows, 1)):
        for c in range(max(n_cols, 1)):
            windows.append((c * chip_px, r * chip_px, chip_px, chip_px))
    return windows
