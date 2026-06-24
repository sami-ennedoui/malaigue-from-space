import geopandas as gpd
from shapely.geometry import Polygon
from malaigue import geo


def test_aoi_bbox_covers_thau():
    minx, miny, maxx, maxy = geo.aoi_bbox_4326()
    # Thau lagoon sits around 3.5-3.7 E, 43.36-43.46 N
    assert minx < 3.55 < maxx
    assert miny < 43.42 < maxy


def test_lagoon_polygon_inside_bbox():
    poly = geo.lagoon_polygon_4326()
    assert isinstance(poly, Polygon)
    minx, miny, maxx, maxy = geo.aoi_bbox_4326()
    assert poly.bounds[0] >= minx - 0.01 and poly.bounds[2] <= maxx + 0.01


def test_sectors_have_named_polygons():
    s = geo.sectors_4326()
    assert set(["bouzigues", "meze", "main_basin"]).issubset(set(s["name"]))
    assert s.crs.to_string() == "EPSG:4326"


def test_reproject_to_utm():
    s = geo.sectors_4326()
    out = geo.reproject_gdf(s, geo.SCENE_CRS)
    assert out.crs.to_string() == geo.SCENE_CRS
    # UTM 31N has central meridian 3E; Thau near 3.6E lands at ~554 km easting
    assert 545000 < out.geometry.iloc[0].centroid.x < 560000
