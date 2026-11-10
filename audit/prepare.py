import argparse
import os
import sys
from io import BytesIO

import geopandas as gpd
import numpy as np
import requests
from PIL import Image
from rasterio.features import shapes
from rasterio.transform import from_origin
from shapely.geometry import shape
from shapely.ops import unary_union

from audit.common import load_config, output_path


def setup_django(back_dir):
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "iarbre_data.settings")
    os.environ.setdefault("TELESCOOP_DEV", "1")
    os.chdir(back_dir)
    sys.path.insert(0, back_dir)
    import django

    django.setup()


def to_gdf(rows, columns, crs_from, crs_to):
    *attrs, geoms = zip(*rows)
    data = dict(zip(columns, attrs))
    geometry = gpd.GeoSeries.from_wkb([bytes(g.wkb) for g in geoms], crs=crs_from)
    return gpd.GeoDataFrame(data, geometry=geometry).to_crs(crs_to)


def export_zones(cfg):
    cwd = os.getcwd()
    setup_django(cfg["paths"]["iarbre_back_dir"])
    from django.conf import settings
    from iarbre_data.models import Iris, Lcz

    os.chdir(cwd)
    srid = f"EPSG:{settings.SRID_DB}"
    iris = to_gdf(
        Iris.objects.order_by("id").values_list(
            "id", "code", "name", "city__code", "city__name", "geometry"
        ),
        ["iris_id", "code", "nom", "code_commune", "commune"],
        srid,
        cfg["crs"],
    )
    lcz = to_gdf(
        Lcz.objects.order_by("id").values_list(
            "lcz_index", "lcz_description", "geometry"
        ),
        ["lcz_index", "lcz_description"],
        srid,
        cfg["crs"],
    )
    zones = output_path(cfg, "zones.gpkg")
    iris.to_file(zones, layer="iris", driver="GPKG")
    lcz.to_file(zones, layer="lcz", driver="GPKG")
    print(f"{len(iris)} IRIS, {len(lcz)} LCZ -> {zones}")


def fetch_alpha(cfg, x0, y1, size_px):
    ex = cfg["exclusion"]
    res = ex["resolution"]
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": ex["wms_layer"],
        "CRS": cfg["crs"],
        "BBOX": f"{x0},{y1 - size_px * res},{x0 + size_px * res},{y1}",
        "WIDTH": size_px,
        "HEIGHT": size_px,
        "FORMAT": "image/png",
        "TRANSPARENT": "TRUE",
        "STYLES": "",
    }
    response = requests.get(ex["wms_url"], params=params, timeout=120)
    response.raise_for_status()
    if "image" not in response.headers.get("Content-Type", ""):
        raise RuntimeError(response.text[:500])
    return np.array(Image.open(BytesIO(response.content)).convert("RGBA"))[..., 3]


def trace_exclusion(cfg):
    ex = cfg["exclusion"]
    xmin, ymin, xmax, ymax = ex["search_bbox"]
    res, tile = ex["resolution"], ex["tile_px"]
    step = tile * res
    nx = int(np.ceil((xmax - xmin) / step))
    ny = int(np.ceil((ymax - ymin) / step))
    alpha = np.zeros((ny * tile, nx * tile), dtype=np.uint8)
    for j in range(ny):
        for i in range(nx):
            alpha[j * tile : (j + 1) * tile, i * tile : (i + 1) * tile] = fetch_alpha(
                cfg, xmin + i * step, ymax - j * step, tile
            )
    gap = alpha == 0
    border = np.concatenate([gap[0], gap[-1], gap[:, 0], gap[:, -1]]).mean()
    transform = from_origin(xmin, ymax, res, res)
    polygons = [
        shape(geom)
        for geom, _ in shapes(gap.astype(np.uint8), mask=gap, transform=transform)
    ]
    geometry = unary_union(polygons)
    gdf = gpd.GeoDataFrame(
        {"source": [ex["source"]], "surface_m2": [geometry.area]},
        geometry=[geometry],
        crs=cfg["crs"],
    )
    gdf.to_file(output_path(cfg, "exclusion.gpkg"), driver="GPKG")
    print(
        f"gap {geometry.area / 1e6:.3f} km2 in {len(polygons)} parts, "
        f"{border:.1%} of search bbox border is transparent"
    )


def main():
    parser = argparse.ArgumentParser(description="Zones IRIS/LCZ et zone exclue")
    parser.add_argument("--config", default="audit/config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    trace_exclusion(cfg)
    export_zones(cfg)


if __name__ == "__main__":
    main()
