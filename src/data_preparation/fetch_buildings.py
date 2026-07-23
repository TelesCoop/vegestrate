import argparse
from io import BytesIO
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds

BDTOPO_URL = "https://data.geopf.fr/wfs/ows"
TYPENAME = "BDTOPO_V3:batiment"
WFS_CRS = "EPSG:2154" 
PAGE_SIZE = 5000
DATE_COLUMNS = ("date_d_apparition", "date_dapparition", "date_de_confirmation")


def raster_bbox(raster_path, crs_override=None, buffer=0.0):
    """Bounding box of a raster, buffered by `buffer` metres."""
    with rasterio.open(raster_path) as src:
        left, bottom, right, top = src.bounds
        crs = src.crs

    if crs_override:
        crs = rasterio.crs.CRS.from_user_input(crs_override)
    elif crs is None or crs.to_epsg() is None:
        raise ValueError(
            f"{raster_path} has no usable CRS ({crs}). Pass --crs, e.g. --crs EPSG:3946."
        )

    bbox = (left - buffer, bottom - buffer, right + buffer, top + buffer)
    return bbox, crs


def _filter_by_date(gdf, before):
    """Keep buildings that already existed at `before` (unknown dates are kept)."""
    by_name = {c.lower(): c for c in gdf.columns}
    column = next((by_name[name] for name in DATE_COLUMNS if name in by_name), None)
    if column is None:
        print(f"  ⚠ No date column in {sorted(gdf.columns)}; keeping all buildings")
        return gdf

    dates = pd.to_datetime(gdf[column], errors="coerce", utc=True)
    cutoff = pd.Timestamp(before, tz="UTC")
    keep = dates.isna() | (dates <= cutoff)
    print(
        f"  Date filter on '{column}' <= {before}: {int(keep.sum())} kept "
        f"({int((~keep).sum())} newer, {int(dates.isna().sum())} undated kept)"
    )
    return gdf[keep]


def download_buildings(
    bbox,
    bbox_crs,
    output_path,
    before=None,
    url=BDTOPO_URL,
    typename=TYPENAME,
    timeout=120,
):
    """Download BD TOPO building footprints covering `bbox` and save them in `bbox_crs`."""
    x1, y1, x2, y2 = transform_bounds(bbox_crs, WFS_CRS, *bbox)
    print(f"Querying {typename}")
    print(f"  BBOX ({WFS_CRS}): {x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}")

    pages = []
    start = 0
    while True:
        params = {
            "SERVICE": "WFS",
            "VERSION": "2.0.0",
            "REQUEST": "GetFeature",
            "TYPENAMES": typename,
            "SRSNAME": WFS_CRS,
            "OUTPUTFORMAT": "text/xml; subtype=gml/3.2",
            "BBOX": f"{x1},{y1},{x2},{y2},{WFS_CRS}",
            "COUNT": PAGE_SIZE,
            "STARTINDEX": start,
        }
        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()

        page = gpd.read_file(BytesIO(resp.content))
        if page.empty:
            break
        pages.append(page)
        start += len(page)
        print(f"\r  {start} buildings downloaded", end="", flush=True)
        if len(page) < PAGE_SIZE:
            break
    print()

    if not pages:
        print("✗ No buildings returned for this bbox")
        return None

    gdf = pd.concat(pages, ignore_index=True)
    gdf = gpd.GeoDataFrame(gdf, geometry="geometry").set_crs(
        WFS_CRS, allow_override=True
    )

    if before:
        gdf = _filter_by_date(gdf, before)

    # Only the footprints matter for masking; dropping attributes keeps the file small.
    gdf = gdf[["geometry"]].to_crs(bbox_crs)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(output_path)
    print(f"✓ {len(gdf)} buildings saved to {output_path} ({bbox_crs})")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Download BD TOPO building footprints covering a raster, "
        "ready to pass to clean_raster.py --buildings",
    )
    parser.add_argument(
        "-r", "--raster", required=True, help="Raster whose extent should be covered"
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="Output vector file; .gpkg is far smaller than .geojson for large areas",
    )
    parser.add_argument(
        "--crs",
        default=None,
        help="Override the raster CRS, e.g. EPSG:3946 (needed when the raster carries "
        "a LOCAL_CS with no EPSG code)",
    )
    parser.add_argument(
        "--buffer",
        type=float,
        default=50.0,
        metavar="M",
        help="Extend the query box by M metres (default: 50) so edge buildings are complete",
    )
    parser.add_argument(
        "--before",
        default=None,
        metavar="YYYY-MM-DD",
        help="Keep only buildings that appeared before this date, for diachronic runs. "
        "Buildings with no recorded date are kept.",
    )
    parser.add_argument("--url", default=BDTOPO_URL, help=f"WFS URL (default: {BDTOPO_URL})")
    parser.add_argument("--typename", default=TYPENAME, help=f"Layer (default: {TYPENAME})")
    args = parser.parse_args()

    if Path(args.output).exists():
        print(f"✓ Building footprints already exist: {args.output}")
        return

    bbox, crs = raster_bbox(args.raster, args.crs, args.buffer)
    print(f"Raster: {args.raster}")
    print(f"  CRS: {crs}")
    print(f"  BBOX: {bbox}")
    download_buildings(
        bbox,
        crs,
        args.output,
        before=args.before,
        url=args.url,
        typename=args.typename,
    )


if __name__ == "__main__":
    main()
