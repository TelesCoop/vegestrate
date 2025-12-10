from io import BytesIO

import rasterio
import requests

from core.common_ortho import resize_and_save


def fetch_wms_for_raster(
    raster_path,
    output_path,
    wms_url,
    layer_name,
    resolution=0.8,
    image_format="image/jpeg",
):
    """
    Fetch WMS imagery matching the bounds of an existing raster file.

    Args:
        raster_path: Path to the input raster file
        output_path: Path where to save the WMS image
        wms_url: Base WMS service URL
        layer_name: WMS layer name to request
        resolution: resolution ot transform the TIF to (in meter)
        image_format: Image format (image/jpeg or image/png)
    """

    with rasterio.open(raster_path) as src:
        bounds = src.bounds
        crs = src.crs
        width = src.width
        height = src.height
        transform = src.transform

        current_res_x = (bounds.right - bounds.left) / width
        current_res_y = (bounds.top - bounds.bottom) / height

        print("Raster info:")
        print(f"  Bounds: {bounds}")
        print(f"  CRS: {crs}")
        print(f"  Size: {width}x{height}")
        print(f"  Transform: {transform}")
        print(f"  Current resolution: {current_res_x:.3f}m x {current_res_y:.3f}m")

    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer_name,
        "CRS": str(crs),
        "BBOX": f"{bounds.left},{bounds.bottom},{bounds.right},{bounds.top}",
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": image_format,
        "STYLES": "",
    }

    print("\nFetching WMS data...")
    print(f"  URL: {wms_url}")
    print(f"  Layer: {layer_name}")
    print(f"  BBOX: {params['BBOX']}")

    response = requests.get(wms_url, params=params, timeout=60)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "image" not in content_type:
        print(f"Error: Expected image, got {content_type}")
        print(f"Response: {response.text[:500]}")
        return
    resize_and_save(
        raster_path=BytesIO(response.content),
        resolution=resolution,
        bounds=bounds,
        crs=crs,
        output_path=output_path,
    )


def main():
    RASTER_PATH = "classification_map.tif"
    OUTPUT_PATH = "orthophoto_bdortho.tif"
    WMS_URL = "https://data.geopf.fr/wms-r"
    LAYER_NAME = "HR.ORTHOIMAGERY.ORTHOPHOTOS"  # BD ORTHO layer

    print("=== WMS Fetch from Raster Bounds ===\n")

    fetch_wms_for_raster(
        raster_path=RASTER_PATH,
        output_path=OUTPUT_PATH,
        wms_url=WMS_URL,
        layer_name=LAYER_NAME,
        image_format="image/jpeg",
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
