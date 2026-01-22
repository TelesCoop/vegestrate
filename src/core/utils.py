import numpy as np
import rasterio
import requests
from rasterio import transform

# For IGN the mapping is 0-255
CLASS_LABELS = {0: "ground", 63: "grass", 127: "hedge", 191: "trees", 255: "else"}

# Mapping from CLASS_LABELS pixel values to SIMPLIFIED_CLASSES keys
# ground and else both map to 0 (else)
CLASS_TO_SIMPLIFIED = {
    0: 0,  # ground → else
    63: 1,  # grass → herbaceous
    127: 2,  # hedge → hedge
    191: 3,  # trees → trees
    255: 0,  # else → else
}


def download_file(url, filename):
    """Download a file from a URL"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        if not filename:
            filename = url.split("/")[-1]

        with open(filename, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"Successfully downloaded {filename}")
        return filename

    except requests.exceptions.RequestException as e:
        raise requests.exceptions.RequestException(f"Error downloading file: {e}")


def classification_to_raster(filtered_las, las, cell_size=0.2):
    """Convert LAS point cloud classification to raster with simplified classes

    Args:
        filtered_las: laspy LAS object containing point cloud data only for vegetation
        las: laspy LAS object containing point cloud data

        cell_size: size of raster cells in meters (default 0.2m)

    Returns:
        raster: 2D numpy array with simplified classification values (uint8)
                Values: 0=else, 1=herbaceous, 2=hedge, 3=trees
        affine_transform: rasterio affine transform for georeferencing
        crs: coordinate reference system from the LAS file
    """
    x_coords = las.x
    y_coords = las.y

    x_coords_filt = filtered_las.x
    y_coords_filt = filtered_las.y
    classifications = np.asarray(filtered_las.classification)

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    cols = int(np.ceil((x_max - x_min) / cell_size))
    rows = int(np.ceil((y_max - y_min) / cell_size))

    col_indices = ((x_coords_filt - x_min) / cell_size).astype(np.int32)
    row_indices = ((y_max - y_coords_filt) / cell_size).astype(np.int32)

    col_indices = np.clip(col_indices, 0, cols - 1)
    row_indices = np.clip(row_indices, 0, rows - 1)

    las_to_simplified_lut = np.zeros(256, dtype=np.uint8)
    las_to_simplified_lut[2] = 0
    las_to_simplified_lut[3] = 1
    las_to_simplified_lut[4] = 2
    las_to_simplified_lut[5] = 3
    las_to_simplified_lut[8] = 3

    simplified_classes = las_to_simplified_lut[classifications]

    num_classes = 4
    counts = np.zeros((num_classes, rows, cols), dtype=np.int32)
    np.add.at(counts, (simplified_classes, row_indices, col_indices), 1)

    raster = np.argmax(counts, axis=0).astype(np.uint8)

    affine_transform = transform.from_bounds(x_min, y_min, x_max, y_max, cols, rows)

    crs = las.header.parse_crs() if hasattr(las.header, "parse_crs") else None

    return raster, affine_transform, crs


def export_raster(data, filename, transform, crs=None):
    """Export numpy array as GeoTIFF raster

    Args:
        data: numpy array to export
        filename: output filename
        transform: affine transform for georeferencing
        crs: coordinate reference system (defaults to EPSG:2154 if None)
    """
    if crs is None:
        crs = "EPSG:2154"
    with rasterio.open(
        filename,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(data, 1)
