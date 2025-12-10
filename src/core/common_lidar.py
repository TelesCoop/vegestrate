import laspy
import numpy as np

from .utils import classification_to_raster, export_raster

CRS = "EPSG:2154"
GROUND_CLASSIFICATION = 2

# LAS Classification codes (ASPRS Standard)
LAS_CLASSIFICATIONS = {
    0: "Never classified / Created",
    1: "Unclassified",
    2: "Ground",
    3: "Low Vegetation < 1.5m",
    4: "Medium Vegetation 1.5-5m",
    5: "High Vegetation 5-15m",
    6: "Building",
    7: "Low Point (noise)",
    8: "Reserved: very high vegetation > 15m",
    9: "Water",
    10: "Rail",
    11: "Road Surface",
    12: "Reserved",
    13: "Wire - Guard (Shield)",
    14: "Wire - Conductor (Phase)",
    15: "Transmission Tower",
    16: "Wire-structure Connector (Insulator)",
    17: "Bridge Deck",
    18: "High Noise",
}


def print_classification_info(las):
    """Print classification codes present in the LAS data"""
    unique_classifications = sorted(np.unique(las.classification))

    print("\n--- Classification Distribution ---")
    print(f"Total points: {len(las.points):,}")
    print("-" * 60)

    for code in unique_classifications:
        count = np.sum(las.classification == code)
        percentage = (count / len(las.points)) * 100
        description = LAS_CLASSIFICATIONS.get(code, "Unknown")
        print(f"  {code:2d} - {description:20s} {count:12,} ({percentage:5.2f}%)")

    print("-" * 60)


def filter_ground_vegetation(las, lyon=False):
    """Filter ground and vegetation points (classes 2-5)

    Args:
        las: laspy LAS object
        lyon(bollean): if data are from data.grandLyon also take las.classification 8

    Returns:
        filtered_las: laspy LAS object with only ground and vegetation points
    """
    print("Filtering ground and vegetation classes...")
    if lyon:
        ground_vegetation_mask = (
            (las.classification <= 5) & (las.classification >= 2)
        ) | (las.classification == 8)
    else:
        ground_vegetation_mask = (las.classification <= 5) & (las.classification >= 2)
    header = laspy.LasHeader(
        point_format=las.header.point_format, version=las.header.version
    )
    header.scales = las.header.scales
    header.offsets = las.header.offsets
    filtered_las = laspy.LasData(header)
    filtered_las.points = las.points[ground_vegetation_mask]
    print(f"✓ Filtered to {len(filtered_las.points):,} points")
    return filtered_las


def create_classification_map(
    filtred_las, las, output_path, resolution=0.8, crs="EPSG:2154"
):
    """Create and export classification raster from LAS data

    Args:
        filtred_las: laspy LAS object with only vegetation
        las: laspy LAS object with all all points
        output_path: Path to save classification_map.tif
        resolution: Raster cell size in meters (default: 0.8)
        crs: Coordinate reference system (default: EPSG:2154)

    Returns:
        Path to classification_map.tif
    """
    print(f"Creating classification raster (resolution={resolution}m)...")
    raster, affine, las_crs = classification_to_raster(
        filtred_las, las, cell_size=resolution
    )
    print(f"✓ Raster shape: {raster.shape}")

    # Use CRS from LAS file if available, otherwise use the provided crs parameter
    final_crs = las_crs if las_crs is not None else crs
    export_raster(raster, str(output_path), affine, crs=final_crs)
    print(f"✓ Saved classification map: {output_path}")

    return output_path
