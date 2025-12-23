import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds


def resize_and_save(raster_path, resolution, bounds, crs, output_path):
    """Take a raster file, update and resolution and save it"""

    img = Image.open(raster_path)
    img_array = np.array(img)

    print("\nReceived image:")
    print(f"  Size: {img.size}")
    print(f"  Mode: {img.mode}")
    print(f"  Array shape: {img_array.shape}")

    target_pixel_size = resolution  # meters per pixel
    new_width = int((bounds.right - bounds.left) / target_pixel_size)
    new_height = int((bounds.top - bounds.bottom) / target_pixel_size)

    print("\nResizing to 0.8m resolution:")
    print(f"  Target size: {new_width}x{new_height}")

    img_resized = img.resize((new_width, new_height), Image.BICUBIC)
    img_array = np.array(img_resized)

    width = new_width
    height = new_height
    transform = from_bounds(
        bounds.left, bounds.bottom, bounds.right, bounds.top, new_width, new_height
    )

    print(f"  Resized array shape: {img_array.shape}")

    if img_resized.mode == "RGB":
        count = 3
        dtype = rasterio.uint8
    elif img_resized.mode == "L":
        count = 1
        dtype = rasterio.uint8
        img_array = (
            img_array[:, :, np.newaxis] if len(img_array.shape) == 2 else img_array
        )
    else:
        count = img_array.shape[2] if len(img_array.shape) == 3 else 1
        dtype = rasterio.uint8

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform,
        compress="JPEG",
        photometric="RGB" if count == 3 else None,
    ) as dst:
        if count == 3:
            # RGB image
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        else:
            # Single band
            dst.write(img_array.squeeze(), 1)

    print(f"\nSaved to: {output_path}")
