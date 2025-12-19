import argparse
import json
import re
from pathlib import Path


def extract_tile_id(url: str) -> str:
    """Extract tile ID from LIDAR URL.

    Example URL:
    https://data.geopf.fr/.../LHD_FXX_0844_6520_PTS_LAMB93_IGN69.copc.laz

    Args:
        url: LIDAR download URL

    Returns:
        Tile ID in format "XXXX_XXXX" (e.g., "0844_6520")

    Raises:
        ValueError: If tile ID cannot be extracted from URL
    """
    # Pattern: LHD_FXX_XXXX_XXXX_...
    match = re.search(r"LHD_FXX_(\d+)_(\d+)_", url)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    else:
        raise ValueError(f"Could not extract tile ID from URL: {url}")


def read_urls_from_file(file_path: str) -> list:
    """Read URLs from file (one per line).

    Args:
        file_path: Path to file containing URLs

    Returns:
        List of URLs (stripped of whitespace)
    """
    urls = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def create_manifest_entry(url: str, split: str) -> dict:
    """Create a manifest entry for a LIDAR URL.

    Args:
        url: LIDAR download URL
        split: 'train' or 'test'

    Returns:
        Dictionary with tile_id, orthophoto, classification_map, url
    """
    tile_id = extract_tile_id(url)

    return {
        "tile_id": tile_id,
        "orthophoto": f"{split}/orthophoto_{tile_id}.tif",
        "classification_map": f"{split}/classification_map_{tile_id}.tif",
        "url": url,
    }


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Generate dataset manifest from LIDAR URL files"
    )
    parser.add_argument(
        "--train",
        type=str,
        default="data/dalles_train.txt",
        help="File containing training LIDAR URLs (one per line)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="data/dalles_test.txt",
        help="File containing testing LIDAR URLs (one per line)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/dataset_manifest.json",
        help="Output manifest file (default: data/dataset_manifest.json)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GENERATE DATASET MANIFEST FROM URL FILES")
    print("=" * 70)
    print(f"Train URLs file: {args.train}")
    print(f"Test URLs file: {args.test}")
    print(f"Output manifest: {args.output}")
    print("=" * 70)

    print("\nReading training URLs...")
    train_urls = read_urls_from_file(args.train)
    print(f"✓ Found {len(train_urls)} training URLs")

    print("Reading testing URLs...")
    test_urls = read_urls_from_file(args.test)
    print(f"✓ Found {len(test_urls)} testing URLs")

    print("\nProcessing training URLs...")
    train_entries = []
    for url in train_urls:
        entry = create_manifest_entry(url, "train")
        train_entries.append(entry)

    print("\nProcessing testing URLs...")
    test_entries = []
    for url in test_urls:
        entry = create_manifest_entry(url, "test")
        test_entries.append(entry)

    manifest = {
        "train": train_entries,
        "test": test_entries,
    }

    output_path = Path(args.output)
    if output_path.exists():
        output_path.unlink()
        print(f"\n✓ Deleted existing manifest: {args.output}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Saving new manifest to {args.output}...")
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 70)
    print("MANIFEST GENERATION COMPLETE")


if __name__ == "__main__":
    main()
