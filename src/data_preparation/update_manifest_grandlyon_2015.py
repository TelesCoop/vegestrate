import csv
import io
import json
from pathlib import Path

import requests

LIDAR_CSV_URL = (
    "https://data.grandlyon.com/geoserver/metropole-de-lyon/ows"
    "?SERVICE=WFS&VERSION=2.0.0&request=GetFeature"
    "&typename=metropole-de-lyon:ima_gestion_images.imacartogrammelidar"
    "&outputFormat=CSV&SRSNAME=EPSG:3946"
)


def main():
    manifest_path = Path("data/dataset_manifest_grandlyon_2015.json")

    resp = requests.get(LIDAR_CSV_URL, timeout=120)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))

    tiles = []
    for row in reader:
        tile_id = row["nom"]
        url = row["url"].strip()
        tiles.append(
            {
                "tile_id": tile_id,
                "orthophoto": f"test/{tile_id}_orthophoto.tif",
                "classification_map": f"test/classification_map_{tile_id}.tif",
                "ir": f"test/{tile_id}_ir.tif",
                "url": url,
            }
        )

    manifest = {"test": tiles, "train": []}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"✓ Wrote {manifest_path} with {len(tiles)} LiDAR 2015 tiles")


if __name__ == "__main__":
    main()
