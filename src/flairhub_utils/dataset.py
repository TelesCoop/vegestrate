import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

from .class_mappings import SIMPLIFIED_CLASSES
from .inference import normalize_image


class FlairDataset(Dataset):
    """Multi-tile dataset for FLAIR fine-tuning.

    Loads orthophotos and classification maps from multiple tiles,
    extracts random patches, applies augmentations, and returns
    normalized images with 4-class segmentation masks.
    Ground truth is the LIDAR.
    """

    def __init__(
        self,
        data_manifest: str,
        split: str = "train",
        patch_size: int = 512,
        patches_per_tile: int = 100,
        classes_to_train: Optional[List[str]] = None,
        augment: bool = True,
    ):
        """Initialize FlairDataset.

        Args:
            data_manifest: Path to dataset_manifest.json
            split: 'train' or 'test'
            patch_size: Size of patches (default: 512)
            patches_per_tile: Patches per tile (default: 100)
            classes_to_train: List of class names to train on (default: all vegetation)
            augment: Data augmentation (default: True)
        """
        self.patch_size = patch_size
        self.augment = augment

        with open(data_manifest, "r") as f:
            manifest = json.load(f)

        self.tiles_info = manifest[split]
        self.data_dir = Path(data_manifest).parent

        # Classes to train on (default: vegetation only, exclude 'else')
        if classes_to_train is None:
            self.classes = [c for c in SIMPLIFIED_CLASSES.values() if c != "else"]
        else:
            self.classes = classes_to_train

        print(f"Training on classes: {self.classes}")

        self.tiles_data = []
        for tile_info in self.tiles_info:
            ortho_path = self.data_dir / tile_info["orthophoto"]
            classmap_path = self.data_dir / tile_info["classification_map"]

            with rasterio.open(ortho_path) as src:
                ortho = src.read([1, 2, 3])
                ortho = np.transpose(ortho, (1, 2, 0))  # (H, W, 3)
                if ortho.dtype == np.uint16:
                    ortho = (ortho / 256).astype(np.uint8)
                elif ortho.dtype == np.float32:
                    ortho = np.clip(ortho * 255, 0, 255).astype(np.uint8)

            with rasterio.open(classmap_path) as src:
                classmap = src.read(1)

            assert (
                ortho.shape[:2] == classmap.shape
            ), f"Shape mismatch: ortho {ortho.shape[:2]} vs classmap {classmap.shape}"

            class_map_4 = self._map_las_to_4classes(classmap)

            self.tiles_data.append(
                {
                    "tile_id": tile_info["tile_id"],
                    "ortho": ortho,
                    "classmap": class_map_4,
                    "height": classmap.shape[0],
                    "width": classmap.shape[1],
                }
            )
            print(f"  ✓ Loaded tile {tile_info['tile_id']}: {ortho.shape}")

        # Extract random patches from all tiles
        self.patches = []
        for tile_idx, tile in enumerate(self.tiles_data):
            max_y = tile["height"] - patch_size
            max_x = tile["width"] - patch_size
            if max_y <= 0 or max_x <= 0:
                print(f"  Warning: Tile {tile['tile_id']} too small for patches")
                continue

            for _ in range(patches_per_tile):
                y = np.random.randint(0, max_y)
                x = np.random.randint(0, max_x)
                self.patches.append({"tile_idx": tile_idx, "y": y, "x": x})

        print(
            f"Extracted {len(self.patches)} patches from {len(self.tiles_data)} tiles"
        )
        print(f"Total dataset size: {len(self.patches)} samples")

    def _map_las_to_4classes(self, classmap: np.ndarray) -> np.ndarray:
        """Map LAS classification values to 4-class system.

        LAS classification (from prepare_training_data.py):
        - Filtered to classes 2-5 (ground and vegetation)
        - Scaled values in raster: 0, 63, 127, 191 (for 4 unique classes)
        - 255 = nodata

        Our 4-class system:
        - 0: else (ground, class 2) -> scaled value 0
        - 1: herbaceous (low vegetation, class 3) -> scaled value 63
        - 2: hedge (medium vegetation, class 4) -> scaled value 127
        - 3: trees (high vegetation, class 5) -> scaled value 191

        Args:
            classmap: Classification raster with scaled values

        Returns:
            Class map with values 0-3
        """
        class_map_4 = np.zeros_like(classmap, dtype=np.uint8)

        # Map scaled LAS values to our 4 classes
        class_map_4[classmap == 0] = 0  # ground -> else
        class_map_4[classmap == 63] = 1  # low veg -> herbaceous
        class_map_4[classmap == 127] = 2  # medium veg -> hedge
        class_map_4[classmap == 191] = 3  # high veg -> trees
        class_map_4[classmap == 255] = 0  # nodata -> else

        return class_map_4

    def _augment(self, image: np.ndarray, mask: np.ndarray):
        """Apply data augmentation.

        Args:
            image: Image patch (H, W, 3)
            mask: Segmentation mask (H, W)

        Returns:
            Augmented image and mask
        """
        if not self.augment:
            return image, mask

        if np.random.rand() > 0.5:
            image = np.fliplr(image)
            mask = np.fliplr(mask)

        if np.random.rand() > 0.5:
            image = np.flipud(image)
            mask = np.flipud(mask)

        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k)
            mask = np.rot90(mask, k)

        return image.copy(), mask.copy()

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        """Get a training sample.

        Returns:
            Dictionary with:
                - image: Normalized image tensor (3, H, W), float32
                - mask: Class mask tensor (H, W), long
                - class_mask: Binary mask for each class (num_classes, H, W), float32
        """
        patch = self.patches[idx]
        tile = self.tiles_data[patch["tile_idx"]]

        y, x = patch["y"], patch["x"]
        s = self.patch_size
        img_patch = tile["ortho"][y : y + s, x : x + s, :].copy()
        class_patch = tile["classmap"][y : y + s, x : x + s].copy()

        img_patch, class_patch = self._augment(img_patch, class_patch)

        # Normalize image (FLAIR normalization)
        # normalize_image returns (H, W, C), need to transpose to (C, H, W)
        img_float = img_patch.astype(np.float32)
        img_normalized = normalize_image(img_float)  # Returns (H, W, 3) float32
        img_normalized = np.transpose(img_normalized, (2, 0, 1))  # (3, H, W)

        img_tensor = torch.from_numpy(img_normalized.copy()).float()

        mask = torch.from_numpy(class_patch).long()

        num_classes = len(SIMPLIFIED_CLASSES)
        class_mask = torch.zeros(num_classes, s, s, dtype=torch.float32)
        for c in range(num_classes):
            class_mask[c] = (mask == c).float()

        return {
            "image": img_tensor,
            "mask": mask,
            "class_mask": class_mask,
        }
