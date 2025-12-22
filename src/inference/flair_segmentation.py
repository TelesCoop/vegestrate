from pathlib import Path
from typing import Optional

import numpy as np
import rasterio
import torch
import torch.nn.functional as F

from ..flairhub_utils import (
    FLAIR_CLASSES,
    SIMPLIFIED_CLASSES,
    FlairInference,
    create_weight_map,
    load_flair_model,
    remap_to_4_classes,
)


class FlairSegmentation:
    """Wrapper for FLAIR-HUB land cover segmentation with 4-class remapping."""

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[str] = None,
        use_simplified_classes: bool = True,
        checkpoint_num_classes: Optional[int] = None,
    ):
        """
        Initialize FLAIR segmentation model.

        Args:
            checkpoint_path: Path to .safetensors checkpoint
            device: 'cuda' or 'cpu' (auto-detected if None)
            use_simplified_classes: If True, expect/use 4 classes (default: True)
            checkpoint_num_classes: Number of classes in checkpoint (auto-detect if None)
                                    Use 4 for fine-tuned, 19 for pretrained
        """
        self.checkpoint_path = Path(checkpoint_path)
        self.use_simplified_classes = use_simplified_classes

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"Using device: {self.device}")

        self.model = None
        self.current_tile_size = None

        if checkpoint_num_classes is not None:
            self.num_classes = checkpoint_num_classes
        else:
            # Default: 4 for simplified, 19 for full
            self.num_classes = 4 if self.use_simplified_classes else 19

        print("FlairSegmentation initialized")
        print(f"Checkpoint: {self.checkpoint_path}")
        print(f"Model classes: {self.num_classes}")
        if self.use_simplified_classes:
            print("Mode: 4 simplified classes (else, herbaceous, hedge, trees)")
        else:
            print("Mode: 19 FLAIR classes")

    def _load_model(self, tile_size: int):
        """Load model with specified tile size (lazy loading)."""
        if self.model is not None and self.current_tile_size == tile_size:
            return

        print(f"Loading model with tile_size={tile_size}...")

        self.model = load_flair_model(
            checkpoint_path=str(self.checkpoint_path),
            tile_size=tile_size,
            num_classes=self.num_classes,
            device=self.device,
        )

        self.current_tile_size = tile_size
        print("  ✓ Model loaded")

    def _load_and_normalize_image(self, image_path: str):
        """Load and normalize raster image.

        Args:
            image_path: Path to input TIF file

        Returns:
            Tuple of (image_data, height, width, meta)
        """
        with rasterio.open(image_path) as src:
            if src.count >= 3:
                image_data = src.read([1, 2, 3])
            else:
                raise ValueError(f"Image must have at least 3 bands, found {src.count}")

            image_data = np.transpose(image_data, (1, 2, 0))

            if image_data.dtype == np.uint16:
                image_data = (image_data / 257).astype(np.uint8)
            elif image_data.dtype == np.float32:
                image_data = np.clip(image_data * 255, 0, 255).astype(np.uint8)

            height, width = image_data.shape[:2]
            meta = src.meta.copy()

        return image_data, height, width, meta

    def _process_tile(self, tile_image, tile_h, tile_w, class_id):
        """Process a single tile through the model.

        Args:
            tile_image: Tile image data
            tile_h: Tile height
            tile_w: Tile width
            class_id: Class ID for single-class mode (or None)

        Returns:
            Tuple of (class_prob or logits_np, depending on class_id)
        """
        tile_tensor = FlairInference.preprocess_tile(tile_image, normalize=True)
        tile_tensor = tile_tensor.to(self.device)

        batch = {
            "AERIAL_LABEL-COSIA": torch.zeros(1, tile_h, tile_w).to(self.device),
            "AERIAL_RGBI": tile_tensor,
        }

        if self.model is None:
            raise RuntimeError("Model not loaded")
        logits_tasks, _ = self.model(batch)
        logits = logits_tasks["AERIAL_LABEL-COSIA"]  # (1, num_classes, H, W)

        logits = logits.squeeze(0)  # (num_classes, H, W)

        if logits.shape[1:] != (tile_h, tile_w):
            logits = F.interpolate(
                logits.unsqueeze(0),
                size=(tile_h, tile_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if class_id is not None:
            probs = F.softmax(logits, dim=0).cpu().numpy()
            return probs[class_id, :, :]
        else:
            return logits.cpu().numpy()

    def _augment_image(self, image: np.ndarray, mode: str) -> np.ndarray:
        """Apply test time augmentation (TTA).

        Args:
            image: Image array (H, W, C)
            mode: Augmentation mode ('original', 'hflip', 'vflip', 'hvflip',
                  'rot90', 'rot180', 'rot270')

        Returns:
            Augmented image
        """
        if mode == "original":
            return image
        elif mode == "hflip":
            return np.fliplr(image)
        elif mode == "vflip":
            return np.flipud(image)
        elif mode == "hvflip":
            return np.flipud(np.fliplr(image))
        elif mode == "rot90":
            return np.rot90(image, k=1)
        elif mode == "rot180":
            return np.rot90(image, k=2)
        elif mode == "rot270":
            return np.rot90(image, k=3)
        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")

    def _deaugment_output(
        self, output: np.ndarray, mode: str, is_probs: bool = False  # noqa: ARG002
    ) -> np.ndarray:
        """Reverse augmentation on output.

        Args:
            output: Output array - either (H, W) for probs or (H, W, C) for logits
            mode: Augmentation mode used
            is_probs: If True, output is (H, W) probs; if False, (H, W, C) logits

        Returns:
            De-augmented output
        """
        if mode == "original":
            return output
        elif mode == "hflip":
            return np.fliplr(output)
        elif mode == "vflip":
            return np.flipud(output)
        elif mode == "hvflip":
            return np.flipud(np.fliplr(output))
        elif mode == "rot90":
            return np.rot90(output, k=-1)  # Rotate back
        elif mode == "rot180":
            return np.rot90(output, k=-2)
        elif mode == "rot270":
            return np.rot90(output, k=-3)
        else:
            raise ValueError(f"Unknown augmentation mode: {mode}")

    def _segment_single_pass(
        self,
        image_data: np.ndarray,
        height: int,
        width: int,
        tile_size: int,
        overlap: int,
        class_id: Optional[int],
    ) -> np.ndarray:
        """Run single segmentation pass on image.

        Args:
            image_data: Image array (H, W, C)
            height: Image height
            width: Image width
            tile_size: Tile size
            overlap: Tile overlap
            class_id: Class ID for binary mode, None for class map

        Returns:
            Either (H, W) probs array or (H, W, num_classes) logits array
        """
        if class_id is not None:
            output_probs = np.zeros((height, width), dtype=np.float32)
        else:
            output_logits = np.zeros(
                (height, width, self.num_classes), dtype=np.float32
            )

        weight_map = np.zeros((height, width), dtype=np.float32)

        tiles = FlairInference.create_tiles_grid(height, width, tile_size, overlap)

        for tile_info in tiles:
            y, x = tile_info["y"], tile_info["x"]
            h, w = tile_info["h"], tile_info["w"]

            tile_image = image_data[y : y + h, x : x + w, :].copy()
            if tile_image.max() < 10:
                continue

            result = self._process_tile(tile_image, h, w, class_id)
            tile_weight = create_weight_map(h, w, overlap)

            if class_id is not None:
                output_probs[y : y + h, x : x + w] += result * tile_weight
            else:
                for c in range(self.num_classes):
                    output_logits[y : y + h, x : x + w, c] += (
                        result[c, :, :] * tile_weight
                    )

            weight_map[y : y + h, x : x + w] += tile_weight

        # Normalize by weight
        if class_id is not None:
            output_probs = np.divide(output_probs, weight_map, where=weight_map > 0)
            return output_probs
        else:
            for c in range(self.num_classes):
                output_logits[:, :, c] = np.divide(
                    output_logits[:, :, c], weight_map, where=weight_map > 0
                )
            return output_logits

    @torch.no_grad()
    def segment_image(
        self,
        image_path: str,
        output_path: str,
        class_id: Optional[int] = None,
        threshold: float = 0.5,
        tile_size: int = 512,
        overlap: int = 32,
        smoothing_sigma: Optional[float] = None,
        use_tta: bool = True,
        tta_modes: Optional[list] = None,
    ):
        """Segment a raster image using FLAIR model.

        Args:
            image_path: Path to input TIF file
            output_path: Path to save output mask (TIF)
            class_id: Class ID to extract (0-3 for simplified, 0-18 for full).
                     If None, returns full class map.
            threshold: Probability threshold for binary mask (when class_id specified)
            tile_size: Size of tiles for processing (default: 512)
            overlap: Overlap between tiles in pixels (default: 32)
            smoothing_sigma: Gaussian smoothing sigma (default: None)
            use_tta: Enable Test-Time Augmentation (default: True)
            tta_modes: List of augmentation modes to use. Options: 'hflip', 'vflip',
                      'hvflip', 'rot90', 'rot180', 'rot270'.
                      Default: ['hflip', 'vflip', 'hvflip'] for 4x augmentation

        Returns:
            Path to output file
        """
        print(f"\nSegmenting image: {image_path}")
        if class_id is not None:
            class_name = self.get_class_name(class_id)
            print(f"Target class: {class_id} - {class_name}")
        else:
            mode_str = "4 classes" if self.use_simplified_classes else "19 classes"
            print(f"Mode: Full class map ({mode_str})")
        print(f"Tile size: {tile_size}, Overlap: {overlap}")

        if use_tta:
            if tta_modes is None:
                tta_modes = ["hflip", "vflip", "hvflip"]
            all_modes = ["original"] + tta_modes
            print(f"TTA enabled with modes: {all_modes}")
            print(f"  ({len(all_modes)}x computation)")

        self._load_model(tile_size)
        image_data, height, width, meta = self._load_and_normalize_image(image_path)
        print(f"Image shape: {image_data.shape}, dtype: {image_data.dtype}")

        if not use_tta:
            # Single pass
            print(
                f"Processing {FlairInference.create_tiles_grid(height, width, tile_size, overlap).__len__()} tiles..."
            )
            output = self._segment_single_pass(
                image_data, height, width, tile_size, overlap, class_id
            )
        else:
            if tta_modes is None:
                tta_modes = ["hflip", "vflip", "hvflip"]
            aug_modes = ["original"] + tta_modes
            outputs = []

            for i, mode in enumerate(aug_modes, 1):
                print(f"\nTTA pass {i}/{len(aug_modes)}: {mode}")
                aug_image = self._augment_image(image_data, mode)

                if mode in ["rot90", "rot270"]:
                    aug_height, aug_width = aug_image.shape[:2]
                else:
                    aug_height, aug_width = height, width

                aug_output = self._segment_single_pass(
                    aug_image, aug_height, aug_width, tile_size, overlap, class_id
                )

                aug_output = self._deaugment_output(
                    aug_output, mode, is_probs=(class_id is not None)
                )
                outputs.append(aug_output)

            print("\nAveraging TTA predictions...")
            output = np.mean(outputs, axis=0).astype(np.float32)

        if class_id is not None:
            self._save_binary_output(
                output,
                threshold,
                smoothing_sigma,
                meta,
                output_path,
            )
        else:
            self._save_classmap_output(output, meta, output_path)

        return output_path

    def _save_binary_output(
        self,
        output_probs,
        threshold,
        smoothing_sigma,
        meta,
        output_path,
    ):
        """Save binary mask output."""
        if smoothing_sigma is not None and smoothing_sigma > 0:
            from scipy.ndimage import gaussian_filter

            output_probs = gaussian_filter(output_probs, sigma=smoothing_sigma)
            print(f"Applied Gaussian smoothing (sigma={smoothing_sigma})")

        binary_mask = (output_probs >= threshold).astype(np.uint8)

        meta.update({"count": 1, "dtype": "uint8", "nodata": 0})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(binary_mask, 1)

        print(f"\n✓ Binary mask saved to {output_path}")
        print(
            f"  Positive pixels: {binary_mask.sum()} / {binary_mask.size} "
            f"({100 * binary_mask.sum() / binary_mask.size:.2f}%)"
        )

    def _save_classmap_output(self, output_logits, meta, output_path):
        """Save class map output."""
        class_map = np.argmax(output_logits, axis=2).astype(np.uint8)

        if self.num_classes == 19 and self.use_simplified_classes:
            class_map = remap_to_4_classes(class_map)
            classes_dict = SIMPLIFIED_CLASSES
            print("\n✓ Remapped 19 classes to 4 simplified classes")
        elif self.num_classes == 4:
            classes_dict = SIMPLIFIED_CLASSES
        else:
            classes_dict = FLAIR_CLASSES

        meta.update({"count": 1, "dtype": "uint8", "nodata": 255})
        with rasterio.open(output_path, "w", **meta) as dst:
            dst.write(class_map, 1)

        print(f"\n✓ Class map saved to {output_path}")
        unique, counts = np.unique(class_map, return_counts=True)
        print("  Class distribution:")
        for cls, cnt in zip(unique, counts):
            class_name = classes_dict.get(cls, "unknown")
            print(
                f"    {cls:2d} ({class_name:20s}): {cnt:8d} pixels "
                f"({100 * cnt / class_map.size:.2f}%)"
            )

    def get_class_name(self, class_id: int, simplified: Optional[bool] = None) -> str:
        """Get class name from ID."""
        if simplified is None:
            simplified = self.use_simplified_classes

        if simplified:
            return SIMPLIFIED_CLASSES.get(class_id, "unknown")
        else:
            return FLAIR_CLASSES.get(class_id, "unknown")

    def list_classes(self, simplified: Optional[bool] = None) -> dict[int, str]:
        """List all available classes."""
        if simplified is None:
            simplified = self.use_simplified_classes

        if simplified:
            return SIMPLIFIED_CLASSES.copy()
        else:
            return FLAIR_CLASSES.copy()
