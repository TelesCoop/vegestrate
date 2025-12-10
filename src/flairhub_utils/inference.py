"""Inference utilities for FLAIR-HUB segmentation."""

from typing import Tuple

import numpy as np
import torch

AERIAL_RGBI_MEANS = np.array([106.59, 105.66, 111.35])
AERIAL_RGBI_STDS = np.array([39.78, 52.23, 45.62])


def normalize_image(
    image: np.ndarray,
    means: np.ndarray = AERIAL_RGBI_MEANS,
    stds: np.ndarray = AERIAL_RGBI_STDS,
) -> np.ndarray:
    """
    Normalize image using FLAIR statistics.

    Args:
        image: Image array (H, W, C) with values 0-255
        means: Mean values for each channel
        stds: Std values for each channel

    Returns:
        Normalized image
    """
    return (image - means) / stds


def create_weight_map(h: int, w: int, overlap: int) -> np.ndarray:
    """Create distance-based weight map for tile blending.

    Same pattern as CLIPSeg inference.

    Args:
        h: Tile height
        w: Tile width
        overlap: Overlap size in pixels

    Returns:
        Weight map (h, w)
    """
    weight = np.ones((h, w), dtype=np.float32)

    if overlap > 0:
        # Cos ramp in overlap regions
        ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, overlap))

        if h > overlap:
            weight[:overlap, :] = ramp[:, np.newaxis]
        if h > overlap:
            weight[-overlap:, :] = ramp[::-1, np.newaxis]
        if w > overlap:
            weight[:, :overlap] *= ramp[np.newaxis, :]
        if w > overlap:
            weight[:, -overlap:] *= ramp[::-1][np.newaxis, :]

    return weight


class FlairInference:
    """Helper class for FLAIR model inference operations.

    Handles preprocessing, tiling, and postprocessing.
    """

    @staticmethod
    def preprocess_tile(tile_image: np.ndarray, normalize: bool = True) -> torch.Tensor:
        """
        Preprocess a tile for model input.

        Args:
            tile_image: Tile image (H, W, C) in range 0-255
            normalize: Whether to normalize (default: True)

        Returns:
            Preprocessed tensor (1, C, H, W)
        """
        if normalize:
            tile_image = normalize_image(tile_image)

        tile_tensor = torch.from_numpy(tile_image).float()
        tile_tensor = tile_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

        return tile_tensor

    @staticmethod
    def create_tiles_grid(
        height: int, width: int, tile_size: int, overlap: int
    ) -> list:
        """
        Create grid of tile positions.

        Args:
            height: Image height
            width: Image width
            tile_size: Tile size
            overlap: Overlap between tiles

        Returns:
            List of tile dicts with keys: 'y', 'x', 'h', 'w'
        """
        stride = tile_size - overlap
        tiles = []

        for y in range(0, height, stride):
            for x in range(0, width, stride):
                y_end = min(y + tile_size, height)
                x_end = min(x + tile_size, width)

                y_start = max(0, y_end - tile_size)
                x_start = max(0, x_end - tile_size)

                tiles.append(
                    {
                        "y": y_start,
                        "x": x_start,
                        "h": y_end - y_start,
                        "w": x_end - x_start,
                    }
                )

        return tiles

    @staticmethod
    def merge_tile_predictions(
        tile_logits_list: list,
        tile_positions: list,
        output_shape: Tuple[int, int],
        num_classes: int,
        overlap: int,
    ) -> np.ndarray:
        """
        Merge tiled predictions with weighted blending.

        Args:
            tile_logits_list: List of tile logits (num_classes, H, W)
            tile_positions: List of tile position dicts
            output_shape: Output shape (H, W)
            num_classes: Number of classes
            overlap: Overlap size

        Returns:
            Merged logits (H, W, num_classes)
        """
        h, w = output_shape
        output_logits = np.zeros((h, w, num_classes), dtype=np.float32)
        weight_map = np.zeros((h, w), dtype=np.float32)

        for tile_logits, tile_info in zip(tile_logits_list, tile_positions):
            y, x = tile_info["y"], tile_info["x"]
            tile_h, tile_w = tile_info["h"], tile_info["w"]

            tile_weight = create_weight_map(tile_h, tile_w, overlap)

            for c in range(num_classes):
                output_logits[y : y + tile_h, x : x + tile_w, c] += (
                    tile_logits[c, :, :] * tile_weight
                )

            weight_map[y : y + tile_h, x : x + tile_w] += tile_weight

        for c in range(num_classes):
            output_logits[:, :, c] = np.divide(
                output_logits[:, :, c], weight_map, where=weight_map > 0
            )

        return output_logits

    @staticmethod
    def postprocess_logits(
        logits: np.ndarray,
        mode: str = "argmax",
        class_id: int = None,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """
        Postprocess logits to final predictions.

        Args:
            logits: Logits (H, W, num_classes)
            mode: 'argmax' for class map, 'single_class' for binary mask
            class_id: Class ID for single_class mode
            threshold: Threshold for single_class mode

        Returns:
            Predictions (H, W) as uint8
        """
        if mode == "argmax":
            return np.argmax(logits, axis=2).astype(np.uint8)
        elif mode == "single_class":
            if class_id is None:
                raise ValueError("class_id must be specified for single_class mode")
            exp_logits = np.exp(logits - np.max(logits, axis=2, keepdims=True))
            probs = exp_logits / np.sum(exp_logits, axis=2, keepdims=True)
            class_prob = probs[:, :, class_id]
            return (class_prob >= threshold).astype(np.uint8)
        else:
            raise ValueError(f"Unknown mode: {mode}")
