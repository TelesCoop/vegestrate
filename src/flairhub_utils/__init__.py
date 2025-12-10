"""FLAIR-HUB utilities for segmentation, training, and fine-tuning."""

from .class_mappings import (  # noqa: F401
    CLASS_REMAP_19_TO_4,
    FLAIR_CLASSES,
    SIMPLIFIED_CLASSES,
    remap_to_4_classes,
)
from .dataset import FlairDataset  # noqa: F401
from .inference import FlairInference, create_weight_map, normalize_image  # noqa: F401
from .model_utils import (  # noqa: F401
    build_flair_config,
    freeze_encoder_selective,
    load_flair_model,
    print_model_structure,
    replace_segmentation_head_4_classes,
)
from .training import (  # noqa: F401
    FocalDiceLoss,
    compute_metrics,
    train_epoch,
    validate_epoch,
)

__all__ = [
    "build_flair_config",
    "load_flair_model",
    "replace_segmentation_head_4_classes",
    "freeze_encoder_selective",
    "print_model_structure",
    "FlairInference",
    "normalize_image",
    "create_weight_map",
    "FLAIR_CLASSES",
    "SIMPLIFIED_CLASSES",
    "CLASS_REMAP_19_TO_4",
    "remap_to_4_classes",
    "FlairDataset",
    "FocalDiceLoss",
    "compute_metrics",
    "train_epoch",
    "validate_epoch",
]
