from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from flair_hub.models.flair_model import FLAIR_HUB_Model
from safetensors.torch import load_file as safe_load_file

from .class_mappings import SIMPLIFIED_CLASSES


def build_flair_config(num_classes: int = 19) -> dict:
    """
    Build configuration dictionary for FLAIR model.

    Args:
        num_classes: Number of output classes (19 for pretrained, 4 for fine-tuning)
        tile_size: Input tile size (default: 512)

    Returns:
        Configuration dictionary
    """
    # Class names
    if num_classes == 4:
        value_name = SIMPLIFIED_CLASSES
    else:
        value_name = {i: f"class_{i}" for i in range(num_classes)}

    return {
        "modalities": {
            "inputs": {
                "AERIAL_RGBI": True,
                "AERIAL-RLT_PAN": False,
                "DEM_ELEV": False,
                "SPOT_RGBI": False,
                "SENTINEL2_TS": False,
                "SENTINEL1-ASC_TS": False,
                "SENTINEL1-DESC_TS": False,
            },
            "inputs_channels": {
                "AERIAL_RGBI": [1, 2, 3],  # RGB
            },
            "aux_loss": {
                "AERIAL_RGBI": False,
                "AERIAL-RLT_PAN": False,
                "DEM_ELEV": False,
                "SPOT_RGBI": False,
                "SENTINEL2_TS": False,
                "SENTINEL1-ASC_TS": False,
                "SENTINEL1-DESC_TS": False,
            },
            "normalization": {
                "norm_type": "custom",
                "AERIAL_RGBI_means": [106.59, 105.66, 111.35],
                "AERIAL_RGBI_stds": [39.78, 52.23, 45.62],
            },
        },
        "models": {
            "monotemp_model": {
                "arch": "swin_large_patch4_window12_384-upernet",
            },
            "multitemp_model": {
                "ref_date": "05-15",
                "encoder_widths": [64, 64, 64, 128],
                "decoder_widths": [32, 32, 64, 128],
                "out_conv": [32, num_classes],
                "str_conv_k": 3,
                "str_conv_s": 1,
                "str_conv_p": 1,
                "agg_mode": "att_group",
                "encoder_norm": "group",
                "n_head": 16,
                "d_model": 256,
                "d_k": 4,
                "pad_value": 0,
                "padding_mode": "reflect",
            },
        },
        "labels": ["AERIAL_LABEL-COSIA"],
        "labels_configs": {
            "AERIAL_LABEL-COSIA": {
                "task_weight": 1,
                "value_name": value_name,
            },
        },
    }


def load_flair_model(
    checkpoint_path: str,
    tile_size: int = 512,
    num_classes: int = 19,
    device: str = "cpu",
) -> FLAIR_HUB_Model:
    """
    Load FLAIR model from checkpoint.

    Args:
        checkpoint_path: Path to .safetensors or .pth checkpoint
        tile_size: Input tile size
        num_classes: Number of classes (use 19 to match checkpoint)
        device: Device to load model on

    Returns:
        Loaded FLAIR_HUB_Model
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    config = build_flair_config(num_classes=num_classes)

    img_input_sizes = {"AERIAL_RGBI": tile_size}
    model = FLAIR_HUB_Model(config=config, img_input_sizes=img_input_sizes)

    if checkpoint_path.suffix == ".pth":
        print("  Loading PyTorch checkpoint (.pth)")
        checkpoint = torch.load(str(checkpoint_path), map_location=device)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if "epoch" in checkpoint:
                print(f"  Checkpoint from epoch {checkpoint['epoch']}")
        else:
            state_dict = checkpoint
    else:
        print("  Loading safetensors checkpoint")
        state_dict = safe_load_file(str(checkpoint_path))

    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("criterion."):
            continue
        elif k.startswith("model."):
            new_state_dict[k[6:]] = v
        else:
            new_state_dict[k] = v

    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)

    if len(missing) > 0:
        print(f"  Warning: {len(missing)} missing keys (criterion weights expected)")
    if len(unexpected) > 0:
        print(f"  Warning: {len(unexpected)} unexpected keys")

    model = model.to(device)
    model.eval()

    return model


def replace_segmentation_head_4_classes(
    model: FLAIR_HUB_Model, num_classes: int = 4
) -> FLAIR_HUB_Model:
    """
    Replace the segmentation head to output 4 classes instead of 19.

    Args:
        model: FLAIR_HUB_Model instance
        num_classes: Number of output classes (default: 4)

    Returns:
        Modified model with new segmentation head
    """
    print(f"\n{'=' * 60}")
    print(f"Replacing segmentation head: 19 classes → {num_classes} classes")
    print(f"{'=' * 60}")

    task_name = "AERIAL_LABEL-COSIA"
    decoder = model.main_decoders[task_name]
    seg_head = decoder.seg_model.segmentation_head

    # Get old conv layer and its device
    old_conv = seg_head[0]
    in_channels = old_conv.in_channels
    old_out_channels = old_conv.out_channels
    device = old_conv.weight.device  # Get device from existing weights

    print(f"\nOriginal: Conv2d({in_channels} → {old_out_channels})")

    # Create new conv
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=num_classes,
        kernel_size=1,
        stride=1,
        padding=0,
        bias=True,
    )

    # Initialize
    nn.init.xavier_uniform_(new_conv.weight)
    if new_conv.bias is not None:
        nn.init.zeros_(new_conv.bias)

    # Move to same device as the model
    new_conv = new_conv.to(device)

    # Replace
    seg_head[0] = new_conv

    print(f"New:      Conv2d({in_channels} → {num_classes})")
    print(f"✓ Randomly initialized (Xavier uniform) on device: {device}")
    print(f"{'=' * 60}\n")

    return model


def freeze_encoder_selective(
    model: FLAIR_HUB_Model,
    freeze_encoder: bool = True,
    freeze_decoder_blocks: Optional[int] = None,
) -> int:
    """
    Freeze encoder and optionally early decoder blocks for efficient fine-tuning.

    Args:
        model: FLAIR_HUB_Model instance
        freeze_encoder: If True, freeze the Swin encoder (default: True)
        freeze_decoder_blocks: Number of decoder blocks to freeze (None = keep all trainable)

    Returns:
        Number of trainable parameters
    """
    print(f"\n{'=' * 70}")
    print("Freezing model layers for fine-tuning")
    print(f"{'=' * 70}")

    task_name = "AERIAL_LABEL-COSIA"
    encoder_key = "AERIAL_RGBI"

    if freeze_encoder:
        encoder = model.encoders[encoder_key].seg_model
        for param in encoder.parameters():
            param.requires_grad = False
        print("✓ Froze Swin encoder")

    # Freeze decoder blocks if requested
    decoder = model.main_decoders[task_name].seg_model.decoder

    if freeze_decoder_blocks is not None and freeze_decoder_blocks > 0:
        decoder_params = list(decoder.named_parameters())
        total_decoder_params = len(decoder_params)

        frozen_count = 0
        for i, (_name, param) in enumerate(decoder_params):
            if i < freeze_decoder_blocks:
                param.requires_grad = False
                frozen_count += 1

        print(
            f"✓ Froze first {frozen_count}/{total_decoder_params} decoder parameter groups"
        )
    else:
        print("  Decoder: All layers trainable")

    # Segmentation head is always trainable
    seg_head = model.main_decoders[task_name].seg_model.segmentation_head
    for param in seg_head.parameters():
        param.requires_grad = True
    print("✓ Segmentation head: Trainable")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    frozen = total - trainable

    print(f"\n{'=' * 70}")
    print("Parameter summary:")
    print(f"  Total:     {total:>15,}")
    print(f"  Frozen:    {frozen:>15,} ({100 * frozen / total:>5.1f}%)")
    print(f"  Trainable: {trainable:>15,} ({100 * trainable / total:>5.1f}%)")
    print(f"{'=' * 70}\n")

    return trainable


def print_model_structure(model: FLAIR_HUB_Model):
    """Print the structure of the FLAIR model."""
    print(f"\n{'=' * 70}")
    print("FLAIR Model Structure")
    print(f"{'=' * 70}")

    print("\nEncoders:")
    for key, encoder in model.encoders.items():
        params = sum(p.numel() for p in encoder.parameters())
        trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        print(f"  {key}:")
        print(f"    Parameters: {params:,}")
        print(f"    Trainable:  {trainable:,}")

    print("\nMain Decoders:")
    for key, decoder in model.main_decoders.items():
        params = sum(p.numel() for p in decoder.parameters())
        trainable = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
        print(f"  {key}:")
        print(f"    Parameters: {params:,}")
        print(f"    Trainable:  {trainable:,}")

        seg_head = decoder.seg_model.segmentation_head[0]
        print(
            f"    Segmentation head: Conv2d({seg_head.in_channels} → {seg_head.out_channels})"
        )

    print(f"{'=' * 70}\n")
