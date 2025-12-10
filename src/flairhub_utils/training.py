import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .class_mappings import SIMPLIFIED_CLASSES


class FocalDiceLoss(nn.Module):
    """Combined Focal + Dice loss for multi-class segmentation.

    Combines Focal Loss (for class imbalance) with Dice Loss (for spatial overlap).
    """

    def __init__(
        self,
        num_classes: int = 4,
        alpha: float = 0.25,
        gamma: float = 2.0,
        dice_weight: float = 0.5,
        ignore_index: int = -100,
        class_weights: list = None,
    ):
        """Initialize FocalDiceLoss.

        Args:
            num_classes: Number of classes
            alpha: Weight for focal loss (default: 0.25)
            gamma: Focusing parameter for focal loss (default: 2.0)
            dice_weight: Weight for dice loss component (default: 0.5)
            ignore_index: Index to ignore in loss computation (default: -100)
            class_weights: Per-class weights (default: None)
        """
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.class_weights = class_weights

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute Focal Loss for multi-class segmentation.

        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Ground truth class indices (B, H, W)

        Returns:
            Focal loss value
        """
        ce_loss = F.cross_entropy(
            logits, targets, reduction="none", ignore_index=self.ignore_index
        )

        pt = torch.exp(-ce_loss)

        focal = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.class_weights is not None:
            weight_mask = torch.zeros_like(targets, dtype=torch.float32)
            for c, w in enumerate(self.class_weights):
                weight_mask[targets == c] = w
            focal = focal * weight_mask

        return focal.mean()

    def dice_loss(
        self, logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1e-5
    ) -> torch.Tensor:
        """Compute Dice Loss for multi-class segmentation.

        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Ground truth class indices (B, H, W)
            smooth: Smoothing factor (default: 1e-5)

        Returns:
            Dice loss value
        """
        probs = F.softmax(logits, dim=1)

        targets_one_hot = F.one_hot(targets, self.num_classes)  # (B, H, W, C)
        targets_one_hot = targets_one_hot.permute(0, 3, 1, 2).float()  # (B, C, H, W)

        dice_scores = []
        for c in range(self.num_classes):
            pred_c = probs[:, c, :, :]
            target_c = targets_one_hot[:, c, :, :]

            intersection = (pred_c * target_c).sum()
            union = pred_c.sum() + target_c.sum()

            dice = (2.0 * intersection + smooth) / (union + smooth)
            dice_scores.append(dice)

        dice_score = torch.stack(dice_scores).mean()

        return 1.0 - dice_score

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined loss.

        Args:
            logits: Predicted logits (B, C, H, W)
            targets: Ground truth class indices (B, H, W)

        Returns:
            Combined loss value
        """
        focal = self.focal_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        # Combine losses
        loss = (1 - self.dice_weight) * focal + self.dice_weight * dice

        return loss


def compute_metrics(
    pred_logits: torch.Tensor, targets: torch.Tensor, num_classes: int = 4
):
    """Compute per-class IoU, precision, recall, F1 score.

    Args:
        pred_logits: Predicted logits (B, C, H, W)
        targets: Ground truth class indices (B, H, W)
        num_classes: Number of classes

    Returns:
        Dictionary of metrics per class
    """
    preds = torch.argmax(pred_logits, dim=1)  # (B, H, W)

    metrics = {}
    class_names = list(SIMPLIFIED_CLASSES.values())

    for c in range(num_classes):
        pred_mask = (preds == c).float()
        target_mask = (targets == c).float()

        pred_flat = pred_mask.reshape(-1)
        target_flat = target_mask.reshape(-1)

        tp = (pred_flat * target_flat).sum().item()
        fp = (pred_flat * (1 - target_flat)).sum().item()
        fn = ((1 - pred_flat) * target_flat).sum().item()

        precision = tp / (tp + fp + 1e-7)
        recall = tp / (tp + fn + 1e-7)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-7)
        iou = tp / (tp + fp + fn + 1e-7)

        metrics[class_names[c]] = {
            "iou": iou,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics


def train_epoch(
    model, dataloader, optimizer, criterion, device, epoch, track_classes=None
):
    """Train for one epoch.

    Args:
        model: FLAIR model
        dataloader: Training dataloader
        optimizer: Optimizer
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        track_classes: List of class names to track metrics for (default: all vegetation)

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.train()
    total_loss = 0
    num_batches = 0

    if track_classes is None:
        vegetation_classes = [c for c in SIMPLIFIED_CLASSES.values() if c != "else"]
    else:
        vegetation_classes = track_classes

    metrics_accumulator = {
        class_name: {"iou": [], "f1": []} for class_name in vegetation_classes
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Train]")

    for _, batch in enumerate(pbar):
        images = batch["image"].to(device)  # (B, 3, H, W)
        masks = batch["mask"].to(device)  # (B, H, W)

        tile_size = images.shape[-1]
        flair_batch = {
            "AERIAL_LABEL-COSIA": torch.zeros(
                images.shape[0], tile_size, tile_size, device=device
            ),
            "AERIAL_RGBI": images,
        }

        logits_tasks, _ = model(flair_batch)
        logits = logits_tasks["AERIAL_LABEL-COSIA"]  # (B, num_classes, H, W)

        if logits.shape[-2:] != masks.shape[-2:]:
            logits = F.interpolate(
                logits,
                size=masks.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        loss = criterion(logits, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        with torch.no_grad():
            batch_metrics = compute_metrics(logits, masks)
            for class_name in vegetation_classes:
                if class_name in batch_metrics:
                    metrics_accumulator[class_name]["iou"].append(
                        batch_metrics[class_name]["iou"]
                    )
                    metrics_accumulator[class_name]["f1"].append(
                        batch_metrics[class_name]["f1"]
                    )

        avg_loss = total_loss / num_batches
        pbar.set_postfix({"loss": f"{avg_loss:.4f}", "batches": num_batches})

    avg_metrics = {}
    for class_name in vegetation_classes:
        if len(metrics_accumulator[class_name]["iou"]) > 0:
            avg_metrics[class_name] = {
                "iou": np.mean(metrics_accumulator[class_name]["iou"]),
                "f1": np.mean(metrics_accumulator[class_name]["f1"]),
            }

    return total_loss / max(num_batches, 1), avg_metrics


def validate_epoch(model, dataloader, criterion, device, epoch, track_classes=None):
    """Validate for one epoch.

    Args:
        model: FLAIR model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device to validate on
        epoch: Current epoch number
        track_classes: List of class names to track metrics for (default: all vegetation)

    Returns:
        Tuple of (average_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0
    num_batches = 0

    if track_classes is None:
        vegetation_classes = [c for c in SIMPLIFIED_CLASSES.values() if c != "else"]
    else:
        vegetation_classes = track_classes

    metrics_accumulator = {
        class_name: {"iou": [], "f1": [], "precision": [], "recall": []}
        for class_name in vegetation_classes
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1} [Val]")

    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(device)  # (B, 3, H, W)
            masks = batch["mask"].to(device)  # (B, H, W)

            tile_size = images.shape[-1]
            flair_batch = {
                "AERIAL_LABEL-COSIA": torch.zeros(
                    images.shape[0], tile_size, tile_size, device=device
                ),
                "AERIAL_RGBI": images,
            }

            logits_tasks, _ = model(flair_batch)
            logits = logits_tasks["AERIAL_LABEL-COSIA"]  # (B, num_classes, H, W)

            if logits.shape[-2:] != masks.shape[-2:]:
                logits = F.interpolate(
                    logits,
                    size=masks.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            loss = criterion(logits, masks)
            total_loss += loss.item()
            num_batches += 1

            batch_metrics = compute_metrics(logits, masks)
            for class_name in vegetation_classes:
                if class_name in batch_metrics:
                    for key in ["iou", "f1", "precision", "recall"]:
                        metrics_accumulator[class_name][key].append(
                            batch_metrics[class_name][key]
                        )

            avg_loss = total_loss / num_batches
            pbar.set_postfix({"loss": f"{avg_loss:.4f}", "batches": num_batches})

    avg_metrics = {}
    for class_name in vegetation_classes:
        if len(metrics_accumulator[class_name]["iou"]) > 0:
            avg_metrics[class_name] = {
                key: np.mean(metrics_accumulator[class_name][key])
                for key in ["iou", "f1", "precision", "recall"]
            }

    return total_loss / max(num_batches, 1), avg_metrics
