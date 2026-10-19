"""Loss functions: CrossEntropy (with label smoothing), FocalLoss."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    FL(pt) = -alpha_t * (1 - pt)^gamma * log(pt)
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


def create_loss(
    loss_name: str = "cross_entropy",
    label_smoothing: float = 0.0,
    class_weights: torch.Tensor | None = None,
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    """Create a loss function by name.

    Args:
        loss_name: "cross_entropy" or "focal"
        label_smoothing: label smoothing factor for CrossEntropyLoss
        class_weights: per-class weights tensor for CrossEntropyLoss
        focal_alpha: alpha for FocalLoss
        focal_gamma: gamma for FocalLoss
    """
    loss_name = loss_name.lower().strip()

    if loss_name in ("cross_entropy", "ce"):
        return nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=label_smoothing,
        )

    if loss_name in ("focal", "focal_loss"):
        return FocalLoss(alpha=focal_alpha, gamma=focal_gamma)

    raise ValueError(f"Unknown loss: {loss_name}. Choices: cross_entropy, focal")
