import torch
import torch.nn as nn


class SingleObjectLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.classification_loss = nn.BCEWithLogitsLoss()
        self.bbox_loss = nn.SmoothL1Loss()

    def forward(self, class_logits, bbox_pred, targets):
        labels = targets["labels"].float()
        boxes = targets["boxes"].squeeze(1)

        # Classification loss
        class_loss = self.classification_loss(
            class_logits,
            labels,
        )

        # Bounding-box loss only for positive samples
        positive_mask = labels.squeeze(1) == 1

        if positive_mask.any():
            bbox_loss = self.bbox_loss(
                bbox_pred[positive_mask],
                boxes[positive_mask],
            )
        else:
            bbox_loss = torch.tensor(
                0.0,
                device=bbox_pred.device,
            )

        # Total loss
        total_loss = class_loss + bbox_loss

        return total_loss, class_loss, bbox_loss
