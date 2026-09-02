import torch
import torch.nn as nn

from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
)


class SingleObjectModel(nn.Module):
    def __init__(self):
        super().__init__()

        weights = MobileNet_V3_Large_Weights.DEFAULT

        self.backbone = mobilenet_v3_large(weights=weights).features

        feature_dim = 960 * 7 * 7

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

        self.bbox_regressor = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 4),
        )

    def forward(self, x):
        features = self.backbone(x)

        features = torch.flatten(features, 1)

        class_logits = self.classifier(features)

        bbox = torch.sigmoid(self.bbox_regressor(features))

        return class_logits, bbox
