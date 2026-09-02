import torch
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F
from torchvision.transforms._presets import ImageClassification
from torchvision.tv_tensors import BoundingBoxes
from torchvision.tv_tensors._dataset_wrapper import VOC_DETECTION_CATEGORY_TO_IDX


class FilterBoundingBoxes(torch.nn.Module):
    def __init__(self, category: str):
        super().__init__()
        self.category_idx = VOC_DETECTION_CATEGORY_TO_IDX[category]

    def forward(self, target):
        boxes = target["boxes"]
        labels = target["labels"]

        mask = labels == self.category_idx

        if mask.any():
            boxes = BoundingBoxes(
                boxes[mask],
                format=boxes.format,
                canvas_size=boxes.canvas_size,
            )

            labels = torch.ones(
                boxes.shape[0],
                dtype=torch.int64,
                device=boxes.device,
            )

        else:
            boxes = BoundingBoxes(
                torch.zeros(
                    (1, 4),
                    dtype=boxes.dtype,
                    device=boxes.device,
                ),
                format=boxes.format,
                canvas_size=boxes.canvas_size,
            )

            labels = torch.zeros(
                1,
                dtype=torch.int64,
                device=boxes.device,
            )

        return {
            "boxes": boxes,
            "labels": labels,
        }


class ResizeBoundingBoxes(torch.nn.Module):
    def __init__(self, preprocess: ImageClassification):
        super().__init__()

        self.resize_size = preprocess.resize_size
        self.crop_size = preprocess.crop_size

    def forward(self, target):
        boxes = target["boxes"]

        boxes = F.resize(
            boxes,
            self.resize_size,
            antialias=True,
        )

        boxes = F.center_crop(
            boxes,
            self.crop_size,
        )

        return {
            "boxes": boxes,
            "labels": target["labels"],
        }


class NormalizeBoundingBoxes(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, target):
        boxes = target["boxes"]

        height, width = boxes.canvas_size

        boxes = boxes / torch.tensor(
            [width, height, width, height],
            dtype=boxes.dtype,
            device=boxes.device,
        )

        return {
            "boxes": boxes,
            "labels": target["labels"],
        }


class SingleClassPreprocess(torch.nn.Module):
    def __init__(
        self,
        category: str,
        preprocess: ImageClassification,
    ):
        super().__init__()

        self.transform = preprocess

        self.target_transform = v2.Compose(
            [
                FilterBoundingBoxes(category),
                ResizeBoundingBoxes(preprocess),
                NormalizeBoundingBoxes(),
            ]
        )

    def forward(self, image, target):
        image = self.transform(image)
        target = self.target_transform(target)

        return image, target
