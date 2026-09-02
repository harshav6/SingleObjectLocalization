import torch
import numpy as np

from torch.utils.data import Dataset, random_split, DataLoader, Sampler
from torchvision.datasets import VOCDetection, wrap_dataset_for_transforms_v2
from torchvision.models import MobileNet_V3_Large_Weights

from transforms import SingleClassPreprocess

VOC_CLASSES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]


class FilteredVOCDetection(Dataset):
    def __init__(self, voc: VOCDetection, category: str, transforms=None):
        self.voc = wrap_dataset_for_transforms_v2(voc)
        self.transforms = transforms

        category_idx = VOC_CLASSES.index(category) + 1

        self.indices = []
        self.labels = []

        for idx in range(len(self.voc)):
            _, target = self.voc[idx]

            labels = target["labels"]

            count = (labels == category_idx).sum().item()

            # Keep images containing zero or one target object.
            if count <= 1:
                self.indices.append(idx)
                self.labels.append(1 if count == 1 else 0)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, target = self.voc[self.indices[idx]]

        if self.transforms:
            image, target = self.transforms(image, target)

        return image, target


class BalancedSampler(Sampler):
    def __init__(self, labels, negative_ratio=1.0):
        labels = np.array(labels)

        self.positive_indices = np.where(labels == 1)[0]
        self.negative_indices = np.where(labels == 0)[0]

        self.pos = len(self.positive_indices)
        self.neg = int(round(self.pos * negative_ratio))

        assert self.pos > 0
        assert self.neg > 0

        if self.neg > len(self.negative_indices):
            self.neg = len(self.negative_indices)

    def __iter__(self):
        indices = np.empty(
            self.pos + self.neg,
            dtype=np.int64,
        )

        # Use every positive example.
        indices[: self.pos] = self.positive_indices

        # Randomly select negatives.
        indices[self.pos :] = np.random.choice(
            self.negative_indices,
            size=self.neg,
            replace=False,
        )

        np.random.shuffle(indices)

        return iter(indices.tolist())

    def __len__(self):
        return self.pos + self.neg


# --------------------------------------------------
# Dataset
# --------------------------------------------------

voc = VOCDetection(
    ".data/",
    download=False,
)

weights = MobileNet_V3_Large_Weights.DEFAULT
preprocess = weights.transforms()

transform = SingleClassPreprocess(
    category="horse",
    preprocess=preprocess,
)

dataset = FilteredVOCDetection(
    voc,
    category="horse",
    transforms=transform,
)


# --------------------------------------------------
# Train / validation split
# --------------------------------------------------

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size

train_dataset, val_dataset = random_split(
    dataset,
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)


# --------------------------------------------------
# Balanced training sampler
# --------------------------------------------------

train_labels = [dataset.labels[i] for i in train_dataset.indices]

train_sampler = BalancedSampler(
    train_labels,
    negative_ratio=3.0,
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    sampler=train_sampler,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
)
