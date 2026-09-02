# Single Object Localization

A PyTorch implementation of single-object localization using the Pascal VOC 2012 dataset.

The model is trained to determine whether a target object is present in an image and, when present, predict its bounding box.

For this implementation, the target object is **horse**.

---

## Overview

This project focuses on **single-object localization**.

For each input image, the model produces two outputs:

1. **Classification** — determines whether the target object is present.
2. **Bounding box regression** — predicts the location of the target object.

The bounding box is represented as:

```text
(xmin, ymin, xmax, ymax)
```

### Architecture

```text
                    Input Image
                         │
                         ▼
                MobileNetV3-Large
                         │
                         ▼
                  Feature Extractor
                         │
                    960 × 7 × 7
                         │
                       Flatten
                         │
                    47,040 features
                         │
              ┌──────────┴──────────┐
              │                     │
              ▼                     ▼
       Classification        BBox Regression
            Head                   Head
              │                     │
              ▼                     ▼
          1 logit             4 coordinates
                              (x1,y1,x2,y2)
```

The MobileNetV3-Large feature extractor produces a spatial feature map of size `960 × 7 × 7` for a `224 × 224` input. This feature map is flattened before being passed to the two prediction heads.

---

## Dataset

The project uses the **Pascal VOC 2012** dataset.

For this implementation, the target category is **horse**.

Images are filtered so that they contain **at most one instance** of the target object.

The resulting dataset contains:

| Dataset          | Images |
| ---------------- | -----: |
| Original dataset |  5,717 |
| Filtered dataset |  5,635 |
| Training set     |  4,508 |
| Validation set   |  1,127 |

The training set contains:

* **129 positive** images containing a horse
* **4,379 negative** images without a horse

Since the dataset is highly imbalanced, a balanced sampler is used during training with a **3:1 negative-to-positive ratio**.

---

## Preprocessing

The input images are resized and center-cropped to:

```text
224 × 224
```

The bounding boxes undergo the same spatial transformations as the input images so that they remain aligned with the transformed images.

Bounding-box coordinates are then normalized to the range `[0, 1]`.

Each bounding box is represented as:

```text
(xmin, ymin, xmax, ymax)
```

For negative images, where no horse is present, the bounding box is represented as:

```text
[0, 0, 0, 0]
```

The images are also normalized using the standard ImageNet normalization used by the pretrained MobileNetV3-Large weights.

---

## Loss Function

The model uses two loss functions.

### Classification Loss

Binary cross entropy with logits is used for the classification task:

```text
BCEWithLogitsLoss
```

### Bounding Box Loss

Smooth L1 loss is used for bounding box regression:

```text
SmoothL1Loss
```

The bounding box loss is calculated only for positive images.

The total loss is:

```text
Total Loss = Classification Loss + Bounding Box Loss
```

---

## Training

The model is trained using:

| Parameter                  |             Value |
| -------------------------- | ----------------: |
| Backbone                   | MobileNetV3-Large |
| Optimizer                  |              Adam |
| Learning rate              |            `1e-4` |
| Batch size                 |              `32` |
| Epochs                     |              `10` |
| Negative-to-positive ratio |             `3:1` |

The MobileNetV3-Large backbone is initialized with pretrained ImageNet weights.

The model is trained end-to-end, with separate heads for classification and bounding box regression.

The best model checkpoint is selected based on validation **Mean IoU**.

---

## Evaluation

The model is evaluated using both classification and localization metrics.

### Classification Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

### Localization Metrics

* Mean IoU
* Percentage of predictions with IoU ≥ 0.50
* Percentage of predictions with IoU ≥ 0.75

Intersection over Union (IoU) measures the overlap between the predicted bounding box and the ground-truth bounding box.

---

## Results

The final model achieves the following results on the validation set.

### Classification

| Metric    |     Result |
| --------- | ---------: |
| Accuracy  | **98.85%** |
| Precision | **70.59%** |
| Recall    | **88.89%** |
| F1 Score  | **78.69%** |

### Confusion Matrix

|                 | Predicted Positive | Predicted Negative |
| --------------- | -----------------: | -----------------: |
| Actual Positive |                 24 |                  3 |
| Actual Negative |                 10 |              1,090 |

### Localization

| Metric     |     Result |
| ---------- | ---------: |
| Mean IoU   | **0.6042** |
| IoU ≥ 0.50 | **66.67%** |
| IoU ≥ 0.75 | **18.52%** |

---

## Visualization

The project includes a visualization script for inspecting the predicted bounding boxes.

For positive validation images, the visualization displays:

* Ground-truth bounding box
* Predicted bounding box
* Horse classification probability
* IoU between the predicted and ground-truth bounding boxes

Run:

```powershell
python src\visualize.py
```

This allows the localization performance to be inspected visually in addition to the numerical evaluation metrics.

---

## Project Structure

```text
SingleObjectLocalization/
│
├── .data/
│   └── VOCdevkit/
│
├── src/
│   ├── dataset.py
│   ├── transforms.py
│   ├── model.py
│   ├── loss.py
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
│
├── notebooks/
│
├── data/
├── checkpoints/
├── outputs/
│
├── README.md
├── requirements.txt
├── .gitignore
└── single_object_model.pth
```

The dataset, generated outputs, checkpoints, and model weights are excluded from version control through `.gitignore`.

---

## Installation

Clone the repository:

```powershell
git clone <repository-url>
cd SingleObjectLocalization
```

Create a virtual environment:

```powershell
python -m venv .venv
```

Activate the virtual environment on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Install the required dependencies:

```powershell
pip install -r requirements.txt
```

---

## Dataset Setup

Download the **Pascal VOC 2012** dataset.

Place the dataset under:

```text
.data/
```

The expected structure is:

```text
.data/
└── VOCdevkit/
    └── VOC2012/
        ├── Annotations/
        ├── ImageSets/
        ├── JPEGImages/
        ├── SegmentationClass/
        └── SegmentationObject/
```

The dataset is not included in the repository because of its size.

---

## Training

From the project root, run:

```powershell
python src\train.py
```

The training script:

1. Loads the Pascal VOC dataset.
2. Filters images based on the target category.
3. Applies image and bounding-box preprocessing.
4. Creates balanced training batches.
5. Trains the MobileNetV3-based localization model.
6. Evaluates the model on the validation set after each epoch.
7. Saves the checkpoint with the best validation Mean IoU.

The best model is saved as:

```text
single_object_model.pth
```

---

## Evaluation

After training, evaluate the saved model using:

```powershell
python src\evaluate.py
```

This reports:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Mean IoU
* IoU ≥ 0.50
* IoU ≥ 0.75

---

## Visualization

To visualize predictions on validation images:

```powershell
python src\visualize.py
```

The script loads the saved model and displays examples containing the target object.

---

## Requirements

The main dependencies are:

* Python
* PyTorch
* TorchVision
* NumPy
* Matplotlib
* Pillow
* tqdm

The exact installed versions are specified in:

```text
requirements.txt
```

---

## Notes

### Spatial Feature Extraction

The MobileNetV3-Large classification head is not used.

Instead, the model uses the spatial feature extractor:

```python
mobilenet_v3_large(weights=weights).features
```

For a `224 × 224` input, this produces:

```text
960 × 7 × 7
```

The resulting feature map is flattened:

```text
960 × 7 × 7 = 47,040
```

These features are then passed independently to the classification and bounding-box regression heads.

### Bounding Box Prediction

The bounding-box head outputs four normalized coordinates:

```text
(xmin, ymin, xmax, ymax)
```

A sigmoid activation keeps the predicted coordinates within the `[0, 1]` range.

### Negative Images

For images without a horse:

```text
label = 0
bbox = [0, 0, 0, 0]
```

The bounding-box loss is not applied to these negative samples.

---

## Future Work

This project focuses specifically on **single-object localization**.

Possible extensions include:

* Multi-object detection
* Multiple bounding boxes per image
* Different target categories
* More advanced detection architectures
* Non-Maximum Suppression
* Mean Average Precision (mAP)
* Faster R-CNN
* YOLO-style object detection
