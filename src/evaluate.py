import torch

from model import SingleObjectModel
from dataset import val_loader

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# Load model
# --------------------------------------------------

model = SingleObjectModel().to(device)

model.load_state_dict(
    torch.load(
        "single_object_model.pth",
        map_location=device,
    )
)

model.eval()


# --------------------------------------------------
# Classification metrics
# --------------------------------------------------

correct = 0
total = 0

true_positive = 0
false_positive = 0
false_negative = 0
true_negative = 0


# --------------------------------------------------
# Localization metrics
# --------------------------------------------------

total_iou = 0.0
num_iou = 0
iou_50 = 0
iou_75 = 0


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

with torch.no_grad():

    for images, targets in val_loader:

        images = images.to(device)

        targets = {key: value.to(device) for key, value in targets.items()}

        class_logits, bbox_pred = model(images)

        probabilities = torch.sigmoid(class_logits)

        predictions = (probabilities >= 0.5).float()

        labels = targets["labels"].float()

        # --------------------------------------------------
        # Classification
        # --------------------------------------------------

        correct += (predictions == labels).sum().item()

        total += labels.numel()

        true_positive += ((predictions == 1) & (labels == 1)).sum().item()

        false_positive += ((predictions == 1) & (labels == 0)).sum().item()

        false_negative += ((predictions == 0) & (labels == 1)).sum().item()

        true_negative += ((predictions == 0) & (labels == 0)).sum().item()

        # --------------------------------------------------
        # IoU
        # --------------------------------------------------

        target_boxes = targets["boxes"].squeeze(1)

        positive_mask = labels.squeeze(1) == 1

        predicted_boxes = bbox_pred[positive_mask]

        ground_truth_boxes = target_boxes[positive_mask]

        for pred_box, true_box in zip(
            predicted_boxes,
            ground_truth_boxes,
        ):

            x1 = torch.maximum(
                pred_box[0],
                true_box[0],
            )

            y1 = torch.maximum(
                pred_box[1],
                true_box[1],
            )

            x2 = torch.minimum(
                pred_box[2],
                true_box[2],
            )

            y2 = torch.minimum(
                pred_box[3],
                true_box[3],
            )

            intersection_width = torch.clamp(
                x2 - x1,
                min=0,
            )

            intersection_height = torch.clamp(
                y2 - y1,
                min=0,
            )

            intersection = intersection_width * intersection_height

            predicted_area = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])

            true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])

            union = predicted_area + true_area - intersection

            iou = intersection / union

            total_iou += iou.item()
            num_iou += 1

            if iou >= 0.50:
                iou_50 += 1

            if iou >= 0.75:
                iou_75 += 1


# --------------------------------------------------
# Calculate metrics
# --------------------------------------------------

accuracy = correct / total

precision = (
    true_positive / (true_positive + false_positive)
    if (true_positive + false_positive) > 0
    else 0.0
)

recall = (
    true_positive / (true_positive + false_negative)
    if (true_positive + false_negative) > 0
    else 0.0
)

f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

mean_iou = total_iou / num_iou if num_iou > 0 else 0.0

iou_50_percentage = iou_50 / num_iou * 100 if num_iou > 0 else 0.0

iou_75_percentage = iou_75 / num_iou * 100 if num_iou > 0 else 0.0


# --------------------------------------------------
# Print results
# --------------------------------------------------

print()
print("Classification")
print("--------------")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

print()
print("Confusion Matrix")
print("----------------")
print(f"True Positive:  {true_positive}")
print(f"False Positive: {false_positive}")
print(f"False Negative: {false_negative}")
print(f"True Negative:  {true_negative}")

print()
print("Localization")
print("------------")
print(f"Mean IoU:      {mean_iou:.4f}")
print(f"IoU >= 0.50:   {iou_50_percentage:.2f}%")
print(f"IoU >= 0.75:   {iou_75_percentage:.2f}%")
