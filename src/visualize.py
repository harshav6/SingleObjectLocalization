import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from model import SingleObjectModel
from dataset import val_dataset

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# ImageNet normalization values
# --------------------------------------------------

mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


# --------------------------------------------------
# Visualization
# --------------------------------------------------

num_images = 5
shown = 0

for i in range(len(val_dataset)):

    image, target = val_dataset[i]

    # Only visualize positive examples.
    if target["labels"].item() != 1:
        continue

    image_input = image.unsqueeze(0).to(device)

    with torch.no_grad():
        class_logits, bbox_pred = model(image_input)

    probability = torch.sigmoid(class_logits).item()

    predicted_box = bbox_pred[0].cpu()
    true_box = target["boxes"][0].cpu()

    # --------------------------------------------------
    # IoU
    # --------------------------------------------------

    x1 = torch.maximum(
        predicted_box[0],
        true_box[0],
    )

    y1 = torch.maximum(
        predicted_box[1],
        true_box[1],
    )

    x2 = torch.minimum(
        predicted_box[2],
        true_box[2],
    )

    y2 = torch.minimum(
        predicted_box[3],
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

    predicted_area = (predicted_box[2] - predicted_box[0]) * (
        predicted_box[3] - predicted_box[1]
    )

    true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])

    union = predicted_area + true_area - intersection

    iou = (intersection / union).item()

    # --------------------------------------------------
    # Denormalize image for display
    # --------------------------------------------------

    image_display = image * std + mean
    image_display = image_display.clamp(0, 1)

    image_np = image_display.permute(1, 2, 0).numpy()

    # --------------------------------------------------
    # Draw image
    # --------------------------------------------------

    fig, ax = plt.subplots()

    ax.imshow(image_np)

    # Ground-truth box
    x1, y1, x2, y2 = true_box.numpy()

    ax.add_patch(
        Rectangle(
            (x1 * 224, y1 * 224),
            (x2 - x1) * 224,
            (y2 - y1) * 224,
            fill=False,
            linewidth=2,
            label="Ground Truth",
        )
    )

    # Predicted box
    x1, y1, x2, y2 = predicted_box.numpy()

    ax.add_patch(
        Rectangle(
            (x1 * 224, y1 * 224),
            (x2 - x1) * 224,
            (y2 - y1) * 224,
            fill=False,
            linewidth=2,
            linestyle="--",
            label="Prediction",
        )
    )

    ax.set_title(f"Horse probability: {probability:.2f} | " f"IoU: {iou:.2f}")

    ax.legend()

    plt.show()

    shown += 1

    if shown == num_images:
        break
