import torch

from model import SingleObjectModel
from loss import SingleObjectLoss
from dataset import train_loader, val_loader

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Device:", device)


# --------------------------------------------------
# Model, loss, optimizer
# --------------------------------------------------

model = SingleObjectModel().to(device)

criterion = SingleObjectLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
)

num_epochs = 10

best_val_iou = -1.0


# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(num_epochs):

    model.train()

    total_loss = 0.0
    total_class_loss = 0.0
    total_bbox_loss = 0.0

    for images, targets in train_loader:

        images = images.to(device)

        targets = {key: value.to(device) for key, value in targets.items()}

        class_logits, bbox_pred = model(images)

        loss, class_loss, bbox_loss = criterion(
            class_logits,
            bbox_pred,
            targets,
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        total_class_loss += class_loss.item()
        total_bbox_loss += bbox_loss.item()

    num_batches = len(train_loader)

    train_loss = total_loss / num_batches
    train_class_loss = total_class_loss / num_batches
    train_bbox_loss = total_bbox_loss / num_batches

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    model.eval()

    val_total_loss = 0.0
    val_class_loss = 0.0
    val_bbox_loss = 0.0

    total_iou = 0.0
    num_iou = 0

    with torch.no_grad():

        for images, targets in val_loader:

            images = images.to(device)

            targets = {key: value.to(device) for key, value in targets.items()}

            class_logits, bbox_pred = model(images)

            loss, class_loss, bbox_loss = criterion(
                class_logits,
                bbox_pred,
                targets,
            )

            val_total_loss += loss.item()
            val_class_loss += class_loss.item()
            val_bbox_loss += bbox_loss.item()

            # --------------------------------------------------
            # IoU
            # --------------------------------------------------

            labels = targets["labels"].float()

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

                predicted_area = (pred_box[2] - pred_box[0]) * (
                    pred_box[3] - pred_box[1]
                )

                true_area = (true_box[2] - true_box[0]) * (true_box[3] - true_box[1])

                union = predicted_area + true_area - intersection

                iou = intersection / union

                total_iou += iou.item()
                num_iou += 1

    num_val_batches = len(val_loader)

    val_loss = val_total_loss / num_val_batches
    val_class = val_class_loss / num_val_batches
    val_bbox = val_bbox_loss / num_val_batches

    val_mean_iou = total_iou / num_iou if num_iou > 0 else 0.0

    # --------------------------------------------------
    # Print results
    # --------------------------------------------------

    print(
        f"Epoch [{epoch + 1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Class: {train_class_loss:.4f} "
        f"BBox: {train_bbox_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} "
        f"Class: {val_class:.4f} "
        f"BBox: {val_bbox:.4f} "
        f"| Mean IoU: {val_mean_iou:.4f}"
    )

    # --------------------------------------------------
    # Save best IoU model
    # --------------------------------------------------

    if val_mean_iou > best_val_iou:

        best_val_iou = val_mean_iou

        torch.save(
            model.state_dict(),
            "single_object_model.pth",
        )

        print(f"  → Best model saved " f"(Mean IoU: {best_val_iou:.4f})")


print()
print("Training complete.")
print(f"Best validation Mean IoU: {best_val_iou:.4f}")
