
from pathlib import Path
import json
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORT = ROOT / "report"

CLASSES = ["blackspot", "canker", "fresh", "grenning"]
IMAGE_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(
            brightness=0.15, contrast=0.15, saturation=0.15
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225],
        ),
    ])

    train_data = datasets.ImageFolder(
        DATA / "train", transform=train_transform
    )
    val_data = datasets.ImageFolder(
        DATA / "val", transform=eval_transform
    )
    test_data = datasets.ImageFolder(
        DATA / "test", transform=eval_transform
    )

    if train_data.classes != CLASSES:
        raise ValueError(
            f"Unexpected class order: {train_data.classes}"
        )

    train_loader = DataLoader(
        train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )
    test_loader = DataLoader(
        test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    print("Device:", DEVICE)
    print("Class mapping:", train_data.class_to_idx)
    print(
        f"Images: train={len(train_data)}, "
        f"val={len(val_data)}, test={len(test_data)}"
    )

    # Use pretrained MobileNetV2 weights.
    # First run may need internet to download the weights.
    try:
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)
    except Exception as exc:
        raise RuntimeError(
            "Could not download pretrained MobileNetV2 weights. "
            "Check internet and run again."
        ) from exc

    # Freeze pretrained feature extractor; train classification head.
    for param in model.features.parameters():
        param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, len(CLASSES))
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier.parameters(), lr=0.001
    )

    best_val_accuracy = 0.0
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            predictions = outputs.argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_accuracy = correct / max(total, 1)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                val_correct += (predictions == labels).sum().item()
                val_total += labels.size(0)

        val_accuracy = val_correct / max(val_total, 1)

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"loss={train_loss:.4f} | "
            f"train_acc={train_accuracy:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()

            all_predictions.extend(predictions.tolist())
            all_labels.extend(labels.numpy().tolist())

    y_true = np.asarray(all_labels)
    y_pred = np.asarray(all_predictions)

    results = {
        "model": "MobileNetV2",
        "split": "test",
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(
            precision_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "recall_macro": float(
            recall_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "f1_macro": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "best_validation_accuracy": float(best_val_accuracy),
        "epochs": EPOCHS,
        "device": str(DEVICE),
    }

    print("\nMobileNetV2 — OFFICIAL TEST")
    print(json.dumps(results, indent=2))
    print(
        classification_report(
            y_true, y_pred, labels=list(range(len(CLASSES))),
            target_names=CLASSES, zero_division=0
        )
    )

    cm = confusion_matrix(
        y_true, y_pred, labels=list(range(len(CLASSES)))
    )
    np.savetxt(
        REPORT / "mobilenetv2_confusion_matrix.csv",
        cm, delimiter=",", fmt="%d"
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": CLASSES,
            "image_size": IMAGE_SIZE,
            "best_validation_accuracy": best_val_accuracy,
        },
        MODELS / "mobilenetv2.pth",
    )

    with open(REPORT / "mobilenetv2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nMobileNetV2 training complete.")
    print("Model:", MODELS / "mobilenetv2.pth")
    print("Results:", REPORT / "mobilenetv2_results.json")


if __name__ == "__main__":
    main()
