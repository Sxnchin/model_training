import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm
from pathlib import Path

DATA_DIR = Path("dataset_sorted")
MODEL_OUT = "efficientnet_humerus.pt"

BATCH_SIZE = 16
LR = 3e-4
EPOCHS = 12

train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(8),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])

train_dataset = datasets.ImageFolder(DATA_DIR, transform=train_transforms)
val_dataset = datasets.ImageFolder(DATA_DIR, transform=val_transforms)

val_split = int(0.2 * len(train_dataset))
train_split = len(train_dataset) - val_split

train_ds, val_ds = torch.utils.data.random_split(train_dataset, [train_split, val_split])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

model = timm.create_model("efficientnet_b0", pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

print("🚀 Training on:", device)
print(f"📂 Training images: {train_split}  |  Validation images: {val_split}")

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0

    loop = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", unit="batch")

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        loop.set_postfix(loss=f"{running_loss:.3f}")

    # Validation
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = torch.argmax(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    print(f"✅ Epoch {epoch} complete | Val Accuracy: {acc:.3f}")

torch.save(model.state_dict(), MODEL_OUT)
print(f"🎉 Model saved → {MODEL_OUT}")
