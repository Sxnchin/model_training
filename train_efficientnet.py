import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from efficientnet_pytorch import EfficientNet
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASET_DIR = "dataset"
MODEL_PATH = "model_efficientnet.pt"

train_tfms = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

val_tfms = transforms.Compose([
    transforms.Resize((380, 380)),
    transforms.ToTensor()
])

train_data = datasets.ImageFolder(f"{DATASET_DIR}/train", transform=train_tfms)
val_data = datasets.ImageFolder(f"{DATASET_DIR}/val", transform=val_tfms)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
val_loader = DataLoader(val_data, batch_size=16)

model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(10):
    model.train()
    loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/10 Training")
    for x, y in loop:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        preds = model(x)
        loss = criterion(preds, y)
        loss.backward()
        optimizer.step()
        loop.set_postfix(loss=loss.item())
    
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            preds = model(x)
            correct += (preds.argmax(1) == y).sum().item()
            total += y.size(0)
    
    acc = correct / total
    print(f"✅ Epoch {epoch+1}: Validation Accuracy = {acc:.4f}")
    torch.save(model.state_dict(), MODEL_PATH)

print(f"🎉 Training complete. Model saved → {MODEL_PATH}")
