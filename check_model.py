import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from pathlib import Path

DATA_DIR = Path("dataset")
MODEL_PATH = "efficientnet_humerus.pt"

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),
])

# Load validation dataset
val_dataset = datasets.ImageFolder(DATA_DIR / "val", transform=val_transforms)
val_loader = DataLoader(val_dataset, batch_size=16)

# Load model
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Evaluate
correct, total = 0, 0
class_correct = [0, 0]
class_total = [0, 0]

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(outputs, 1)
        
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        for i in range(len(labels)):
            label = labels[i].item()
            class_correct[label] += (preds[i] == labels[i]).item()
            class_total[label] += 1

overall_acc = correct / total
normal_acc = class_correct[0] / class_total[0] if class_total[0] > 0 else 0
osteo_acc = class_correct[1] / class_total[1] if class_total[1] > 0 else 0

print(f"\n{'='*50}")
print(f"MODEL PERFORMANCE SUMMARY")
print(f"{'='*50}")
print(f"Overall Validation Accuracy: {overall_acc*100:.2f}%")
print(f"Normal Class Accuracy:       {normal_acc*100:.2f}% ({class_correct[0]}/{class_total[0]})")
print(f"Osteoporotic Class Accuracy: {osteo_acc*100:.2f}% ({class_correct[1]}/{class_total[1]})")
print(f"{'='*50}\n")
