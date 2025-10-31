import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = "efficientnet_humerus.pt"

# Load model architecture (same as train.py)
model = timm.create_model("efficientnet_b0", pretrained=False)
model.classifier = nn.Linear(model.classifier.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Preprocessing transforms (same as train.py)
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485], [0.229]),  # Same as train.py
])

def classify(image_path):
    img = Image.open(image_path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1)

    normal_prob = probs[0][0].item()
    osteo_prob = probs[0][1].item()

    label = "Normal" if normal_prob > osteo_prob else "Osteoporotic"

    print("\n🔍 Analysis Result")
    print(f"File: {image_path}")
    print(f"Prediction: **{label}**")
    print(f"Normal Probability: {normal_prob:.4f}")
    print(f"Osteoporotic Probability: {osteo_prob:.4f}\n")

if __name__ == "__main__":
    classify(r"C:\Users\sanch\Downloads\bone2.jpg")
