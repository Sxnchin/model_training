
import shutil
from pathlib import Path
import torch
from torchvision import transforms, models
from PIL import Image

MODEL_PATH = Path('model.pt')
INPUT_DIR = Path('input_images')
OUT_NORMAL = Path('classified/normal')
OUT_OSTEO = Path('classified/osteoporotic')

for d in [OUT_NORMAL, OUT_OSTEO]:
    d.mkdir(parents=True, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
model.eval()

for img_file in INPUT_DIR.glob('*.*'):
    try:
        img = Image.open(img_file).convert('RGB')
    except:
        continue
    x = transform(img).unsqueeze(0)
    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(1).item()
    shutil.copy2(img_file, OUT_NORMAL if pred == 0 else OUT_OSTEO)

print("✅ Classification done.")
