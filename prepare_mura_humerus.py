import shutil
from pathlib import Path
import random

random.seed(42)

RAW_ROOT = Path(r"C:\Users\sanch\Downloads\MURA-v1.1\train\XR_HUMERUS")
OUT_ROOT = Path("dataset")

TRAIN_NORMAL = OUT_ROOT / "train" / "normal"
TRAIN_OSTEO = OUT_ROOT / "train" / "osteoporotic"
VAL_NORMAL = OUT_ROOT / "val" / "normal"
VAL_OSTEO = OUT_ROOT / "val" / "osteoporotic"

for p in [TRAIN_NORMAL, TRAIN_OSTEO, VAL_NORMAL, VAL_OSTEO]:
    p.mkdir(parents=True, exist_ok=True)

all_images = []

for patient_dir in RAW_ROOT.iterdir():
    for study_dir in patient_dir.iterdir():
        if "negative" in study_dir.name.lower():
            label = "normal"
        elif "positive" in study_dir.name.lower():
            label = "osteoporotic"
        else:
            continue
        
        for img in study_dir.glob("*.png"):
            all_images.append((img, label, f"{patient_dir.name}_{study_dir.name}_{img.name}"))

random.shuffle(all_images)

split = int(len(all_images) * 0.8)
train = all_images[:split]
val = all_images[split:]

for img_path, label, name in train:
    shutil.copy2(img_path, (TRAIN_NORMAL if label == "normal" else TRAIN_OSTEO) / name)

for img_path, label, name in val:
    shutil.copy2(img_path, (VAL_NORMAL if label == "normal" else VAL_OSTEO) / name)

print("✅ Dataset successfully prepared at ./dataset")
