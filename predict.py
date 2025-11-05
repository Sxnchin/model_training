import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn
import argparse
import csv
import os
import re
from typing import Optional, List, Tuple, Dict

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

def compute_severity(model_prob_osteo, failure_load_n, prop_speed_mm_s, 
                     F_REF=300, V_REF=60, V_MAX=500):
    
    # Load score (weaker bones fracture at lower force)
    S_F = max(0, min(1, failure_load_n / F_REF))

    # Propagation speed score (higher speed = more brittle)
    S_v = max(0, min(1, (prop_speed_mm_s - V_REF) / (V_MAX - V_REF)))

    severity = 0.45 * model_prob_osteo + 0.35 * (1 - S_F) + 0.20 * S_v
    return severity

def bone_message(severity, failure_load_n, prop_speed_mm_s):
    if severity < 0.30:
        category = "Normal"
        msg = "No significant weakening detected. Maintain healthy bone habits."
    elif severity < 0.55:
        category = "Mild"
        msg = (f"While not a medical diagnosis, the bone shows mild weakening. "
               f"Expected fracture force ≈ {failure_load_n} N. "
               f"Consult a healthcare provider if symptoms or risk factors exist.")
    elif severity < 0.75:
        category = "Moderate"
        msg = (f"Bone shows moderate structural weakening. "
               f"Fracture may occur around {failure_load_n} N. "
               f"Propagation speed ({prop_speed_mm_s} mm/s) suggests reduced toughness.")
    else:
        category = "Severe"
        msg = (f"Bone shows **severe** weakening. Expected fracture load ≈ {failure_load_n} N. "
               f"Propagation speed ({prop_speed_mm_s} mm/s) indicates brittle failure. "
               f"Seek medical evaluation promptly.")

    return category, msg

def classify(image_path, failure_load_n=180, prop_speed_mm_s=90, *, F_REF=300, V_REF=60, V_MAX=500):
    img = Image.open(image_path).convert("RGB")
    x = tfms(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(x)
        probs = torch.softmax(preds, dim=1)

    normal_prob = probs[0][0].item()
    osteo_prob = probs[0][1].item()

    label = "Normal" if normal_prob > osteo_prob else "Osteoporotic"

    print(f"\n🔍 {image_path} → **{label}**")
    print(f"Normal={normal_prob:.2f}, Osteoporotic={osteo_prob:.2f}")

    # Compute severity score
    severity = compute_severity(osteo_prob, failure_load_n, prop_speed_mm_s, F_REF=F_REF, V_REF=V_REF, V_MAX=V_MAX)
    category, message = bone_message(severity, failure_load_n, prop_speed_mm_s)

    print(f"\n🦴 Classification: {category} (severity={severity:.2f})")
    print(f"💬 {message}\n")


# ------------------------ CSV utilities ------------------------
def _parse_breaking_point_to_newtons(value: str) -> Optional[float]:
    """Parse strings like '1.98 Lbs' or '.35 Lbs' or 'NULL' to Newtons (N). 1 lbf = 4.4482216153 N.
    Returns None if value cannot be parsed.
    """
    if not value or str(value).strip().upper() == "NULL":
        return None
    s = str(value)
    # Extract numeric part (handles leading dot like '.35')
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
    if not m:
        return None
    lbs = float(m.group(0))
    return lbs * 4.4482216153


def _parse_speed_to_mm_per_s(value: str) -> Optional[float]:
    """Parse strings like '.37 M/s' to mm/s. Returns None if not parsable."""
    if not value or str(value).strip().upper() == "NULL":
        return None
    s = str(value)
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
    if not m:
        return None
    m_per_s = float(m.group(0))
    return m_per_s * 1000.0  # to mm/s


def load_trials_from_csv(csv_path: str):
    """Load trials from CSV and compute reference values from available rows.
    Returns (rows, F_REF, V_REF, V_MAX) where rows is list of dicts with
    keys: trial, failure_load_n, prop_speed_mm_s.
    """
    rows = []
    failure_vals = []
    speed_vals = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            trial_str = r.get("Trial")
            try:
                trial = int(str(trial_str).strip()) if trial_str is not None else None
            except ValueError:
                trial = None

            bp_n = _parse_breaking_point_to_newtons(r.get("Breaking Point"))
            sp_mm_s = _parse_speed_to_mm_per_s(r.get("Fracture Propogation Speed"))

            if bp_n is not None:
                failure_vals.append(bp_n)
            if sp_mm_s is not None:
                speed_vals.append(sp_mm_s)

            rows.append({
                "trial": trial,
                "failure_load_n": bp_n,
                "prop_speed_mm_s": sp_mm_s,
            })

    # Compute references from available values; fall back to defaults if empty
    F_REF = sum(failure_vals) / len(failure_vals) if failure_vals else 300.0
    V_REF = sum(speed_vals) / len(speed_vals) if speed_vals else 60.0
    if speed_vals:
        vmax_raw = max(speed_vals)
        # add a small headroom to avoid division-by-zero when V_MAX ~ V_REF
        V_MAX = max(vmax_raw, V_REF + 1.0)
    else:
        V_MAX = 500.0

    return rows, F_REF, V_REF, V_MAX


def classify_from_csv(csv_path: str, image: Optional[str] = None, image_template: Optional[str] = None,
                      override_F_REF: Optional[float] = None, override_V_REF: Optional[float] = None, override_V_MAX: Optional[float] = None,
                      only_trial: Optional[int] = None):
    """Batch classify using CSV values. If image_template includes '{trial}', it'll be formatted per row.
    If only a single image path is provided (image), it will be reused for all rows.
    """
    rows, F_REF, V_REF, V_MAX = load_trials_from_csv(csv_path)
    if override_F_REF is not None:
        F_REF = override_F_REF
    if override_V_REF is not None:
        V_REF = override_V_REF
    if override_V_MAX is not None:
        V_MAX = override_V_MAX

    print(f"\n📊 Reference values from CSV: F_REF={F_REF:.2f} N, V_REF={V_REF:.2f} mm/s, V_MAX={V_MAX:.2f} mm/s")

    def image_for_trial(trial: int):
        if image_template and "{trial}" in image_template:
            return image_template.format(trial=trial)
        return image

    any_done = False
    for r in rows:
        trial = r["trial"]
        if only_trial is not None and trial != only_trial:
            continue

        f_n = r["failure_load_n"]
        v_mm_s = r["prop_speed_mm_s"]

        # Impute missing values with reference so they don't bias the score undesirably
        if f_n is None:
            f_n = F_REF
        if v_mm_s is None:
            v_mm_s = V_REF

        img_path = image_for_trial(trial) if trial is not None else image
        if not img_path:
            print(f"⚠️ Skipping trial {trial}: no image path provided.")
            continue
        if not os.path.exists(img_path):
            print(f"⚠️ Image not found for trial {trial}: {img_path}")
            continue

        print(f"\n—— Trial {trial} ——")
        classify(img_path, f_n, v_mm_s, F_REF=F_REF, V_REF=V_REF, V_MAX=V_MAX)
        any_done = True

    if not any_done:
        print("No rows classified. Check --trial, image paths, or CSV contents.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify bone image and compute severity with optional CSV-based refs.")
    parser.add_argument("-i", "--image", type=str, help="Path to image (or single image to reuse). If contains '{trial}', use --image-template instead.")
    parser.add_argument("--image-template", type=str, help="Image template with {trial} placeholder, e.g., C:\\imgs\\bone_{trial}.jpg")
    parser.add_argument("--csv", type=str, help="Path to CSV with trial data.")
    parser.add_argument("--trial", type=int, help="Only process a specific trial number from the CSV.")
    parser.add_argument("--failure-load-n", type=float, help="Manual failure load in Newtons (if not using CSV).")
    parser.add_argument("--prop-speed-mm-s", type=float, help="Manual propagation speed in mm/s (if not using CSV).")
    parser.add_argument("--F_REF", type=float, help="Override reference failure load (N). If not set and CSV provided, computed from CSV.")
    parser.add_argument("--V_REF", type=float, help="Override reference propagation speed (mm/s). If not set and CSV provided, computed from CSV.")
    parser.add_argument("--V_MAX", type=float, help="Override max propagation speed (mm/s). If not set and CSV provided, computed from CSV max.")

    args = parser.parse_args()

    # CSV-driven mode
    if args.csv:
        classify_from_csv(
            csv_path=args.csv,
            image=args.image,
            image_template=args.image_template or args.image,
            override_F_REF=args.F_REF,
            override_V_REF=args.V_REF,
            override_V_MAX=args.V_MAX,
            only_trial=args.trial,
        )
    else:
        # Simple single example mode (fallback)
        image_path = args.image or r"C:\\Users\\sanch\\Downloads\\bone2.jpg"
        failure_load_n = args.failure_load_n if args.failure_load_n is not None else 180.0
        prop_speed_mm_s = args.prop_speed_mm_s if args.prop_speed_mm_s is not None else 90.0
        F_REF = args.F_REF if args.F_REF is not None else 300.0
        V_REF = args.V_REF if args.V_REF is not None else 60.0
        V_MAX = args.V_MAX if args.V_MAX is not None else 500.0

        classify(image_path, failure_load_n, prop_speed_mm_s, F_REF=F_REF, V_REF=V_REF, V_MAX=V_MAX)
