# 🦴 Osteoporosis X‑Ray Analyzer — Science Fair Edition

A student-friendly project that demonstrates how computer vision (AI) and simple mechanical measurements can work together to estimate bone weakening on humerus X‑rays. It runs as a small web app on your laptop—no command line needed.

## What this project does

- Classifies an input X‑ray as “Normal” or “Osteoporotic” using a pre‑trained EfficientNet‑B0 model
- Combines the AI score with lab measurements (fracture force and propagation speed) to compute a Severity score (0–1)
- Converts the score into an easy category: Normal, Mild, Moderate, Severe
- Shows a friendly recommendation message and lets you download a text report

> Educational use only. This is NOT a medical device and must not be used for diagnosis or treatment decisions.

---

## Materials required (for demo at the fair)

- A Windows laptop (8 GB RAM or more; GPU optional)
- This project folder
- Python 3.10+ installed
- A couple hundred sample humerus X‑ray images (JPG/PNG)
- The bundled CSV with lab test values: `humpty dumpty is humping my leg.csv`

Optional (for advanced experimentation):
- NVIDIA GPU with CUDA for faster inference (not required)
- Access to the MURA dataset if you want to reproduce training

---

## Software dependencies

The app uses these Python packages (see `requirements.txt`):

- torch
- torchvision
- Pillow
- timm
- streamlit

Install them once:

```powershell
pip install -r requirements.txt
```

---

## Training data and model

- Base dataset: MURA (Musculoskeletal Radiographs) — publicly available research dataset by Stanford ML Group.
- Subset used: Humerus X‑rays. Studies labeled “positive” (abnormal) vs “negative” (normal). For this demo, “positive” is treated as osteoporotic, which is a simplification for learning purposes.
- Model: EfficientNet‑B0 fine‑tuned for 2 classes (Normal vs Osteoporotic). The trained weights are already included as `efficientnet_humerus.pt`.

> Limitation: MURA’s “abnormal” label is not a clinical osteoporosis label. This project demonstrates technique, not clinical diagnostic accuracy.

---

## How it works (under the hood)

1. Image analysis (AI):
   - Preprocess the X‑ray to 224×224, normalize, and run it through EfficientNet‑B0.
   - Output probabilities: Normal and Osteoporotic.
2. Lab test features (from CSV):
   - Breaking Point (lbs) → converted to Newtons: `1 lbf = 4.4482216153 N`.
   - Fracture Propagation Speed (m/s) → converted to `mm/s`.
   - The app computes reference values from the CSV:
     - `F_REF = mean(failure_load_n)`
     - `V_REF = mean(prop_speed_mm_s)`
     - `V_MAX = max(prop_speed_mm_s)` (with a small safety headroom)
3. Severity score (no re‑training needed):

   ```text
   S_F = clamp(failure_load_n / F_REF, 0, 1)
   S_v = clamp((prop_speed_mm_s − V_REF) / (V_MAX − V_REF), 0, 1)
   severity = 0.45 * model_prob_osteo + 0.35 * (1 − S_F) + 0.20 * S_v
   ```

4. Category + recommendation:
   - Normal (< 0.30) — Mild (0.30–0.55) — Moderate (0.55–0.75) — Severe (≥ 0.75)
   - Human‑readable message appears with a soft severity‑colored card.

---

## How to run the app (no command line required)

Option 1 — Double‑click

- Double‑click `run_app.bat`
- Your browser opens to `http://localhost:8501`

Option 2 — From PowerShell

```powershell
pip install -r requirements.txt   # only once
streamlit run app.py
```

In the app:

- In the sidebar: select a Trial from the bundled CSV. The app calculates reference values and fills in the trial’s measurements (with friendly unit conversions).
- In the main panel: upload an X‑ray image and click “Analyze X‑Ray”.
- You’ll see prediction, probabilities, a severity category card, and a message. You can download a text report.

Stopping the app:

- Press `Ctrl + C` in the Streamlit terminal window to stop the server.

---

## What it can and cannot do

What it can do
- Demonstrate an end‑to‑end AI pipeline for medical‑style images
- Show how to combine model probabilities with mechanical measurements
- Convert real‑world units (lb→N, m/s→mm/s) and normalize against references
- Provide a simple severity score and friendly recommendation message

What it cannot do
- Provide a medical diagnosis (it’s for learning only)
- Guarantee accuracy on all bones/images/settings
- Replace qualified clinical assessment or lab testing
- Infer personalized risk without proper clinical context

---

## Reproduce training (optional)

If you want to explore the dataset and training flow:

1. Prepare sorted data from MURA Humerus studies (positive/negative):
   - Edit `RAW_ROOT` in `sort_mura_humerus.py` to your MURA path
   - Run it to populate `dataset_sorted/normal` and `dataset_sorted/osteoporotic`
2. Train EfficientNet‑B0 on your machine:
   - Run `python train.py` (tweaks: batch size, epochs, LR)
   - The model is saved as `efficientnet_humerus.pt`

> Training is optional; the app already includes a trained model.

---

## Ethics and safety
- Do not use this tool for medical decisions.
---

## Credits

- MURA Dataset — Stanford ML Group
- EfficientNet — Tan & Le
- Streamlit — the web UI framework used here
