# 📊 Model Analysis & Visualization Guide

Your bone osteoporosis detection model now includes **complete visualization and analysis tools** generating 10 comprehensive reports!

## 🚀 Quick Start

### Step 1: Train Your Model (or use existing model)
```bash
python train.py
```
This saves `efficientnet_humerus.pt` and `training_history.json`

### Step 2: Run Complete Analysis
```bash
python analyze_model.py
```

All visualizations are saved to `analysis_results/` folder.

---

## 📊 Complete Visualization Suite

### **Essential Visualizations** ✅

#### 1️⃣ **Confusion Matrix (2×2)**
- **File:** `01_confusion_matrix_training.png` & `01_confusion_matrix_validation.png`
- **What it shows:** True Positives, False Positives, False Negatives, True Negatives
- **Use case:** Understand prediction accuracy and error types
```
              Normal  Osteoporotic
Normal           TP       FP
Osteoporotic     FN       TN
```

#### 2️⃣ **ROC Curves with AUC**
- **File:** `02_roc_curves.png`
- **What it shows:** Training and Validation AUC scores
- **Use case:** Evaluate model discrimination ability
- **Key metric:** Higher AUC = Better classifier (perfect: 1.0, random: 0.5)

#### 3️⃣ **Loss Curves Over Epochs**
- **File:** `03_loss_curves.png`
- **What it shows:** Training loss + Validation accuracy over 12+ epochs
- **Use case:** Detect overfitting, underfitting, convergence
- **Interpretation:**
  - Decreasing loss = Good
  - Diverging lines = Overfitting
  - Flat loss = Learning rate too low

#### 4️⃣ **Feature Importance Bar Chart**
- **File:** `04_feature_importance.png`
- **What it shows:** Top 20 most important convolutional filters
- **Use case:** Understand what features the model learned
- **Interpretation:** Taller bars = More important features for classification

#### 5️⃣ **Breaking Force Analysis**
- **File:** `05_breaking_force.png`
- **What it shows:** Breaking force (Newtons) by bone type with error bars
- **Use case:** Correlate mechanical properties with bone quality
- **Features:**
  - Error bars show ± standard deviation
  - Significance brackets (*) show meaningful differences
  - Groups by Bone Type (A, B, C, D)

#### 6️⃣ **Porosity vs Strength Correlation**
- **File:** `06_porosity_strength.png`
- **What it shows:** Scatter plot with linear regression fit
- **Key metric:** R² value indicates correlation strength
  - R² = 1.0 → Perfect correlation
  - R² = 0.5 → Moderate correlation
  - R² = 0.0 → No correlation
- **Use case:** Validate inverse relationship between porosity and strength

---

### **Advanced Visualizations** 🚀

#### 7️⃣ **Grad-CAM Heatmap** (HUGE competitive edge!)
- **File:** `07_grad_cam_heatmap.png`
- **What it shows:** WHERE the CNN looks in X-ray images
- **Why it matters:**
  - Shows model interpretability (doctors can validate)
  - Identifies if model uses clinically relevant regions
  - Red = High attention, Blue = Low attention
- **Example:** Grad-CAM shows if model focuses on bone cortex (good) or noise (bad)
- **3 columns per sample:**
  1. Original X-ray
  2. Attention overlay on image
  3. Pure heatmap

#### 8️⃣ **Crack Velocity Distribution Box Plots**
- **File:** `08_crack_velocity.png`
- **What it shows:** 
  - Box plot: quartiles, median, outliers
  - Violin plot: full distribution shape
- **Use case:** Compare crack propagation speeds across bone types
- **Interpretation:**
  - Tall boxes = High variability
  - Outliers = Anomalous samples
  - Median line shows typical value

#### 9️⃣ **Network Architecture Diagram**
- **File:** `09_network_architecture.png`
- **What it shows:** Visual flowchart of EfficientNet-B0
- **Layers visualized:**
  - Input (224×224×3)
  - 8 MobileNet-style blocks
  - Output classifier (2 classes)
- **Use case:** Presentation & documentation

#### 🔟 **Feature Extraction Overlay**
- **File:** `10_feature_extraction.png`
- **What it shows:** 
  - Original X-ray images
  - Edge detection (Sobel operator)
  - Feature boundaries
- **Use case:** Understand preprocessing and low-level features

---

## 📁 Output Files Structure

```
analysis_results/
├── 01_confusion_matrix_training.png      ✅ TP/FP/FN/TN counts
├── 01_confusion_matrix_validation.png    ✅ Validation set metrics
├── 02_roc_curves.png                     ✅ AUC scores
├── 03_loss_curves.png                    ✅ Training history
├── 04_feature_importance.png             ✅ Top features
├── 05_breaking_force.png                 ✅ Mechanical properties
├── 06_porosity_strength.png              ✅ Correlation (R²)
├── 07_grad_cam_heatmap.png               🚀 Model attention
├── 08_crack_velocity.png                 🚀 Distribution analysis
├── 09_network_architecture.png           🚀 Network diagram
└── 10_feature_extraction.png             🚀 Feature visualization
```

---

## 🎯 How to Interpret Each Chart

### Confusion Matrix - What's Good?
✅ **Good:** High diagonal values (TP & TN)  
❌ **Bad:** High off-diagonal (FP & FN)

### ROC Curve - What's Good?
✅ **Good:** AUC > 0.85 (curves towards top-left)  
⚠️ **Fair:** AUC = 0.7-0.85  
❌ **Bad:** AUC < 0.7

### Loss Curves - What's Good?
✅ **Good:** Loss decreases, validation accuracy increases  
⚠️ **Overfit:** Training loss ↓ but validation accuracy plateaus  
❌ **Underfit:** Both losses stay high

### Feature Importance - What's Good?
✅ **Good:** Clear differences in feature heights  
⚠️ **Questionable:** All features equally important (model confused?)

### Grad-CAM - What's Good?
✅ **Good:** Heatmap focuses on bone region, not background  
❌ **Bad:** Heatmap focuses on random areas or image edges

---

## 🔧 Customization Options

### Modify Confidence Thresholds
Edit `analyze_model.py` line ~400:
```python
F_REF = 300  # Reference force (Newtons)
V_REF = 60   # Reference velocity (mm/s)
V_MAX = 500  # Maximum velocity
```

### Change Grad-CAM Target Layer
Edit line ~550:
```python
target_layer = model.blocks[-1][-1].conv_pwl  # Last conv layer
```

### Adjust Plot Styles
Edit top of `analyze_model.py`:
```python
sns.set_style("darkgrid")  # Change to "whitegrid", "dark", etc.
plt.rcParams['figure.dpi'] = 200  # Higher resolution
```

---

## 📈 Typical Values for Good Models

| Metric | Good | Excellent |
|--------|------|-----------|
| Training Confusion Matrix Accuracy | >85% | >95% |
| Validation Confusion Matrix Accuracy | >80% | >90% |
| Training AUC | >0.85 | >0.95 |
| Validation AUC | >0.80 | >0.90% |
| Feature Top-5 Importance Sum | >40% | >50% |
| Porosity-Strength R² | >0.5 | >0.75 |

---

## 🐛 Troubleshooting

### "training_history.json not found"
→ Run `python train.py` first to generate training history

### "testdata.csv not found"
→ Breaking force and velocity plots use synthetic data; use real CSV for accuracy

### "Not enough samples for Grad-CAM"
→ Ensure `dataset_sorted/` has both `normal/` and `osteoporotic/` folders with images

### Memory issues with Grad-CAM?
→ Reduce batch size or process fewer samples:
```python
sample_indices = sample_indices[:1]  # Use 1 sample instead of 2
```

---

## 💡 Pro Tips

1. **For presentations:** Use `09_network_architecture.png` to explain model structure
2. **For validation:** Show `07_grad_cam_heatmap.png` to doctors/colleagues
3. **For papers:** Include both `02_roc_curves.png` and `01_confusion_matrix.png`
4. **For debugging:** Check `03_loss_curves.png` first for training issues
5. **For reproducibility:** Save all visualizations + `training_history.json`

---

## 🚀 Advanced: Generate High-Resolution Versions

Edit `analyze_model.py`:
```python
# Line 15
plt.rcParams['savefig.dpi'] = 300  # Was 150
```

Then run analysis again for publication-quality images.

---

**Questions?** Check each visualization's docstring in `analyze_model.py` for more details!

Happy analyzing! 🎉
