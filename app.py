import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn
import io
import numpy as np
import csv
import re
import pandas as pd
from datetime import datetime
import os

# Import the analysis module
from streamlit_analysis import get_analyzer

# Page configuration
st.set_page_config(
    page_title="🦴 X-Ray Osteoporosis Analyzer",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    /* Softer, modern card styles that work in dark/light themes */
    .prediction-card {
        padding: 1.25rem 1.5rem;
        border-radius: 14px;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        box-shadow: 0 8px 24px rgba(0,0,0,0.18);
        backdrop-filter: blur(6px);
    }
    .normal-card {
        background: linear-gradient(135deg, rgba(76,175,80,0.18) 0%, rgba(76,175,80,0.10) 100%);
        border: 1px solid rgba(76,175,80,0.35);
    }
    .osteoporotic-card {
        /* Warm amber/rose instead of harsh red */
        background: linear-gradient(135deg, rgba(255,171,64,0.20) 0%, rgba(255,138,101,0.12) 100%);
        border: 1px solid rgba(255,171,64,0.40);
    }

    .prediction-title {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        font-size: 1.5rem;
        font-weight: 700;
        margin: 0;
    }
    .prediction-meta {
        opacity: 0.85;
        margin-top: 4px;
        font-size: 0.95rem;
    }
    .prediction-subtle {
        margin-top: 8px;
        font-size: 0.85rem;
        opacity: 0.7;
    }
    .status-dot {
        width: 14px;
        height: 14px;
        border-radius: 50%;
        display: inline-block;
        box-shadow: 0 0 0 4px rgba(255,255,255,0.06) inset;
    }
    .dot-normal { background: radial-gradient(circle at 30% 30%, #7bd88f, #2e7d32); }
    .dot-osteo  { background: radial-gradient(circle at 30% 30%, #ffcc80, #ef6c00); }

    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 30px;
        margin: 10px 0;
    }

    /* Severity cards */
    .severity-card {
        padding: 1.1rem 1.25rem;
        border-radius: 14px;
        margin: 0.75rem 0 1.25rem 0;
        border: 1px solid rgba(255,255,255,0.08);
        background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
        box-shadow: 0 8px 24px rgba(0,0,0,0.14);
        backdrop-filter: blur(4px);
    }
    .severity-normal  { background: linear-gradient(135deg, rgba(76,175,80,0.20) 0%, rgba(76,175,80,0.10) 100%); border: 1px solid rgba(76,175,80,0.35);} 
    .severity-mild    { background: linear-gradient(135deg, rgba(129,199,132,0.22) 0%, rgba(174,213,129,0.14) 100%); border: 1px solid rgba(129,199,132,0.45);} 
    .severity-moderate{ background: linear-gradient(135deg, rgba(255,213,79,0.22) 0%, rgba(255,183,77,0.14) 100%); border: 1px solid rgba(255,213,79,0.45);} 
    .severity-severe  { background: linear-gradient(135deg, rgba(255,171,145,0.22) 0%, rgba(255,138,128,0.14) 100%); border: 1px solid rgba(255,171,145,0.45);} 

    .severity-header { display: flex; align-items: center; justify-content: space-between; gap: 0.75rem; }
    .severity-chip   { padding: 0.25rem 0.6rem; border-radius: 999px; font-weight: 700; font-size: 0.9rem; background: rgba(255,255,255,0.18); }
    .severity-score  { font-weight: 600; opacity: 0.9; }
    .severity-message{ margin-top: 0.6rem; line-height: 1.5; }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
    st.session_state.model_loaded = False

@st.cache_resource
def load_analyzer():
    """Load analyzer with caching"""
    return get_analyzer()

# Load analyzer
analyzer = load_analyzer()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_image(image):
    """Preprocess image for model prediction"""
    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485], [0.229]),
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return tfms(image).unsqueeze(0)

def predict_xray(image, model, device):
    """Make prediction on X-ray image"""
    try:
        # Preprocess image
        x = preprocess_image(image).to(device)
        
        # Make prediction
        with torch.no_grad():
            preds = model(x)
            probs = torch.softmax(preds, dim=1)
        
        return probs[0].cpu().numpy()
    
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None


# ------------------------ Severity scoring & CSV helpers ------------------------
def compute_severity(model_prob_osteo, failure_load_n, prop_speed_mm_s,
                     F_REF=300.0, V_REF=60.0, V_MAX=500.0):
    """Compute severity score using model probability and mechanical features."""
    # Load score (weaker bones fracture at lower force)
    S_F = max(0.0, min(1.0, failure_load_n / float(F_REF))) if F_REF else 1.0
    # Propagation speed score (higher speed = more brittle)
    denom = max(1e-6, float(V_MAX) - float(V_REF))
    S_v = max(0.0, min(1.0, (prop_speed_mm_s - float(V_REF)) / denom))
    severity = 0.45 * float(model_prob_osteo) + 0.35 * (1.0 - S_F) + 0.20 * S_v
    return severity


def bone_message(severity, failure_load_n, prop_speed_mm_s):
    if severity < 0.30:
        category = "✅ Healthy Bone"
        message = f"Strong bone structure detected. Breaking force: {failure_load_n:.0f}N. Low fracture risk."
    elif severity < 0.55:
        category = "⚠️ Mild Osteoporosis"
        message = f"Early signs detected. Recommend monitoring. Breaking force: {failure_load_n:.0f}N."
    elif severity < 0.75:
        category = "⚠️⚠️ Moderate Osteoporosis"
        message = f"Significant bone loss. Consult specialist. Breaking force: {failure_load_n:.0f}N."
    else:
        category = "🚨 Severe Osteoporosis"
        message = f"Critical condition. Immediate medical attention needed. Breaking force: {failure_load_n:.0f}N."
    
    return category, message


def _parse_breaking_point_to_newtons(value):
    """Parse strings like '1.98 Lbs' or '.35 Lbs' or 'NULL' to Newtons (N). 1 lbf = 4.4482216153 N."""
    if value is None:
        return None
    s = str(value).strip()
    if s.upper() == "NULL" or s == "":
        return None
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
    if not m:
        return None
    lbs = float(m.group(0))
    return lbs * 4.4482216153


def _parse_speed_to_mm_per_s(value):
    """Parse strings like '.37 M/s' to mm/s."""
    if value is None:
        return None
    s = str(value).strip()
    if s.upper() == "NULL" or s == "":
        return None
    m = re.search(r"[-+]?[0-9]*\.?[0-9]+", s)
    if not m:
        return None
    m_per_s = float(m.group(0))
    return m_per_s * 1000.0


@st.cache_data(show_spinner=False)
def parse_csv_bytes(csv_bytes: bytes):
    """Parse uploaded CSV bytes. Returns (rows, F_REF, V_REF, V_MAX, trials).
    rows: list of dicts with keys trial, failure_load_n, prop_speed_mm_s
    """
    text = csv_bytes.decode("utf-8", errors="ignore")
    f = io.StringIO(text)
    reader = csv.DictReader(f)

    rows = []
    failure_vals = []
    speed_vals = []

    for r in reader:
        trial_str = r.get("Trial")
        try:
            trial = int(str(trial_str).strip()) if trial_str is not None else None
        except Exception:
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

    F_REF = sum(failure_vals) / len(failure_vals) if failure_vals else 300.0
    V_REF = sum(speed_vals) / len(speed_vals) if speed_vals else 60.0
    if speed_vals:
        vmax_raw = max(speed_vals)
        V_MAX = max(vmax_raw, V_REF + 1.0)
    else:
        V_MAX = 500.0

    trials = [r["trial"] for r in rows if r.get("trial") is not None]
    return rows, F_REF, V_REF, V_MAX, trials


@st.cache_data(show_spinner=False)
def parse_csv_file(csv_path: str):
    """Parse CSV from a file path. Returns (rows, F_REF, V_REF, V_MAX, trials)."""
    rows = []
    failure_vals = []
    speed_vals = []

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            trial_str = r.get("Trial")
            try:
                trial = int(str(trial_str).strip()) if trial_str is not None else None
            except Exception:
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
                "raw_breaking_point": r.get("Breaking Point"),
                "raw_speed": r.get("Fracture Propogation Speed"),
            })

    F_REF = sum(failure_vals) / len(failure_vals) if failure_vals else 300.0
    V_REF = sum(speed_vals) / len(speed_vals) if speed_vals else 60.0
    if speed_vals:
        vmax_raw = max(speed_vals)
        V_MAX = max(vmax_raw, V_REF + 1.0)
    else:
        V_MAX = 500.0

    trials = [r["trial"] for r in rows if r.get("trial") is not None]
    return rows, F_REF, V_REF, V_MAX, trials

# ==================== MAIN APP ====================

st.markdown('<h1 class="main-header">🦴 X-Ray Osteoporosis Analyzer</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 📊 Model Information")
    st.markdown("""
    - **Architecture:** EfficientNet-B0
    - **Input Size:** 224×224 pixels
    - **Classes:** Normal / Osteoporotic
    - **Training:** 12 epochs with validation split
    """)
    
    st.markdown("## 📝 Instructions")
    st.markdown("""
    1. Upload an X-ray image (PNG, JPG, JPEG)
    2. Click "Analyze X-Ray"
    3. View prediction results
    4. Explore detailed visualizations
    """)
    
    st.markdown("## ⚠️ Medical Disclaimer")
    st.warning(
        "This AI tool is for **educational and research purposes only**. "
        "It is NOT a replacement for professional medical diagnosis. "
        "Always consult qualified healthcare professionals."
    )

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["🔍 Image Analysis", "📊 Model Performance", "📈 Statistical Analysis"])

with tab1:
    st.markdown("### Upload and Analyze X-Ray Image")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Choose an X-ray image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a bone X-ray image in PNG, JPG, or JPEG format"
        )
    
    with col2:
        analyze_button = st.button("🔍 Analyze X-Ray", use_container_width=True, key="analyze_btn")
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        st.markdown("### Input Image")
        st.image(image, caption="Uploaded X-Ray Image", use_column_width=True)
        
        if analyze_button:
            with st.spinner("🤖 Analyzing image..."):
                probs = predict_xray(image, analyzer.model, device)
                
                if probs is not None:
                    normal_prob = probs[0]
                    osteo_prob = probs[1]
                    
                    # Determine prediction
                    prediction = "Normal" if normal_prob > osteo_prob else "Osteoporotic"
                    confidence = max(normal_prob, osteo_prob)
                    
                    # Store in session for later use
                    st.session_state.last_prediction = {
                        'image': image,
                        'normal_prob': normal_prob,
                        'osteo_prob': osteo_prob,
                        'prediction': prediction,
                        'confidence': confidence
                    }
        
        # Display prediction
        if 'last_prediction' in st.session_state:
            st.markdown("### 🎯 Prediction Results")
            
            pred_data = st.session_state.last_prediction
            normal_prob = pred_data['normal_prob']
            osteo_prob = pred_data['osteo_prob']
            prediction = pred_data['prediction']
            confidence = pred_data['confidence']
            
            # Prediction card
            card_class = "normal-card" if prediction == "Normal" else "osteoporotic-card"
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <p class="prediction-title">
                    <span class="status-dot dot-{'normal' if prediction == 'Normal' else 'osteo'}"></span>
                    <strong>{prediction}</strong>
                </p>
                <p class="prediction-meta">Confidence: <strong>{confidence*100:.1f}%</strong></p>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed probabilities
            st.markdown("#### Detailed Probabilities")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Normal Probability", f"{normal_prob*100:.2f}%")
            
            with col2:
                st.metric("Osteoporotic Probability", f"{osteo_prob*100:.2f}%")
            
            # ==================== IMAGE-SPECIFIC VISUALIZATIONS ====================
            
            st.markdown("---")
            st.markdown("### 🔥 Advanced Image Analysis")
            
            sub_tab1, sub_tab2, sub_tab3 = st.tabs([
                "🧠 Grad-CAM Attention",
                "🎨 Feature Extraction",
                "📊 Confidence Breakdown"
            ])
            
            with sub_tab1:
                st.markdown("#### Where the AI Looks in Your Image")
                st.markdown("""
                **Grad-CAM (Gradient-weighted Class Activation Mapping)** shows which regions 
                of the X-ray the AI model focuses on when making predictions.
                - 🔴 **Red areas** = High attention (important for diagnosis)
                - 🔵 **Blue areas** = Low attention
                """)
                
                with st.spinner("Generating Grad-CAM..."):
                    grad_cam_fig = analyzer.grad_cam_single_image(image)
                    st.pyplot(grad_cam_fig, use_container_width=True)
            
            with sub_tab2:
                st.markdown("#### Feature Extraction Analysis")
                st.markdown("""
                Shows low-level features extracted from the X-ray:
                - **Original** = Input X-ray
                - **Edge Detection** = Boundary detection (Canny)
                - **Gradient Magnitude** = Intensity changes (Sobel)
                - **Contours** = Detected boundaries
                """)
                
                with st.spinner("Extracting features..."):
                    feature_fig = analyzer.feature_extraction_single_image(image)
                    st.pyplot(feature_fig, use_container_width=True)
            
            with sub_tab3:
                st.markdown("#### Model Confidence Analysis")
                st.markdown("Visual breakdown of how confident the model is in its prediction.")
                
                with st.spinner("Computing confidence metrics..."):
                    confidence_fig = analyzer.confidence_breakdown(image)
                    st.pyplot(confidence_fig, use_container_width=True)

with tab2:
    st.markdown("### Model Performance Metrics")
    st.markdown("""
    These metrics evaluate the model's performance across the entire dataset.
    They help understand the model's overall reliability and diagnostic accuracy.
    """)
    
    # Create sub-tabs for different metrics
    perf_tab1, perf_tab2, perf_tab3 = st.tabs([
        "🎯 Confusion Matrix",
        "📉 ROC Curve",
        "📚 Training History"
    ])
    
    with perf_tab1:
        st.markdown("#### Confusion Matrix")
        st.markdown("""
        Shows how often the model correctly/incorrectly classifies samples:
        - **TP (Top-left)** = Correctly identified Normal
        - **FP (Top-right)** = Incorrectly labeled as Osteoporotic (False Positive)
        - **FN (Bottom-left)** = Incorrectly labeled as Normal (False Negative)
        - **TN (Bottom-right)** = Correctly identified Osteoporotic
        """)
        
        with st.spinner("Generating confusion matrix..."):
            cm_fig = analyzer.plot_confusion_matrix_fig()
            st.pyplot(cm_fig, use_container_width=True)
    
    with perf_tab2:
        st.markdown("#### ROC Curve Analysis")
        st.markdown("""
        The ROC (Receiver Operating Characteristic) curve shows the trade-off between 
        True Positive Rate and False Positive Rate.
        - **AUC = 1.0** = Perfect classifier
        - **AUC = 0.5** = Random classifier
        - **AUC > 0.8** = Good classifier
        """)
        
        with st.spinner("Generating ROC curve..."):
            roc_fig = analyzer.plot_roc_curve_fig()
            st.pyplot(roc_fig, use_container_width=True)
    
    with perf_tab3:
        st.markdown("#### Training Loss & Accuracy Curves")
        st.markdown("""
        Visualizes how the model improved during training:
        - **Left plot** = Training loss (should decrease)
        - **Right plot** = Validation accuracy (should increase)
        """)
        
        with st.spinner("Generating loss curves..."):
            loss_fig = analyzer.plot_loss_curves_fig()
            st.pyplot(loss_fig, use_container_width=True)

with tab3:
    st.markdown("### Detailed Statistical Analysis")
    st.markdown("""
    Advanced analysis including feature importance, mechanical properties, and network architecture.
    """)
    
    stat_tab1, stat_tab2, stat_tab3, stat_tab4, stat_tab5 = st.tabs([
        "🎯 Feature Importance",
        "💪 Breaking Force",
        "📊 Porosity-Strength",
        "⚡ Crack Velocity",
        "🏗️ Architecture"
    ])
    
    with stat_tab1:
        st.markdown("#### Top Important Features")
        st.markdown("""
        The features the model learned to distinguish between Normal and Osteoporotic bones.
        Taller bars indicate more important features in the CNN.
        """)
        
        with st.spinner("Computing feature importance..."):
            feat_fig = analyzer.plot_feature_importance_fig()
            st.pyplot(feat_fig, use_container_width=True)
    
    with stat_tab2:
        st.markdown("#### Breaking Force Analysis")
        st.markdown("""
        Shows mechanical properties across different bone types:
        - **Height** = Average breaking force (Newtons)
        - **Error bars** = ± Standard deviation
        - Indicates which bone types are stronger/weaker
        """)
        
        with st.spinner("Analyzing breaking force..."):
            force_fig = analyzer.plot_breaking_force_fig()
            st.pyplot(force_fig, use_container_width=True)
    
    with stat_tab3:
        st.markdown("#### Porosity vs Bone Strength Correlation")
        st.markdown("""
        Demonstrates the relationship between bone porosity and mechanical strength:
        - **R² value** indicates correlation strength
        - **Red line** = Linear regression fit
        - Strong negative correlation validates the model's logic
        """)
        
        with st.spinner("Computing correlation..."):
            porosity_fig = analyzer.plot_porosity_strength_fig()
            st.pyplot(porosity_fig, use_container_width=True)
    
    with stat_tab4:
        st.markdown("#### Crack Propagation Velocity Distribution")
        st.markdown("""
        Shows how quickly cracks propagate in different bone types:
        - **Box plot** = Quartiles and outliers
        - **Violin plot** = Full probability distribution
        """)
        
        with st.spinner("Analyzing crack velocity..."):
            velocity_fig = analyzer.plot_crack_velocity_fig()
            st.pyplot(velocity_fig, use_container_width=True)
    
    with stat_tab5:
        st.markdown("#### Network Architecture")
        st.markdown("""
        Visual representation of the EfficientNet-B0 architecture used for classification.
        Shows how data flows through the network layers.
        """)
        
        with st.spinner("Drawing architecture..."):
            arch_fig = analyzer.plot_network_architecture_fig()
            st.pyplot(arch_fig, use_container_width=True)

# ==================== FOOTER ====================

st.markdown("---")
st.markdown("""
<div style="text-align: center; opacity: 0.7; font-size: 0.9rem;">
    <p>🦴 X-Ray Osteoporosis Analyzer | Medical AI | Educational Use Only</p>
    <p>⚠️ Not a replacement for professional medical diagnosis</p>
</div>
""", unsafe_allow_html=True)