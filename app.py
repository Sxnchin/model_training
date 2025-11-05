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
from datetime import datetime
import os

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
def load_model():
    """Load the trained model"""
    try:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        MODEL_PATH = "efficientnet_humerus.pt"
        
        # Load model architecture (same as train.py)
        model = timm.create_model("efficientnet_b0", pretrained=False)
        model.classifier = nn.Linear(model.classifier.in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        
        return model, DEVICE
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

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
        
        normal_prob = probs[0][0].item()
        osteo_prob = probs[0][1].item()
        
        prediction = "Normal" if normal_prob > osteo_prob else "Osteoporotic"
        confidence = max(normal_prob, osteo_prob)
        
        return prediction, normal_prob, osteo_prob, confidence
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None, None, None


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
        category = "Normal"
        msg = "No significant weakening detected. Maintain healthy bone habits."
    elif severity < 0.55:
        category = "Mild"
        msg = (f"While not a medical diagnosis, the bone shows mild weakening. "
               f"Expected fracture force ≈ {failure_load_n:.0f} N. "
               f"Consult a healthcare provider if symptoms or risk factors exist.")
    elif severity < 0.75:
        category = "Moderate"
        msg = (f"Bone shows moderate structural weakening. "
               f"Fracture may occur around {failure_load_n:.0f} N. "
               f"Propagation speed ({prop_speed_mm_s:.0f} mm/s) suggests reduced toughness.")
    else:
        category = "Severe"
        msg = (f"Bone shows **severe** weakening. Expected fracture load ≈ {failure_load_n:.0f} N. "
               f"Propagation speed ({prop_speed_mm_s:.0f} mm/s) indicates brittle failure. "
               f"Seek medical evaluation promptly.")
    return category, msg


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

# Main app
def main():
    # Header
    st.markdown('<h1 class="main-header">🦴 X-Ray Osteoporosis Analyzer</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("ℹ️ About")
        st.markdown("""
        This AI-powered tool analyzes X-ray images to detect signs of osteoporosis in humerus bones.
        
        **How to use:**
        1. Upload an X-ray image
        2. Wait for AI analysis
        3. Review the results
        
        **Supported formats:** PNG, JPG, JPEG
        """)
        
        st.header("🔧 Model Info")
        model, device = load_model()
        if model is not None:
            st.success("✅ Model loaded successfully")
            st.info(f"🖥️ Running on: {device}")
        else:
            st.error("❌ Model failed to load")
            return

        st.header("📄 Mechanical Test Data")
        st.caption("Using bundled CSV shipped with this app (no upload required).")

        # Defaults
        default_F_REF, default_V_REF, default_V_MAX = 300.0, 60.0, 500.0
        failure_load_n = 180.0
        prop_speed_mm_s = 90.0
        F_REF, V_REF, V_MAX = default_F_REF, default_V_REF, default_V_MAX
        selected_trial = None

        # Load internal CSV
        try:
            csv_path = os.path.join(os.path.dirname(__file__), "humpty dumpty is humping my leg.csv")
            rows, F_REF, V_REF, V_MAX, trials = parse_csv_file(csv_path)
            st.caption(f"Computed references from CSV → F_REF={F_REF:.0f} N, V_REF={V_REF:.0f} mm/s, V_MAX={V_MAX:.0f} mm/s")
            if trials:
                selected_trial = st.selectbox("Select Trial", trials)
                row = next((r for r in rows if r.get("trial") == selected_trial), None)
                if row:
                    failure_load_n = row.get("failure_load_n") or F_REF
                    prop_speed_mm_s = row.get("prop_speed_mm_s") or V_REF
                    st.caption(
                        f"Trial {selected_trial} → Breaking Point (raw: {row.get('raw_breaking_point','?')}) ≈ {failure_load_n:.2f} N, "
                        f"Speed (raw: {row.get('raw_speed','?')}) ≈ {prop_speed_mm_s:.0f} mm/s"
                    )
        except Exception as e:
            st.warning(f"Could not load internal CSV: {e}")

        # Allow manual override or manual entry when no CSV
        with st.expander("Advanced overrides", expanded=False):
            failure_load_n = st.number_input("Failure load (N)", min_value=0.0, value=float(failure_load_n), step=10.0)
            prop_speed_mm_s = st.number_input("Propagation speed (mm/s)", min_value=0.0, value=float(prop_speed_mm_s), step=10.0)
            F_REF = st.number_input("Reference failure load F_REF (N)", min_value=1.0, value=float(F_REF), step=10.0)
            V_REF = st.number_input("Reference speed V_REF (mm/s)", min_value=0.0, value=float(V_REF), step=10.0)
            V_MAX = st.number_input("Max speed V_MAX (mm/s)", min_value=1.0, value=float(V_MAX), step=10.0)

        # Stash in session for use after model prediction
        st.session_state.mech = dict(
            failure_load_n=failure_load_n,
            prop_speed_mm_s=prop_speed_mm_s,
            F_REF=F_REF,
            V_REF=V_REF,
            V_MAX=V_MAX,
            trial=selected_trial,
        )
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload X-Ray Image")
        uploaded_file = st.file_uploader(
            "Choose an X-ray image...",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear X-ray image of a humerus bone"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded X-Ray", use_column_width=True)
            
            # Analysis button
            if st.button("🔍 Analyze X-Ray", type="primary"):
                with st.spinner("🤖 AI is analyzing your X-ray..."):
                    prediction, normal_prob, osteo_prob, confidence = predict_xray(image, model, device)
                
                if prediction is not None:
                    # Store results in session state
                    st.session_state.prediction = prediction
                    st.session_state.normal_prob = normal_prob
                    st.session_state.osteo_prob = osteo_prob
                    st.session_state.confidence = confidence
                    st.session_state.analysis_time = datetime.now().isoformat(timespec="seconds")

                    # Compute severity based on sidebar values
                    mech = st.session_state.get("mech", {})
                    failure_load_n = mech.get("failure_load_n", 180.0)
                    prop_speed_mm_s = mech.get("prop_speed_mm_s", 90.0)
                    F_REF = mech.get("F_REF", 300.0)
                    V_REF = mech.get("V_REF", 60.0)
                    V_MAX = mech.get("V_MAX", 500.0)

                    severity = compute_severity(osteo_prob, failure_load_n, prop_speed_mm_s, F_REF, V_REF, V_MAX)
                    category, message = bone_message(severity, failure_load_n, prop_speed_mm_s)
                    st.session_state.severity = severity
                    st.session_state.severity_category = category
                    st.session_state.severity_message = message
    
    with col2:
        st.header("📊 Analysis Results")
        
        if hasattr(st.session_state, 'prediction'):
            prediction = st.session_state.prediction
            normal_prob = st.session_state.normal_prob
            osteo_prob = st.session_state.osteo_prob
            confidence = st.session_state.confidence
            
            # Main prediction card
            card_class = "normal-card" if prediction == "Normal" else "osteoporotic-card"
            
            st.markdown(f"""
            <div class="prediction-card {card_class}">
                <div class="prediction-title">
                    <span class="status-dot {'dot-normal' if prediction=='Normal' else 'dot-osteo'}"></span>
                    <span class="prediction-text">{prediction}</span>
                </div>
                <div class="prediction-meta">Confidence: {confidence:.1%}</div>
                <div class="prediction-subtle">AI-assisted screening result — not a medical diagnosis.</div>
            </div>
            """, unsafe_allow_html=True)
            
            # Detailed probabilities
            st.subheader("📈 Detailed Analysis")
            
            # Normal probability
            st.markdown("**Normal Bone:**")
            st.progress(normal_prob)
            st.text(f"{normal_prob:.1%}")
            
            # Osteoporotic probability
            st.markdown("**Osteoporotic Bone:**")
            st.progress(osteo_prob)
            st.text(f"{osteo_prob:.1%}")
            
            # Severity and recommendation
            if 'severity' in st.session_state:
                st.subheader("🦴 Severity & Recommendation")
                cat = st.session_state.severity_category
                sev = float(st.session_state.severity)
                css_class = {
                    "Normal": "severity-normal",
                    "Mild": "severity-mild",
                    "Moderate": "severity-moderate",
                    "Severe": "severity-severe",
                }.get(cat, "severity-mild")

                st.markdown(f"""
                <div class="severity-card {css_class}">
                    <div class="severity-header">
                        <div class="severity-chip">{cat}</div>
                        <div class="severity-score">Severity: {sev:.2f}</div>
                    </div>
                    <div class="severity-message">{st.session_state.severity_message}</div>
                </div>
                """, unsafe_allow_html=True)

                # Move technical details into an optional expander (less scary)
                mech = st.session_state.get("mech", {})
                with st.expander("Details (technical)"):
                    st.write(
                        f"Using F_REF={mech.get('F_REF', 300.0):.0f} N, "
                        f"V_REF={mech.get('V_REF', 60.0):.0f} mm/s, "
                        f"V_MAX={mech.get('V_MAX', 500.0):.0f} mm/s; "
                        f"Trial={mech.get('trial', '—')}"
                    )

            # Additional information
            st.subheader("⚕️ Medical Disclaimer")
            st.warning("""
            **Important:** This AI tool is for educational and research purposes only. 
            It should NOT be used as a substitute for professional medical diagnosis. 
            Always consult with a qualified healthcare provider for medical decisions.
            """)
            
            # Download results
            if st.button("📥 Download Results"):
                results_text = f"""
X-Ray Osteoporosis Analysis Results
==================================

Prediction: {prediction}
Confidence: {confidence:.1%}

Detailed Probabilities:
- Normal: {normal_prob:.4f} ({normal_prob:.1%})
- Osteoporotic: {osteo_prob:.4f} ({osteo_prob:.1%})

 Severity Assessment:
 - Severity score: {st.session_state.get('severity', float('nan')):.2f}
 - Category: {st.session_state.get('severity_category', 'N/A')}
 - Message: {st.session_state.get('severity_message', 'N/A')}

 Mechanical Inputs:
 - Failure load (N): {st.session_state.get('mech', {}).get('failure_load_n', 'N/A')}
 - Propagation speed (mm/s): {st.session_state.get('mech', {}).get('prop_speed_mm_s', 'N/A')}
 - F_REF (N): {st.session_state.get('mech', {}).get('F_REF', 'N/A')}
 - V_REF (mm/s): {st.session_state.get('mech', {}).get('V_REF', 'N/A')}
 - V_MAX (mm/s): {st.session_state.get('mech', {}).get('V_MAX', 'N/A')}

Generated by AI X-Ray Analyzer
Timestamp: {str(st.session_state.get('analysis_time', 'Unknown'))}

MEDICAL DISCLAIMER: This analysis is for educational purposes only 
and should not replace professional medical diagnosis.
                """
                st.download_button(
                    label="📄 Download as Text",
                    data=results_text,
                    file_name=f"xray_analysis_{prediction.lower()}.txt",
                    mime="text/plain"
                )
        else:
            st.info("👆 Upload an X-ray image and click 'Analyze' to see results here.")
            
            # Example images section
            st.subheader("🖼️ Example Analysis")
            st.markdown("""
            **What you'll see after analysis:**
            - 🎯 Clear prediction (Normal/Osteoporotic)
            - 📊 Confidence percentages
            - 📈 Detailed probability breakdown
            - 📥 Downloadable results
            """)

if __name__ == "__main__":
    main()