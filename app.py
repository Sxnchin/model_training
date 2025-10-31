import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
import timm
import torch.nn as nn
import io
import numpy as np

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
    .prediction-card {
        padding: 1.5rem;
        border-radius: 10px;
        border: 2px solid #e1e5e9;
        margin: 1rem 0;
    }
    .normal-card {
        background-color: #d4edda;
        border-color: #28a745;
    }
    .osteoporotic-card {
        background-color: #f8d7da;
        border-color: #dc3545;
    }
    .confidence-bar {
        background-color: #e9ecef;
        border-radius: 10px;
        overflow: hidden;
        height: 30px;
        margin: 10px 0;
    }
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
                <h2 style="margin: 0; text-align: center;">
                    {'🟢' if prediction == 'Normal' else '🔴'} {prediction}
                </h2>
                <p style="text-align: center; font-size: 1.2rem; margin: 10px 0;">
                    Confidence: {confidence:.1%}
                </p>
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