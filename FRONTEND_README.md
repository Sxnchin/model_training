# 🦴 X-Ray Osteoporosis Analyzer - Web Frontend

A beautiful web interface for analyzing X-ray images to detect osteoporosis using AI.

## 🚀 Quick Start

### Option 1: Run with Batch File (Easiest)
```bash
# Double-click on run_app.bat
# OR run in terminal:
run_app.bat
```

### Option 2: Run with Streamlit Command
```bash
streamlit run app.py
```

The web app will automatically open in your browser at `http://localhost:8501`

## 📱 How to Use

1. **Open the Web App**: The interface will load in your browser
2. **Upload X-Ray**: Click "Browse files" and select an X-ray image (PNG, JPG, JPEG)
3. **Analyze**: Click the "🔍 Analyze X-Ray" button
4. **View Results**: See the AI prediction with confidence scores
5. **Download**: Save results as a text file

## 🎯 Features

- **🖼️ Drag & Drop Upload**: Easy image upload interface
- **🤖 Real-time AI Analysis**: Instant predictions using your trained model
- **📊 Visual Results**: Clear prediction cards with confidence percentages
- **📈 Detailed Probabilities**: Bar charts showing normal vs osteoporotic probabilities
- **📥 Export Results**: Download analysis results as text files
- **📱 Responsive Design**: Works on desktop, tablet, and mobile
- **⚕️ Medical Disclaimer**: Built-in safety warnings

## 🛡️ Safety Features

- Medical disclaimer prominently displayed
- Clear indication this is for educational use only
- Recommendation to consult healthcare professionals

## 🔧 Technical Details

- **Framework**: Streamlit
- **AI Model**: EfficientNet-B0 (trained on MURA dataset)
- **Image Processing**: PIL + torchvision transforms
- **GPU Support**: Automatically detects CUDA if available

## 📋 Requirements

All dependencies are listed in `requirements.txt`:
```
torch
torchvision
Pillow
timm
streamlit
```

## 🎨 Interface Preview

The web interface includes:
- **Header**: Beautiful gradient title
- **Sidebar**: Model information and instructions
- **Upload Section**: Drag-and-drop file uploader
- **Results Panel**: Prediction cards with color coding
- **Progress Bars**: Visual confidence indicators
- **Download Options**: Export functionality

## 🔗 URLs

When running locally:
- **Web Interface**: http://localhost:8501
- **Stop Server**: Press `Ctrl+C` in terminal

## 📞 Support

If you encounter any issues:
1. Make sure all requirements are installed
2. Check that `efficientnet_humerus.pt` model file exists
3. Verify your image is in a supported format (PNG, JPG, JPEG)
4. Ensure you have enough memory for the model

Enjoy analyzing X-rays with your AI-powered web interface! 🎉