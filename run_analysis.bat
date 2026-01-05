@echo off
REM 🔬 Quick Analysis Script - Generates all visualizations
echo.
echo ========================================
echo   🔬 MODEL ANALYSIS & VISUALIZATION
echo ========================================
echo.

echo Checking for trained model...
if not exist "efficientnet_humerus.pt" (
    echo ❌ ERROR: Model file not found!
    echo Run 'python train.py' first
    pause
    exit /b 1
)

echo ✅ Model found: efficientnet_humerus.pt
echo.
echo 📊 Generating comprehensive analysis...
echo    - Confusion matrices
echo    - ROC curves with AUC
echo    - Loss curves
echo    - Feature importance
echo    - Grad-CAM heatmaps
echo    - Statistical plots
echo    - Network architecture
echo    - And more!
echo.

python analyze_model.py

if errorlevel 1 (
    echo.
    echo ❌ Analysis failed. Check error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo ✅ ANALYSIS COMPLETE!
echo ========================================
echo.
echo 📁 Results saved to: analysis_results\
echo.
echo Generated files:
echo   01_confusion_matrix_training.png
echo   01_confusion_matrix_validation.png
echo   02_roc_curves.png
echo   03_loss_curves.png
echo   04_feature_importance.png
echo   05_breaking_force.png
echo   06_porosity_strength.png
echo   07_grad_cam_heatmap.png
echo   08_crack_velocity.png
echo   09_network_architecture.png
echo   10_feature_extraction.png
echo.
echo 📖 See ANALYSIS_GUIDE.md for interpretation help
echo.
pause
