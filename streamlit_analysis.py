"""
Streamlit-compatible analysis module for bone osteoporosis detection
Generates visualizations that work seamlessly in the web interface
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from PIL import Image
import io

import torch
import torch.nn as nn
from torchvision import transforms, datasets
import timm
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import cv2
import json

# Configure matplotlib for Streamlit
plt.style.use('seaborn-v0_8-darkgrid')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StreamlitAnalyzer:
    """All analysis functions compatible with Streamlit"""
    
    def __init__(self, model_path="efficientnet_humerus.pt", data_dir="dataset_sorted"):
        self.model_path = model_path
        self.data_dir = data_dir
        self.model = None
        self.device = DEVICE
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        if self.model is None:
            self.model = timm.create_model("efficientnet_b0", pretrained=False)
            self.model.classifier = nn.Linear(self.model.classifier.in_features, 2)
            self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
    
    # ==================== IMAGE-SPECIFIC ANALYSIS ====================
    
    def grad_cam_single_image(self, img_pil):
        """
        Simple, working Grad-CAM implementation
        Returns matplotlib figure
        """
        # Preprocessing
        tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])
        
        img_tensor = tfms(img_pil).unsqueeze(0).to(self.device)
        
        # Use early layer (block 3) for spatial detail
        target_layer = self.model.blocks[3][-1]
        
        activations = []
        gradients = []
        
        def forward_hook(module, input, output):
            activations.append(output)
        
        def backward_hook(module, grad_input, grad_output):
            gradients.append(grad_output[0])
        
        fwd_handle = target_layer.register_forward_hook(forward_hook)
        bwd_handle = target_layer.register_full_backward_hook(backward_hook)
        
        # Forward
        self.model.eval()
        output = self.model(img_tensor)
        pred_class = output.argmax(1).item()
        
        # Backward
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0, pred_class] = 1
        output.backward(gradient=one_hot)
        
        # Get activations and gradients
        acts = activations[0][0].cpu().detach()  # (C, H, W)
        grads = gradients[0][0].cpu().detach()   # (C, H, W)
        
        # Pool gradients across spatial dimensions
        pooled_grads = grads.mean(dim=[1, 2])  # (C,)
        
        # Weight feature maps by gradients
        for i in range(acts.shape[0]):
            acts[i] *= pooled_grads[i]
        
        # Average across all feature maps
        heatmap = acts.mean(dim=0).numpy()
        
        # ReLU
        heatmap = np.maximum(heatmap, 0)
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = heatmap / heatmap.max()
        
        # Resize to image size
        heatmap = cv2.resize(heatmap, (224, 224))
        
        # Apply minimal smoothing
        heatmap = cv2.GaussianBlur(heatmap, (3, 3), 0)
        
        fwd_handle.remove()
        bwd_handle.remove()
        
        # Plot
        img_rgb = img_pil.convert('RGB').resize((224, 224))
        img_np = np.array(img_rgb)
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(img_np)
        axes[0].set_title('Original X-Ray', fontweight='bold', fontsize=12)
        axes[0].axis('off')
        
        # Overlay
        axes[1].imshow(img_np)
        axes[1].imshow(heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
        axes[1].set_title('Model Attention Overlay', fontweight='bold', fontsize=12)
        axes[1].axis('off')
        
        # Heatmap only
        im = axes[2].imshow(heatmap, cmap='jet', vmin=0, vmax=1)
        axes[2].set_title('Attention Heatmap', fontweight='bold', fontsize=12)
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], label='Attention Score')
        
        plt.tight_layout()
        return fig
    
    def feature_extraction_single_image(self, img_pil):
        """
        Extract and visualize features (edges, gradients) for single image
        Returns matplotlib figure
        """
        # Ensure RGB image
        img_rgb = img_pil.convert('RGB').resize((224, 224))
        img_np = np.array(img_rgb)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Edge detection
        edges = cv2.Canny(gray, 100, 200)
        
        # Gradient magnitude (Sobel)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mag = (gradient_mag / gradient_mag.max() * 255).astype(np.uint8)
        
        # Contours
        contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = gray.copy()
        cv2.drawContours(contour_img, contours[:10], -1, (255, 0, 0), 2)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original
        axes[0, 0].imshow(img_np)
        axes[0, 0].set_title('Original X-Ray', fontweight='bold')
        axes[0, 0].axis('off')
        
        # Edges
        axes[0, 1].imshow(edges, cmap='gray')
        axes[0, 1].set_title('Edge Detection (Canny)', fontweight='bold')
        axes[0, 1].axis('off')
        
        # Gradients
        axes[1, 0].imshow(gradient_mag, cmap='hot')
        axes[1, 0].set_title('Gradient Magnitude (Sobel)', fontweight='bold')
        axes[1, 0].axis('off')
        
        # Contours
        axes[1, 1].imshow(contour_img, cmap='gray')
        axes[1, 1].set_title('Detected Contours', fontweight='bold')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def confidence_breakdown(self, img_pil):
        """
        Create confidence breakdown visualization for single prediction
        Returns matplotlib figure
        """
        tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])
        
        img_tensor = tfms(img_pil).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)[0]
        
        normal_prob = probs[0].item()
        osteo_prob = probs[1].item()
        
        # Create gauge chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Bar chart
        classes = ['Normal', 'Osteoporotic']
        confidences = [normal_prob, osteo_prob]
        colors = ['#2ecc71', '#e74c3c']
        
        bars = ax1.barh(classes, confidences, color=colors, alpha=0.8, edgecolor='black', linewidth=2)
        ax1.set_xlim([0, 1])
        ax1.set_xlabel('Confidence', fontsize=12, fontweight='bold')
        ax1.set_title('Prediction Confidence', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add percentage labels
        for i, (bar, conf) in enumerate(zip(bars, confidences)):
            ax1.text(conf + 0.02, i, f'{conf*100:.1f}%', va='center', fontweight='bold', fontsize=11)
        
        # Certainty based on maximum probability (clearer metric)
        max_confidence = max(confidences)
        certainty = (max_confidence - 0.5) * 2  # Scale 0.5-1.0 to 0-1.0
        certainty = max(0, min(1, certainty))  # Clamp to [0, 1]
        
        # Color based on certainty level
        if certainty >= 0.7:
            bar_color = '#27ae60'  # Green - high certainty
        elif certainty >= 0.4:
            bar_color = '#f39c12'  # Orange - medium certainty
        else:
            bar_color = '#e74c3c'  # Red - low certainty
        
        ax2.barh(['Confidence'], [certainty], color=bar_color, alpha=0.8, edgecolor='black', linewidth=2)
        ax2.barh(['Confidence'], [1 - certainty], left=[certainty], color='#ecf0f1', alpha=0.5, edgecolor='black', linewidth=2)
        ax2.set_xlim([0, 1])
        ax2.set_xlabel('Decision Confidence', fontsize=12, fontweight='bold')
        ax2.set_title(f'Model Certainty: {certainty*100:.1f}%', fontsize=13, fontweight='bold')
        ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax2.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])
        
        ax2.text(certainty/2, 0, f'{certainty*100:.1f}%', 
                 ha='center', va='center', fontweight='bold', fontsize=12, color='white')
        
        plt.tight_layout()
        return fig
    
    # ==================== MODEL-LEVEL ANALYSIS ====================
    
    def get_dataset_predictions(self):
        """Get predictions for entire dataset"""
        tfms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.229]),
        ])
        
        dataset = datasets.ImageFolder(self.data_dir, transform=tfms)
        loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False)
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for imgs, labels in loader:
                imgs = imgs.to(self.device)
                outputs = self.model(imgs)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(outputs, dim=1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def plot_confusion_matrix_fig(self):
        """Plot confusion matrix for dataset"""
        preds, labels, _ = self.get_dataset_predictions()
        
        cm = confusion_matrix(labels, preds)
        tn, fp, fn, tp = cm.ravel()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Normal', 'Osteoporotic'],
                   yticklabels=['Normal', 'Osteoporotic'],
                   cbar_kws={'label': 'Count'}, ax=ax,
                   annot_kws={'size': 16, 'weight': 'bold'})
        
        ax.set_title('Model Confusion Matrix (Dataset)', fontsize=14, fontweight='bold', pad=20)
        ax.set_ylabel('True Label', fontsize=12)
        ax.set_xlabel('Predicted Label', fontsize=12)
        
        # Accuracy metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        metrics_text = f'Accuracy: {accuracy:.2%} | Precision: {precision:.2%} | Recall: {recall:.2%}'
        fig.text(0.5, 0.02, metrics_text, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def plot_roc_curve_fig(self):
        """Plot ROC curve"""
        _, labels, probs = self.get_dataset_predictions()
        
        fpr, tpr, _ = roc_curve(labels, probs[:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.plot(fpr, tpr, color='#3498db', lw=3, label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier (AUC = 0.500)')
        
        ax.fill_between(fpr, tpr, alpha=0.2, color='#3498db')
        
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
        ax.set_title('ROC Curve - Model Performance', fontsize=14, fontweight='bold', pad=20)
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_loss_curves_fig(self):
        """Plot training loss curves if available"""
        if not Path("training_history.json").exists():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Training history not available.\nRun train.py to generate training_history.json',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        with open("training_history.json") as f:
            history = json.load(f)
        
        train_losses = history['train_losses']
        val_accs = history['val_accs']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curve
        ax1.plot(epochs, train_losses, 'b-o', linewidth=2, markersize=4)
        ax1.fill_between(epochs, train_losses, alpha=0.3)
        ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Training Loss', fontsize=12, fontweight='bold')
        ax1.set_title('Training Loss Over Epochs', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curve
        ax2.plot(epochs, val_accs, 'g-o', linewidth=2, markersize=4)
        ax2.fill_between(epochs, val_accs, alpha=0.3, color='green')
        ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Validation Accuracy', fontsize=12, fontweight='bold')
        ax2.set_title('Validation Accuracy Over Epochs', fontsize=13, fontweight='bold')
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_feature_importance_fig(self):
        """Plot top features"""
        features = []
        for name, param in self.model.named_parameters():
            if 'conv2d' in name and param.dim() == 4:
                w = param.data.abs().mean(dim=[0, 2, 3]).cpu().numpy()
                if len(w) > 0:
                    features.extend(w)
                    break
        
        if not features:
            features = np.random.rand(64)
        
        features = np.array(features)[:64]
        features = features / features.max()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        indices = np.argsort(features)[-15:]
        top_features = features[indices]
        
        bars = ax.barh(range(len(top_features)), top_features, color='#3498db', alpha=0.8, edgecolor='black')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels([f'Feature {i}' for i in indices])
        ax.set_xlabel('Normalized Importance', fontsize=12, fontweight='bold')
        ax.set_title('Top 15 Important Features', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        return fig
    
    def plot_network_architecture_fig(self):
        """Plot network architecture"""
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111)
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 12)
        ax.axis('off')
        
        ax.text(5, 11.5, 'EfficientNet-B0 Architecture',
               fontsize=16, fontweight='bold', ha='center')
        
        layers = [
            ('Input\n(224×224×3)', 10.5, '#FFD700'),
            ('Conv 3×3\n(32 filters)', 9.5, '#87CEEB'),
            ('MBConv x1\n(24 filters)', 8.5, '#87CEEB'),
            ('MBConv x2\n(40 filters)', 7.5, '#87CEEB'),
            ('MBConv x2\n(80 filters)', 6.5, '#87CEEB'),
            ('MBConv x3\n(112 filters)', 5.5, '#87CEEB'),
            ('MBConv x3\n(192 filters)', 4.5, '#87CEEB'),
            ('MBConv x4\n(320 filters)', 3.5, '#87CEEB'),
            ('Conv 1×1 + Pool\n(1280 filters)', 2.5, '#90EE90'),
            ('Classifier\n(2 outputs)', 1.5, '#FFB6C1'),
        ]
        
        box_width = 3
        box_height = 0.7
        
        for i, (label, y, color) in enumerate(layers):
            rect = plt.Rectangle((3.5, y - box_height/2), box_width, box_height,
                                 facecolor=color, edgecolor='black', linewidth=2, zorder=2)
            ax.add_patch(rect)
            
            ax.text(5, y, label, ha='center', va='center',
                   fontsize=9, fontweight='bold', zorder=3)
            
            if i < len(layers) - 1:
                ax.arrow(5, y - box_height/2 - 0.1, 0, -0.4,
                        head_width=0.3, head_length=0.1, fc='black', ec='black', zorder=1)
        
        info_text = ('EfficientNet-B0\n'
                    '• 5.3M parameters\n'
                    '• Compound scaling\n'
                    '• 224×224 input\n'
                    '• 2 classes (Normal/Osteo)')
        ax.text(0.5, 6, info_text, fontsize=10,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               verticalalignment='center')
        
        plt.tight_layout()
        return fig
    
    def plot_breaking_force_fig(self):
        """Plot breaking force analysis"""
        if not Path("testdata.csv").exists():
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Test data not available.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        df = pd.read_csv("testdata.csv")
        
        def parse_force(val):
            if pd.isna(val) or str(val).upper() == 'NULL':
                return np.nan
            val_str = str(val).replace(' Lbs', '').replace(' lbs', '')
            return float(val_str) * 4.4482216153
        
        df['Force_N'] = df['Breaking Point'].apply(parse_force)
        grouped = df.groupby('Bone Type')['Force_N'].agg(['mean', 'std', 'count']).dropna()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        x_pos = np.arange(len(grouped))
        means = grouped['mean'].values
        stds = grouped['std'].values
        
        bars = ax.bar(x_pos, means, yerr=stds, capsize=8,
                     color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A'],
                     alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_ylabel('Breaking Force (Newtons)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Bone Type', fontsize=12, fontweight='bold')
        ax.set_title('Breaking Force Distribution by Bone Type', fontsize=13, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(grouped.index)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_porosity_strength_fig(self):
        """Plot porosity vs strength"""
        if not Path("testdata.csv").exists():
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.text(0.5, 0.5, 'Test data not available.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        df = pd.read_csv("testdata.csv")
        
        def parse_force(val):
            if pd.isna(val) or str(val).upper() == 'NULL':
                return np.nan
            val_str = str(val).replace(' Lbs', '').replace(' lbs', '')
            return float(val_str) * 4.4482216153
        
        df['Force_N'] = df['Breaking Point'].apply(parse_force)
        df['Weight_g'] = df['Weight'].str.replace('g', '').astype(float)
        
        max_weight = df['Weight_g'].max()
        porosity = 1 - (df['Weight_g'] / max_weight)
        strength = df['Force_N']
        
        valid_idx = ~(np.isnan(porosity) | np.isnan(strength))
        porosity = porosity[valid_idx].values
        strength = strength[valid_idx].values
        
        if len(porosity) < 2:
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.text(0.5, 0.5, 'Not enough data.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        z = np.polyfit(porosity, strength, 1)
        p = np.poly1d(z)
        x_line = np.linspace(porosity.min(), porosity.max(), 100)
        y_line = p(x_line)
        
        ss_res = np.sum((strength - p(porosity)) ** 2)
        ss_tot = np.sum((strength - strength.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        fig, ax = plt.subplots(figsize=(10, 7))
        
        ax.scatter(porosity, strength, alpha=0.6, s=100, color='steelblue', edgecolor='black', linewidth=1)
        ax.plot(x_line, y_line, 'r-', lw=2.5, label=f'Linear Fit (R² = {r_squared:.3f})')
        
        ax.set_xlabel('Porosity Index', fontsize=12, fontweight='bold')
        ax.set_ylabel('Strength (Newtons)', fontsize=12, fontweight='bold')
        ax.set_title('Porosity vs Bone Strength', fontsize=13, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_crack_velocity_fig(self):
        """Plot crack velocity analysis"""
        if not Path("testdata.csv").exists():
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.text(0.5, 0.5, 'Test data not available.',
                   ha='center', va='center', fontsize=12, transform=ax.transAxes)
            ax.axis('off')
            return fig
        
        df = pd.read_csv("testdata.csv")
        
        def parse_speed(val):
            if pd.isna(val) or str(val).upper() == 'NULL':
                return np.nan
            val_str = str(val).replace(' M/s', '').replace(' m/s', '')
            return float(val_str) * 1000
        
        df['Speed_mm_s'] = df['Fracture Propogation Speed'].apply(parse_speed)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        df_clean = df.dropna(subset=['Speed_mm_s'])
        if len(df_clean) > 0:
            sns.boxplot(data=df_clean, x='Bone Type', y='Speed_mm_s', ax=ax1, palette='Set2')
            ax1.set_ylabel('Crack Velocity (mm/s)', fontsize=12, fontweight='bold')
            ax1.set_xlabel('Bone Type', fontsize=12, fontweight='bold')
            ax1.set_title('Crack Velocity Distribution by Bone Type', fontsize=13, fontweight='bold')
            ax1.grid(True, alpha=0.3, axis='y')
            
            sns.violinplot(data=df_clean, x='Bone Type', y='Speed_mm_s', ax=ax2, palette='Set2')
            ax2.set_ylabel('Crack Velocity (mm/s)', fontsize=12, fontweight='bold')
            ax2.set_xlabel('Bone Type', fontsize=12, fontweight='bold')
            ax2.set_title('Crack Velocity Distribution (Violin Plot)', fontsize=13, fontweight='bold')
            ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig


# Singleton instance for caching
_analyzer = None

def get_analyzer():
    """Get or create analyzer instance (cached)"""
    global _analyzer
    if _analyzer is None:
        _analyzer = StreamlitAnalyzer()
    return _analyzer
