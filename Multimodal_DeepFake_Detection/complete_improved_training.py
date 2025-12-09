"""
Complete Improved Deepfake Detection Training Script
Addresses: Class Imbalance, Focal Loss, Threshold Tuning, Comprehensive Analysis

Usage: python complete_improved_training.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
from torchvision import models

# Image/Video/Audio processing
import cv2
from PIL import Image
import librosa

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("\n" + "="*80)
print("🎯 COMPLETE IMPROVED DEEPFAKE DETECTION")
print("="*80)
print("\n✅ Issues Addressed:")
print("   1. Class Balancing (Highest Priority)")
print("   2. Focal Loss (Hard Examples)")
print("   3. Class Weights (Simplest Fix)")
print("   4. Threshold Tuning (Quick Fix)")
print("   5. Comprehensive Analysis")
print("\n" + "="*80 + "\n")

# =============================================================================
# DATASET PATHS
# =============================================================================
DATASET_PATHS = {
    'deepfake_images': {
        'train_real': '../Deepfake image detection dataset/train-20250112T065955Z-001/train/real',
        'train_fake': '../Deepfake image detection dataset/train-20250112T065955Z-001/train/fake',
        'test_real': '../Deepfake image detection dataset/test-20250112T065939Z-001/test/real',
        'test_fake': '../Deepfake image detection dataset/test-20250112T065939Z-001/test/fake'
    },
    'faceforensics': {
        'original': '../FaceForensics++/FaceForensics++_C23/original',
        'deepfakes': '../FaceForensics++/FaceForensics++_C23/Deepfakes',
        'face2face': '../FaceForensics++/FaceForensics++_C23/Face2Face',
        'faceswap': '../FaceForensics++/FaceForensics++_C23/FaceSwap',
        'neuraltextures': '../FaceForensics++/FaceForensics++_C23/NeuralTextures'
    },
    'celebdf': {
        'celeb_real': '../Celeb V2/Celeb-real',
        'youtube_real': '../Celeb V2/YouTube-real',
        'celeb_synthesis': '../Celeb V2/Celeb-synthesis'
    },
    'dfd': {
        'original': '../DFD/DFD_original sequences',
        'manipulated': '../DFD/DFD_manipulated_sequences/DFD_manipulated_sequences'
    },
    'audio': {
        'real': '../DeepFake_AudioDataset/KAGGLE/AUDIO/REAL',
        'fake': '../DeepFake_AudioDataset/KAGGLE/AUDIO/FAKE'
    },
    'fakeavceleb': {
        'real_av': '../FakeAVCeleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/RealVideo-RealAudio',
        'fake_vv_aa': '../FakeAVCeleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/FakeVideo-FakeAudio',
        'fake_v_ra': '../FakeAVCeleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/FakeVideo-RealAudio',
        'fake_rv_a': '../FakeAVCeleb/FakeAVCeleb_v1.2/FakeAVCeleb_v1.2/RealVideo-FakeAudio'
    }
}

# =============================================================================
# FOCAL LOSS IMPLEMENTATION
# =============================================================================
class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance."""
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

class WeightedFocalLoss(nn.Module):
    """Focal Loss with class weights."""
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.weight is not None:
            weights = self.weight[targets.long()]
            F_loss = F_loss * weights
        
        if self.reduction == 'mean':
            return F_loss.mean()
        elif self.reduction == 'sum':
            return F_loss.sum()
        else:
            return F_loss

# =============================================================================
# CLASS BALANCING FUNCTIONS
# =============================================================================
def compute_class_weights_from_labels(labels):
    """Compute class weights for imbalanced datasets."""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    print(f"📊 Class weights computed: {class_weights}")
    return torch.FloatTensor(class_weights)

def create_weighted_sampler(labels):
    """Create WeightedRandomSampler for balanced batch sampling."""
    class_counts = np.bincount(labels)
    print(f"📊 Class distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
    print(f"📊 Imbalance ratio: {class_counts[0]/class_counts[1]:.2f}:1")
    
    class_weights = 1. / class_counts
    sample_weights = class_weights[labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    print("✅ WeightedRandomSampler created")
    return sampler

def balance_dataset_with_smote(X, y, strategy='auto'):
    """Balance dataset using SMOTE."""
    print(f"Before SMOTE - Class distribution: {np.bincount(y)}")
    try:
        smote = SMOTE(sampling_strategy=strategy, random_state=42, k_neighbors=5)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        print(f"After SMOTE - Class distribution: {np.bincount(y_balanced)}")
        print("✅ SMOTE balancing completed")
        return X_balanced, y_balanced
    except Exception as e:
        print(f"⚠️ SMOTE failed: {e}. Returning original data.")
        return X, y

# =============================================================================
# THRESHOLD TUNING FUNCTIONS
# =============================================================================
def find_optimal_threshold_f1(y_true, y_probs):
    """Find threshold that maximizes F1 score."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        f1_scores.append(f1)
    optimal_idx = np.argmax(f1_scores)
    return thresholds[optimal_idx], f1_scores[optimal_idx], thresholds, f1_scores

def find_optimal_threshold_youden(y_true, y_probs):
    """Find threshold using Youden's J statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    return thresholds[optimal_idx], j_scores[optimal_idx]

def plot_threshold_analysis(y_true, y_probs, title='Threshold Analysis', save_path='threshold_analysis.png'):
    """Comprehensive threshold analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # F1 vs Threshold
    opt_thresh_f1, opt_f1, thresholds, f1_scores = find_optimal_threshold_f1(y_true, y_probs)
    axes[0, 0].plot(thresholds, f1_scores, 'b-', linewidth=2.5)
    axes[0, 0].axvline(opt_thresh_f1, color='r', linestyle='--', linewidth=2, 
                       label=f'Optimal: {opt_thresh_f1:.3f} (F1={opt_f1:.3f})')
    axes[0, 0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2.5, label=f'AUC = {roc_auc:.3f}')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('ROC Curve', fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    axes[1, 0].plot(recall, precision, 'b-', linewidth=2.5, label=f'AP = {avg_precision:.3f}')
    axes[1, 0].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # All Metrics vs Threshold
    thresholds_metric = np.arange(0.1, 0.9, 0.01)
    precisions, recalls, accuracies, f1s = [], [], [], []
    for thresh in thresholds_metric:
        y_pred = (y_probs >= thresh).astype(int)
        precisions.append(precision_score(y_true, y_pred, zero_division=0))
        recalls.append(recall_score(y_true, y_pred, zero_division=0))
        accuracies.append(accuracy_score(y_true, y_pred))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    
    axes[1, 1].plot(thresholds_metric, precisions, 'r-', linewidth=2, label='Precision')
    axes[1, 1].plot(thresholds_metric, recalls, 'g-', linewidth=2, label='Recall')
    axes[1, 1].plot(thresholds_metric, accuracies, 'b-', linewidth=2, label='Accuracy')
    axes[1, 1].plot(thresholds_metric, f1s, 'purple', linewidth=2, label='F1')
    axes[1, 1].axvline(opt_thresh_f1, color='orange', linestyle='--', linewidth=2)
    axes[1, 1].axvline(0.5, color='gray', linestyle=':', linewidth=1.5, label='Default (0.5)')
    axes[1, 1].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Score', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('All Metrics vs Threshold', fontsize=14, fontweight='bold')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved to {save_path}")
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 THRESHOLD OPTIMIZATION RESULTS")
    print("="*70)
    print(f"F1-Optimal Threshold:     {opt_thresh_f1:.4f} (F1: {opt_f1:.4f})")
    opt_thresh_youden, opt_j = find_optimal_threshold_youden(y_true, y_probs)
    print(f"Youden-Optimal Threshold: {opt_thresh_youden:.4f} (J: {opt_j:.4f})")
    print(f"Default Threshold:        0.5000")
    
    # Show improvement
    y_pred_default = (y_probs >= 0.5).astype(int)
    y_pred_optimal = (y_probs >= opt_thresh_f1).astype(int)
    f1_default = f1_score(y_true, y_pred_default)
    f1_optimal = f1_score(y_true, y_pred_optimal)
    improvement = ((f1_optimal - f1_default) / f1_default) * 100 if f1_default > 0 else 0
    
    print(f"\n🎯 F1 Improvement: {improvement:+.2f}%")
    print(f"   Default (0.5):  F1 = {f1_default:.4f}")
    print(f"   Optimal ({opt_thresh_f1:.3f}): F1 = {f1_optimal:.4f}")
    print("="*70 + "\n")
    
    return opt_thresh_f1, opt_thresh_youden

# =============================================================================
# CONTINUE IN NEXT FILE...
# =============================================================================
print("✅ Core functions loaded successfully!")
print("\n📝 This script contains:")
print("   ✓ Focal Loss (standard + weighted)")
print("   ✓ Class balancing functions")
print("   ✓ Threshold tuning functions")
print("   ✓ Comprehensive analysis plots")
print("\n⏭️  Continue with dataset classes, models, and training loop...")
