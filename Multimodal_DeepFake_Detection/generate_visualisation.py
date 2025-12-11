import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.metrics import precision_score, recall_score, f1_score
from dataclasses import dataclass

# Define ModelConfig (needed for unpickling the checkpoint)
@dataclass
class ModelConfig:
    """Configuration for model architecture"""
    preset: str = "large"
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    vision_backbone: str = "vit_base_patch16_224"
    audio_backbone: str = "facebook/wav2vec2-large-960h"
    text_backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_vision: bool = True
    freeze_audio: bool = True
    freeze_text: bool = True
    batch_size: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    gradient_accumulation_steps: int = 4
    alpha_domain: float = 0.5
    k_frames: int = 5
    k_audio_chunks: int = 5
    sample_rate: int = 16000
    image_size: int = 224
    max_text_tokens: int = 256

# Load checkpoint
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
checkpoint = torch.load('best_multimodal_12datasets_balanced.pth', map_location=device, weights_only=False)

print("="*60)
print("GENERATING VISUALIZATIONS FROM SAVED MODEL")
print("="*60)

# Extract data from checkpoint
epoch = checkpoint['epoch']
test_acc = checkpoint['test_acc']
test_f1 = checkpoint['test_f1']
test_precision = checkpoint['test_precision']
test_recall = checkpoint['test_recall']
all_labels = np.array(checkpoint['all_labels'])
all_probs = np.array(checkpoint['all_probs']).squeeze()

print(f"\n✅ Loaded checkpoint from Epoch {epoch}")
print(f"   Test Accuracy: {test_acc:.2f}%")
print(f"   Test F1 Score: {test_f1:.2f}%")
print(f"   Test Precision: {test_precision:.2f}%")
print(f"   Test Recall: {test_recall:.2f}%")

# ========== PLOT 1: Metrics Bar Chart ==========
plt.figure(figsize=(10, 6))
metrics = {
    'Accuracy': test_acc,
    'Precision': test_precision,
    'Recall': test_recall,
    'F1 Score': test_f1
}
colors = ['#3498db', '#9b59b6', '#f39c12', '#1abc9c']
bars = plt.bar(metrics.keys(), metrics.values(), color=colors, alpha=0.8, edgecolor='black', linewidth=2)
plt.ylabel('Score (%)', fontsize=12, fontweight='bold')
plt.title(f'Test Metrics (Epoch {epoch})', fontsize=14, fontweight='bold')
plt.ylim([0, 100])
plt.grid(True, alpha=0.3, axis='y')

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
            f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig('test_metrics.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: test_metrics.png")
plt.close()

# ========== PLOT 2: ROC Curve ==========
fpr, tpr, thresholds = roc_curve(all_labels, all_probs)
roc_auc = roc_auc_score(all_labels, all_probs)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]

plt.figure(figsize=(10, 6))
plt.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.4f})', color='#3498db')
plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random classifier')
plt.scatter(fpr[optimal_idx], tpr[optimal_idx], s=200, c='red', marker='*', 
            label=f'Optimal threshold = {optimal_threshold:.4f}', zorder=5)
plt.xlabel('False Positive Rate', fontsize=12, fontweight='bold')
plt.ylabel('True Positive Rate', fontsize=12, fontweight='bold')
plt.title('ROC Curve', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
print("✅ Saved: roc_curve.png")
plt.close()

# ========== PLOT 3: Confusion Matrix (at threshold 0.5) ==========
preds_default = (all_probs > 0.5).astype(float)
cm_default = confusion_matrix(all_labels, preds_default)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_default, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.title('Confusion Matrix (Threshold = 0.5)', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_default.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix_default.png")
plt.close()

# ========== PLOT 4: Confusion Matrix (at optimal threshold) ==========
preds_optimal = (all_probs > optimal_threshold).astype(float)
cm_optimal = confusion_matrix(all_labels, preds_optimal)

plt.figure(figsize=(10, 8))
sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', 
            xticklabels=['Real', 'Fake'], 
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'},
            annot_kws={'fontsize': 14, 'fontweight': 'bold'})
plt.title(f'Confusion Matrix (Optimal Threshold = {optimal_threshold:.4f})', fontsize=14, fontweight='bold')
plt.ylabel('True Label', fontsize=12, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix_optimal.png', dpi=300, bbox_inches='tight')
print("✅ Saved: confusion_matrix_optimal.png")
plt.close()

# ========== PLOT 5: Prediction Distribution ==========
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Real samples
real_probs = all_probs[all_labels == 0]
axes[0].hist(real_probs, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
axes[0].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
axes[0].axvline(optimal_threshold, color='orange', linestyle='--', linewidth=2, label=f'Optimal = {optimal_threshold:.3f}')
axes[0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[0].set_title('Real Samples Distribution', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Fake samples
fake_probs = all_probs[all_labels == 1]
axes[1].hist(fake_probs, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
axes[1].axvline(0.5, color='red', linestyle='--', linewidth=2, label='Threshold = 0.5')
axes[1].axvline(optimal_threshold, color='orange', linestyle='--', linewidth=2, label=f'Optimal = {optimal_threshold:.3f}')
axes[1].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=11, fontweight='bold')
axes[1].set_title('Fake Samples Distribution', fontsize=12, fontweight='bold')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_distribution.png', dpi=300, bbox_inches='tight')
print("✅ Saved: prediction_distribution.png")
plt.close()

# ========== Print Summary ==========
print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)
print(f"Epoch: {epoch}")
print(f"Accuracy: {test_acc:.2f}%")
print(f"Precision: {test_precision:.2f}%")
print(f"Recall: {test_recall:.2f}%")
print(f"F1 Score: {test_f1:.2f}%")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Optimal Threshold: {optimal_threshold:.4f}")

print(f"\n--- At Default Threshold (0.5) ---")
print(classification_report(all_labels, preds_default, target_names=['Real', 'Fake'], digits=4))

print(f"\n--- At Optimal Threshold ({optimal_threshold:.4f}) ---")
print(classification_report(all_labels, preds_optimal, target_names=['Real', 'Fake'], digits=4))

print("="*60)
print("\n✅ All visualizations saved successfully!")
print("\nGenerated files:")
print("  1. test_metrics.png - Bar chart of all metrics")
print("  2. roc_curve.png - ROC curve with AUC")
print("  3. confusion_matrix_default.png - CM at threshold 0.5")
print("  4. confusion_matrix_optimal.png - CM at optimal threshold")
print("  5. prediction_distribution.png - Probability distributions")