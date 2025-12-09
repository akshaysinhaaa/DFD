# 📋 Copy-Paste These Cells into Notebook 15

## Current Status: 11 cells complete
## Add these cells: 12-30 (training, testing, evaluation)

---

## CELL 12: Class Balancing Strategies (Markdown)

Copy this as a **Markdown cell**:

```markdown
## ⚖️ Class Balancing Strategies

Multiple strategies to handle severe class imbalance:
1. **WeightedRandomSampler** - Oversamples minority class during training
2. **SMOTE** - Synthetic Minority Over-sampling Technique
3. **Data Augmentation** - Enhanced augmentation for minority class
4. **Hybrid Approach** - Combination strategies
```

---

## CELL 13: Class Balancing Functions (Code)

Copy this as a **Code cell**:

```python
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

print("✅ Class balancing strategies ready!")
```

---

## CELL 14: Threshold Tuning Header (Markdown)

```markdown
## 🎚️ Threshold Tuning (No Retraining Required)

Optimize decision threshold for better performance on imbalanced data:
- **F1-Score Optimization** - Maximize F1 score
- **Youden's J Statistic** - Maximize (Sensitivity + Specificity - 1)
- **Precision-Recall Trade-off** - Find optimal point on PR curve

This is a **quick fix** that requires NO retraining!
```

---

## CELL 15: Threshold Tuning Functions (Code)

```python
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

def plot_threshold_analysis(y_true, y_probs, title='Threshold Analysis', save_path=None):
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved to {save_path}")
    plt.show()
    
    # Print summary
    print("\\n" + "="*70)
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
    
    print(f"\\n🎯 F1 Improvement: {improvement:+.2f}%")
    print(f"   Default (0.5):  F1 = {f1_default:.4f}")
    print(f"   Optimal ({opt_thresh_f1:.3f}): F1 = {f1_optimal:.4f}")
    print("="*70 + "\\n")
    
    return opt_thresh_f1, opt_thresh_youden

print("✅ Threshold tuning functions ready!")
```

---

**Continue to next file for cells 16-30 (Dataset classes, Models, Training, Evaluation)...**
