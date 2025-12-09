# 📓 Complete Notebook Cells - Copy & Paste Guide

Your notebook currently has **11 cells** with:
- ✅ Title & Introduction
- ✅ Installation
- ✅ Imports
- ✅ Dataset Paths
- ✅ Utility Functions (dataset statistics)
- ✅ Focal Loss Implementation

## 🎯 What's Already in the Notebook (Cells 1-11)

The notebook already contains:
1. Title and improvements overview
2. Installation cell
3. Imports cell
4. Dataset paths configuration
5. Utility functions for counting files
6. Comprehensive dataset statistics function
7. Focal Loss implementation

## 📋 Remaining Cells to Add

Since we can't run Python directly, here's what you need to add manually to complete the notebook:

---

## Cell 12: Class Balancing Strategies (Markdown)

```markdown
## ⚖️ Class Balancing Strategies

Multiple strategies to handle severe class imbalance:
1. **WeightedRandomSampler** - Oversamples minority class during training
2. **SMOTE** - Synthetic Minority Over-sampling Technique  
3. **Data Augmentation** - Enhanced augmentation for minority class
4. **Hybrid Approach** - Combination strategies
```

---

## Cell 13: Class Balancing Functions (Code)

```python
def compute_class_weights_from_labels(labels):
    """Compute class weights for imbalanced datasets."""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight('balanced', classes=unique_labels, y=labels)
    print(f"📊 Class weights computed: {class_weights}")
    return torch.FloatTensor(class_weights)

def create_weighted_sampler(labels):
    """Create WeightedRandomSampler for balanced batch sampling.
    
    This ensures each batch has balanced representation of both classes.
    """
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
    """Balance dataset using SMOTE (Synthetic Minority Over-sampling).
    
    SMOTE creates synthetic examples of minority class by interpolating
    between existing minority samples.
    """
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

def hybrid_sampling(X, y, oversample_ratio=0.5, undersample_ratio=0.5):
    """Hybrid: undersample majority + oversample minority.
    
    This reduces the majority class while increasing minority class.
    """
    print(f"Before hybrid sampling: {np.bincount(y)}")
    
    # First undersample majority class
    rus = RandomUnderSampler(sampling_strategy=undersample_ratio, random_state=42)
    X_under, y_under = rus.fit_resample(X, y)
    print(f"After undersampling: {np.bincount(y_under)}")
    
    # Then oversample minority class
    ros = RandomOverSampler(sampling_strategy=oversample_ratio, random_state=42)
    X_balanced, y_balanced = ros.fit_resample(X_under, y_under)
    print(f"After oversampling: {np.bincount(y_balanced)}")
    print("✅ Hybrid sampling completed")
    
    return X_balanced, y_balanced

print("✅ Class balancing strategies ready!")
```

---

## Cell 14: Threshold Tuning Header (Markdown)

```markdown
## 🎚️ Threshold Tuning (No Retraining Required)

Optimize decision threshold for better performance on imbalanced data:
- **F1-Score Optimization** - Maximize F1 score
- **Youden's J Statistic** - Maximize (Sensitivity + Specificity - 1)
- **Precision-Recall Trade-off** - Find optimal point on PR curve

This is a **quick fix** that requires NO retraining - just run on validation set!
```

---

## Cell 15: Threshold Tuning Functions (Code)

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
    optimal_threshold = thresholds[optimal_idx]
    optimal_f1 = f1_scores[optimal_idx]
    
    return optimal_threshold, optimal_f1, thresholds, f1_scores

def find_optimal_threshold_youden(y_true, y_probs):
    """Find threshold using Youden's J statistic.
    
    J = Sensitivity + Specificity - 1
    Maximizes the distance from the diagonal in ROC space.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, j_scores[optimal_idx]

def find_optimal_threshold_precision_recall(y_true, y_probs, min_precision=0.9):
    """Find threshold maintaining minimum precision while maximizing recall."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    
    # Find thresholds where precision >= min_precision
    valid_indices = np.where(precision[:-1] >= min_precision)[0]
    
    if len(valid_indices) == 0:
        print(f"⚠️ No threshold found with precision >= {min_precision}")
        return 0.5, 0, 0
    
    # Among valid thresholds, choose one with highest recall
    best_idx = valid_indices[np.argmax(recall[valid_indices])]
    optimal_threshold = thresholds[best_idx]
    
    return optimal_threshold, precision[best_idx], recall[best_idx]

def plot_threshold_analysis(y_true, y_probs, title='Threshold Analysis', save_path=None):
    """Comprehensive threshold analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. F1 Score vs Threshold
    opt_thresh_f1, opt_f1, thresholds, f1_scores = find_optimal_threshold_f1(y_true, y_probs)
    axes[0, 0].plot(thresholds, f1_scores, 'b-', linewidth=2.5)
    axes[0, 0].axvline(opt_thresh_f1, color='r', linestyle='--', linewidth=2, 
                       label=f'Optimal: {opt_thresh_f1:.3f} (F1={opt_f1:.3f})')
    axes[0, 0].axhline(opt_f1, color='r', linestyle='--', alpha=0.3, linewidth=1.5)
    axes[0, 0].set_xlabel('Threshold', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('F1 Score vs Threshold', fontsize=14, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    opt_thresh_youden, opt_j = find_optimal_threshold_youden(y_true, y_probs)
    axes[0, 1].plot(fpr, tpr, 'b-', linewidth=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[0, 1].plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=1.5)
    axes[0, 1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    axes[0, 1].set_title(f'ROC Curve (Youden: {opt_thresh_youden:.3f})', 
                         fontsize=14, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    avg_precision = average_precision_score(y_true, y_probs)
    axes[1, 0].plot(recall, precision, 'b-', linewidth=2.5, 
                    label=f'PR (AP = {avg_precision:.3f})')
    axes[1, 0].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. All Metrics vs Threshold
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
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"💾 Saved plot to {save_path}")
    
    plt.show()
    
    # Print summary
    print("\n" + "="*70)
    print("📊 THRESHOLD OPTIMIZATION RESULTS")
    print("="*70)
    print(f"F1-Optimal Threshold:     {opt_thresh_f1:.4f} (F1: {opt_f1:.4f})")
    print(f"Youden-Optimal Threshold: {opt_thresh_youden:.4f} (J: {opt_j:.4f})")
    print(f"Default Threshold:        0.5000")
    print("="*70)
    
    # Show improvement
    y_pred_default = (y_probs >= 0.5).astype(int)
    y_pred_optimal = (y_probs >= opt_thresh_f1).astype(int)
    
    f1_default = f1_score(y_true, y_pred_default)
    f1_optimal = f1_score(y_true, y_pred_optimal)
    improvement = ((f1_optimal - f1_default) / f1_default) * 100
    
    print(f"\n🎯 F1 Improvement: {improvement:+.2f}%")
    print(f"   Default (0.5):  F1 = {f1_default:.4f}")
    print(f"   Optimal ({opt_thresh_f1:.3f}): F1 = {f1_optimal:.4f}")
    print("="*70 + "\n")
    
    return opt_thresh_f1, opt_thresh_youden

print("✅ Threshold tuning functions ready!")
```

---

## ⚡ Quick Start - What to Do Now

Since the notebook building is challenging with PowerShell, I recommend:

### Option 1: Use Jupyter Notebook/Lab Directly (RECOMMENDED)
1. Open Jupyter Notebook
2. Open `15_Complete_Improved_All_Datasets.ipynb`
3. Add the cells above (12-15) manually by:
   - Click "+ Code" or "+ Markdown" button
   - Copy-paste the content from above
   - Run each cell to verify

### Option 2: I'll create a complete standalone Python script
Instead of a notebook, I can create a complete `.py` script that:
- Loads all datasets
- Shows statistics with imbalance ratios
- Trains models with Focal Loss + class balancing
- Optimizes thresholds
- Generates comprehensive analysis

Would you like me to create this complete Python script instead?

### Option 3: Download Additional Balanced Datasets First
Based on the severe imbalance we expect, download these from Kaggle:
1. **140K Real and Fake Faces** - Perfectly balanced image dataset
2. **ASVspoof 2019** or **WaveFake** - Balanced audio dataset

This will make training much more effective!

---

## 📝 Summary of Current Status

✅ **What's Complete:**
- Notebook structure created with 11 cells
- Dataset path configuration
- File counting utilities
- Comprehensive statistics display
- Focal Loss implementation
- Guide documents created

⏳ **What's Remaining:**
- Add cells 12-15 (class balancing + threshold tuning)
- Create dataset loader classes
- Implement model architectures
- Training loops with all improvements
- Evaluation and analysis code

🎯 **Your Next Steps:**
1. Run the first 11 cells to see your dataset statistics
2. Check the imbalance ratios
3. Decide: Download balanced datasets OR proceed with balancing techniques
4. Add remaining cells manually in Jupyter
5. Start training!

---

Would you like me to create a **complete standalone Python script** that does everything in one file? This might be easier than building the notebook!
