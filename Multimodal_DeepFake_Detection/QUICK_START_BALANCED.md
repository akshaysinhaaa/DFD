# Quick Start Guide - Improved Training with Class Balancing

## 🎯 What's Different?

This improved version (`14_Complete_All_Datasets copy.ipynb`) includes:

1. ✅ **Focal Loss** - Handles class imbalance automatically
2. ✅ **Balanced Sampling** - Equal Real/Fake in each batch  
3. ✅ **Threshold Tuning** - Find optimal decision boundary
4. ✅ **Better Metrics** - F1 score prioritized over accuracy

## 🚀 How to Run

### Step 1: Open the Notebook
```
File: Multimodal_DeepFake_Detection/14_Complete_All_Datasets copy.ipynb
```

### Step 2: Run All Cells
Just execute cells in order. Key sections:

1. **Setup & Imports** (Cells 1-10)
2. **Model Definition** (Cells 11-15)
3. **Dataset Loading + Balancing** ⭐ NEW (Cell 22)
4. **Training Loop + Focal Loss** ⭐ IMPROVED (Cell 26)
5. **Results Analysis** (Cells 27-28)
6. **Threshold Tuning** ⭐ NEW (Cell 29)
7. **Visualizations** ⭐ NEW (Cell 30)

### Step 3: Monitor Training
Watch for these metrics in progress bars:
- **Loss**: Should decrease steadily
- **Accuracy**: Should improve (but F1 is more important)
- **F1 Score**: Primary metric for imbalanced data

### Step 4: Check Results
After training, the notebook shows:
- Best F1 score and accuracy
- Precision vs Recall balance
- Optimal threshold from ROC curve
- Confusion matrix

## 📊 Expected Output

### Class Distribution
```
📊 Class Distribution:
   Real (0): 3,677 samples
   Fake (1): 33,425 samples
   Imbalance Ratio: 9.09:1

⚖️ Pos Weight for BCE Loss: 0.1100
```

### Training Progress
```
EPOCH 1/10
========================================
[TRAINING]
  GRL Alpha: 0.0000
  Training: 100%|████████| Loss: 0.3245 | Acc: 85.32%
  
  >>> TRAINING RESULTS:
      Total Loss:    0.3245
      Class Loss:    0.2987
      Accuracy:      85.32%

[EVALUATION]
  Testing: 100%|████████| Acc: 82.15%
  
  >>> TEST RESULTS:
      Accuracy:  82.15%
      Precision: 76.83%
      Recall:    71.25%
      F1 Score:  73.94%

  ✅ NEW BEST MODEL SAVED! Test F1: 73.94% | Accuracy: 82.15%
```

### Threshold Optimization
```
📊 ROC Analysis:
   ROC AUC Score: 0.8542
   Current threshold: 0.5000
   Optimal threshold: 0.4237
   Improvement: 0.0763

PERFORMANCE AT DIFFERENT THRESHOLDS
================================================
Threshold    Accuracy     Precision    Recall       F1          
------------------------------------------------
0.3000       79.45        68.23        82.15        74.58       
0.4000       82.67        72.45        78.92        75.55       
0.5000       82.15        76.83        71.25        73.94        ← DEFAULT
0.4237       84.32        74.15        80.67        77.28        ← OPTIMAL
0.6000       80.23        81.45        65.32        72.52       
0.7000       76.89        86.78        58.43        69.87       
```

## 📈 What to Look For

### Good Signs ✅
- F1 score > 70%
- Precision and Recall within 10% of each other
- ROC AUC > 0.80
- Recall on Real class > 60%

### Warning Signs ⚠️
- F1 score < 60% → May need more epochs or hyperparameter tuning
- Large Precision/Recall gap (>20%) → Adjust threshold
- Decreasing validation accuracy → May be overfitting

## 🎛️ Hyperparameter Tuning

If results aren't good enough, try adjusting:

### Focal Loss Parameters
```python
focal_loss_fn = FocalLoss(alpha=0.75, gamma=2.0)

# Increase alpha (0.75 → 0.85) to favor minority class more
# Increase gamma (2.0 → 3.0) to focus more on hard examples
```

### Training Settings
```python
config.learning_rate = 1e-4  # Try 5e-5 or 2e-4
config.batch_size = 2        # Increase if you have GPU memory
config.epochs = 10           # Try 15-20 for better convergence
```

### Threshold
```python
# Use threshold from ROC curve analysis
# For high precision: use higher threshold (0.6-0.7)
# For high recall: use lower threshold (0.3-0.4)
# For balanced F1: use optimal threshold from ROC
```

## 💾 Model Files

### Generated Files
After training completes, you'll have:

1. **best_multimodal_balanced_focal_loss.pth** - Best model checkpoint
2. **training_progress_balanced.png** - Learning curves
3. **roc_curve_optimal_threshold.png** - ROC analysis

### Loading the Model
```python
import torch

# Load checkpoint
checkpoint = torch.load('best_multimodal_balanced_focal_loss.pth')

# Restore model
model.load_state_dict(checkpoint['model_state_dict'])

# Get optimal threshold
optimal_threshold = checkpoint.get('optimal_threshold', 0.5)

# Inference
with torch.no_grad():
    outputs = model(images=images, audio=audio)
    probs = torch.sigmoid(outputs['logits'])
    preds = (probs > optimal_threshold).float()
```

## 🔍 Troubleshooting

### Issue: Out of Memory
```python
# Reduce batch size
config.batch_size = 1

# Or use gradient accumulation
config.gradient_accumulation_steps = 8
```

### Issue: Training Too Slow
```python
# Reduce model size
config = ModelConfig.from_gpu_memory(gpu_memory_gb=10)  # Forces small model

# Or reduce epochs
config.epochs = 5
```

### Issue: Poor Recall on Real Class
```python
# Increase alpha in Focal Loss
focal_loss_fn = FocalLoss(alpha=0.85, gamma=2.0)

# Or use lower threshold
threshold = 0.3  # From ROC curve analysis
```

### Issue: High Variance in Results
```python
# Add more regularization
config.dropout = 0.2  # Increase from 0.1
config.weight_decay = 1e-3  # Increase from 1e-4
```

## 📞 Support

For questions or issues:
1. Check `CLASS_BALANCING_IMPROVEMENTS.md` for detailed explanation
2. Review training logs for error messages
3. Verify dataset paths are correct
4. Ensure GPU has sufficient memory

## 🎓 Key Takeaways

1. **F1 Score > Accuracy** for imbalanced datasets
2. **Threshold tuning** is free performance boost
3. **Balanced sampling** + **Focal Loss** = Best results
4. **ROC curve** guides threshold selection
5. **Precision/Recall trade-off** depends on use case

---

**Happy Training! 🚀**
