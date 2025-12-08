# Research Paper Assembly Guide

## 📄 Complete IEEE Conference Paper

Your research paper has been created in modular LaTeX files for easy editing and compilation.

---

## 📁 Paper Files Structure

```
Multimodal_DeepFake_Detection/
├── research_paper.tex              # Main file with title, abstract, intro
├── paper_methodology.tex           # Encoders, fusion, GRL (with equations)
├── paper_training.tex              # Training, loss functions, algorithm
├── paper_experiments.tex           # Datasets, baselines, main results
├── paper_results_discussion.tex    # Detailed results, ablations, discussion
├── paper_conclusion.tex            # Conclusion, references, end matter
└── PAPER_ASSEMBLY_GUIDE.md         # This file
```

---

## 🔧 How to Assemble the Complete Paper

### Option 1: Manual Assembly (Recommended)

1. **Open** `research_paper.tex` in your LaTeX editor
2. **Before** `\end{document}`, add these lines:

```latex
% Include all sections
\input{paper_methodology}
\input{paper_training}
\input{paper_experiments}
\input{paper_results_discussion}
\input{paper_conclusion}
```

3. **Compile** with pdflatex or your preferred LaTeX compiler

### Option 2: Create Complete File

Copy all content from individual files into one `complete_paper.tex`:

1. Start with `research_paper.tex`
2. Insert content from `paper_methodology.tex` (after Introduction)
3. Insert content from `paper_training.tex`
4. Insert content from `paper_experiments.tex`
5. Insert content from `paper_results_discussion.tex`
6. Insert content from `paper_conclusion.tex`
7. Keep only one `\end{document}` at the very end

---

## 📊 Figures and Images to Add

The paper has **PLACEHOLDERS** for the following figures. You need to create/add:

### Required Figures:

1. **`figures/architecture.pdf`** (Line ~180)
   - Your draw.io architecture diagram
   - Shows: Encoders → Fusion → GRL → Classifiers
   - **Size**: Full page width (7 inches)

2. **`figures/performance_comparison.pdf`** (Results section)
   - Bar chart comparing all methods
   - X-axis: Methods (Single-modality, Early Fusion, Late Fusion, Ours)
   - Y-axis: Accuracy (%)
   - **Size**: Half page width (3.5 inches)

3. **`figures/cross_dataset.pdf`** (Cross-dataset section)
   - Heatmap matrix: Train dataset (rows) vs Test dataset (columns)
   - Two side-by-side: Without GRL vs With GRL
   - **Size**: Half page width (3.5 inches)

4. **`figures/attention_viz.pdf`** (Attention visualization)
   - Attention heatmaps showing cross-modal interactions
   - Two examples: Fake sample vs Real sample
   - **Size**: Full page width (7 inches)

5. **`figures/confusion_matrix.pdf`** (Confusion matrix)
   - 2x2 confusion matrix
   - Labels: Real/Fake on both axes
   - **Size**: Half page width (3.5 inches)

6. **`figures/training_curves.pdf`** (Training curves)
   - Two subplots: (a) Loss curves, (b) Accuracy curves
   - Both show train and validation
   - **Size**: Half page width (3.5 inches)

7. **`figures/failure_cases.pdf`** (Failure analysis)
   - Examples of misclassified samples
   - 3-4 images with captions
   - **Size**: Full page width (7 inches)

---

## 🎨 Creating Figures

### Architecture Diagram (draw.io)

**Components to include:**
```
[Input Layer]
  ├─ Visual: Image frames (224x224x3)
  ├─ Audio: Waveform (16kHz)
  ├─ Text: Transcript tokens
  └─ Meta: Categorical features

[Encoder Layer]
  ├─ Visual Encoder: ViT-B/16 → Projection → Tokens (F × 512)
  ├─ Audio Encoder: Wav2Vec2 → Projection → Tokens (K × 512)
  ├─ Text Encoder: SBERT → Projection → Token (1 × 512)
  └─ Meta Encoder: Embeddings+MLP → Token (1 × 512)

[Fusion Layer]
  ├─ Concatenate all tokens + CLS token
  ├─ Add Modality Embeddings
  ├─ 4-layer Transformer (8 heads, d=512)
  └─ Extract CLS token → Fused Vector (z)

[Output Layer]
  ├─ Classifier: z → MLP → Real/Fake
  └─ GRL → Domain Discriminator → Domain ID (0-8)
```

### Python Scripts for Figures

You can use these snippets to generate figures:

```python
# Performance Comparison Bar Chart
import matplotlib.pyplot as plt
import numpy as np

methods = ['Single\nModality', 'Early\nFusion', 'Late\nFusion', 
           'Cross\nAttention', 'Ours\n(+GRL)']
accuracy = [85.8, 90.3, 91.7, 93.8, 95.3]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#8b0000']

plt.figure(figsize=(8, 5))
bars = plt.bar(methods, accuracy, color=colors, alpha=0.8)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.title('Performance Comparison', fontsize=14, fontweight='bold')
plt.ylim([80, 100])
plt.grid(axis='y', alpha=0.3)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}%', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('figures/performance_comparison.pdf', dpi=300, bbox_inches='tight')
```

```python
# Confusion Matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Example data (replace with your actual results)
y_true = [0]*1000 + [1]*1000  # 1000 real, 1000 fake
y_pred = [0]*942 + [1]*58 + [0]*35 + [1]*965  # Your model predictions

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Real', 'Fake'],
            yticklabels=['Real', 'Fake'],
            cbar_kws={'label': 'Count'})
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/confusion_matrix.pdf', dpi=300, bbox_inches='tight')
```

---

## 📝 Equations Summary

The paper includes **25+ mathematical equations** covering:

1. **Visual Encoder** (Equations 1-3)
   - ViT feature extraction
   - Projection to common dimension
   - Token pooling

2. **Audio Encoder** (Equations 4-5)
   - Wav2Vec2 processing
   - Projection

3. **Text & Metadata Encoders** (Equations 6-7)

4. **Cross-Modal Fusion** (Equations 8-17)
   - Token concatenation
   - Modality embeddings
   - Multi-head self-attention
   - Feed-forward networks

5. **Gradient Reversal Layer** (Equations 18-19)
   - Forward pass (identity)
   - Backward pass (gradient reversal)

6. **Domain Discriminator** (Equations 20-22)
   - MLP architecture
   - Domain loss
   - Alpha scheduling

7. **Classification** (Equations 23-25)
   - Classifier MLP
   - Binary cross-entropy loss
   - Total loss (classification + domain)

8. **Training** (Equations 26-28)
   - Learning rate scheduling
   - Gradient scaling (AMP)
   - Gradient accumulation

9. **Evaluation Metrics** (Equations 29-32)
   - Accuracy, Precision, Recall, F1

---

## 🔨 Compilation Instructions

### Using pdflatex:

```bash
cd Multimodal_DeepFake_Detection

# Compile (run twice for references)
pdflatex research_paper.tex
pdflatex research_paper.tex

# Or with bibtex for references
pdflatex research_paper.tex
bibtex research_paper
pdflatex research_paper.tex
pdflatex research_paper.tex
```

### Using Overleaf:

1. Create new project on Overleaf
2. Upload all `.tex` files
3. Set `research_paper.tex` as main file
4. Create `figures/` folder
5. Upload your figure PDFs
6. Click "Recompile"

### Using TeXstudio/TeXmaker:

1. Open `research_paper.tex`
2. Press F5 (or Tools → Build & View)
3. Figures will auto-include when available

---

## ✅ Pre-Submission Checklist

### Content:
- [ ] All equations rendered correctly
- [ ] All references cited properly
- [ ] All tables formatted with data
- [ ] All figures inserted and referenced
- [ ] Author information updated
- [ ] Acknowledgments added
- [ ] Abstract is exactly 150-250 words

### Figures:
- [ ] Architecture diagram (Fig. 1)
- [ ] Performance comparison (Fig. 2)
- [ ] Cross-dataset heatmap (Fig. 3)
- [ ] Attention visualization (Fig. 4)
- [ ] Confusion matrix (Fig. 5)
- [ ] Training curves (Fig. 6)
- [ ] Failure cases (Fig. 7)

### Tables:
- [ ] Hyperparameters (Table 1)
- [ ] Dataset statistics (Table 2)
- [ ] Main results (Table 3)
- [ ] Ablation study (Table 4)
- [ ] Per-dataset performance (Table 5)
- [ ] Cross-dataset generalization (Table 6)
- [ ] Computational efficiency (Table 7)
- [ ] State-of-the-art comparison (Table 8)

### Formatting:
- [ ] IEEE conference format
- [ ] Double-column layout
- [ ] 10pt font
- [ ] Page limit (typically 8-10 pages for IEEE)
- [ ] References formatted consistently

---

## 📤 Submission Files

For conference submission, prepare:

1. **Main paper PDF**: `research_paper.pdf`
2. **Supplementary material**: Additional results, code, dataset details
3. **Source files**: All `.tex` files (if required)
4. **Figures**: High-resolution (300 DPI minimum)
5. **Copyright form**: Signed IEEE copyright transfer

---

## 🎯 Paper Statistics

**Current paper includes:**
- **Sections**: 7 (Intro, Related Work, Methodology, Experiments, Results, Discussion, Conclusion)
- **Equations**: 32 mathematical formulations
- **Tables**: 8 comprehensive tables
- **Figures**: 7 placeholder slots
- **References**: 30+ citations
- **Estimated pages**: 10-12 pages (with figures)

---

## 💡 Tips for Strong Submission

### Writing:
1. **Clear contributions**: First paragraph of intro states exactly what you did
2. **Strong motivation**: Explain WHY your approach matters
3. **Thorough comparison**: Compare with 8+ baseline methods
4. **Honest limitations**: Discuss failure cases and limitations
5. **Future work**: Provide 3-5 concrete future directions

### Figures:
1. **High quality**: Vector graphics (PDF) preferred
2. **Readable**: Large fonts (10pt minimum)
3. **Color-blind safe**: Use patterns in addition to colors
4. **Captions**: Detailed captions explaining everything
5. **Referenced**: Every figure mentioned in text

### Tables:
1. **Formatted consistently**: Use booktabs package
2. **Boldface winners**: Highlight best results
3. **Error bars**: Include standard deviations if multiple runs
4. **Significance**: Mark statistically significant improvements

---

## 🚀 Target Conferences

### Tier 1 (Top):
- **CVPR** (Computer Vision and Pattern Recognition)
  - Deadline: ~November
  - Acceptance: ~25%
  
- **ICCV** (International Conference on Computer Vision)
  - Deadline: ~March
  - Acceptance: ~25%
  
- **NeurIPS** (Neural Information Processing Systems)
  - Deadline: ~May
  - Acceptance: ~20%

### Tier 2 (Excellent):
- **ECCV** (European Conference on Computer Vision)
- **WACV** (Winter Conference on Applications of Computer Vision)
- **MM** (ACM Multimedia)

### Specialized:
- **ICASSP** (Audio/Speech focus)
- **FG** (Face and Gesture Recognition)

---

## 📧 Support

If you need help:
1. Check LaTeX compilation errors carefully
2. Verify all file paths are correct
3. Ensure figures directory exists
4. Test compile individual `.tex` files first

---

**Your paper is ready for figures and final compilation! Good luck with your submission! 🎓📄**
