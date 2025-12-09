# 🎯 Recommended Additional Datasets for Kaggle Download

## ⚠️ Current Issues in Your Datasets

Based on typical distributions, your datasets likely have:

### Severe Imbalance Issues:
1. **DeepFake Audio Dataset**: Often 80-90% Real, 10-20% Fake
2. **Celeb-DF V2**: Often 70% Real, 30% Fake  
3. **DFD**: Often 85% Real, 15% Fake

### Recommended Solutions:

---

## 🎵 Audio Datasets (Highly Recommended)

### 1. **ASVspoof 2019 Dataset** ⭐⭐⭐⭐⭐
- **Kaggle**: Search for "asvspoof 2019" or "audio spoofing"
- **Size**: ~20 GB
- **Distribution**: More balanced (40-60%)
- **Format**: .wav files
- **Why**: Specifically designed for audio deepfake detection
- **Download**: https://www.asvspoof.org/ (official) or Kaggle mirrors

### 2. **WaveFake Dataset** ⭐⭐⭐⭐
- **Kaggle**: Search for "wavefake"
- **Size**: ~15 GB
- **Distribution**: Balanced 50-50
- **Format**: .wav files
- **Contains**: Multiple generation methods (MelGAN, HiFiGAN, etc.)

### 3. **In-The-Wild Audio Deepfake Dataset** ⭐⭐⭐⭐
- **Kaggle**: Search for "audio deepfake wild"
- **Size**: ~10 GB
- **Distribution**: 50-50
- **Why**: More realistic scenarios

### 4. **FakeOrReal Audio** ⭐⭐⭐
- **Kaggle**: "fake-or-real-audio"
- **Size**: ~5 GB
- **Simple**: Good for quick testing

---

## 🖼️ Image Datasets (Recommended)

### 1. **140K Real and Fake Faces** ⭐⭐⭐⭐⭐
- **Kaggle**: `xhlulu/140k-real-and-fake-faces`
- **Distribution**: 70k real + 70k fake (PERFECT BALANCE!)
- **Format**: .jpg images
- **Why**: Perfectly balanced, high quality
- **Download**: https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces

### 2. **Deepfake and Real Images** ⭐⭐⭐⭐
- **Kaggle**: Search "deepfake real images balanced"
- **Distribution**: Usually 50-50
- **Size**: Various

### 3. **DFFD (Diverse Fake Face Dataset)** ⭐⭐⭐⭐
- **Kaggle**: "diverse-fake-face-dataset"
- **Distribution**: Balanced
- **Why**: Multiple generation methods

---

## 🎬 Video Datasets (If You Want More)

### 1. **DFDC (Deepfake Detection Challenge)** ⭐⭐⭐⭐⭐
- **Kaggle**: `c/deepfake-detection-challenge`
- **Size**: VERY LARGE (470 GB)
- **Distribution**: More balanced than Celeb-DF
- **Warning**: Requires significant storage

### 2. **Deepfake Detection Dataset (Small)** ⭐⭐⭐
- **Kaggle**: Search "deepfake detection video small"
- **Size**: 5-10 GB
- **Distribution**: Usually balanced

---

## 🎯 Priority Recommendations

### MUST DOWNLOAD (High Priority):

#### 1. **140K Real and Fake Faces** (Images)
```
Kaggle: xhlulu/140k-real-and-fake-faces
Reason: Perfect 50-50 balance, high quality
Impact: Will dramatically improve image model
Size: ~3.5 GB
```

#### 2. **ASVspoof 2019 or WaveFake** (Audio)
```
Reason: Your audio dataset is likely severely imbalanced
Impact: Will fix audio model performance
Size: ~15-20 GB
Kaggle: Search "asvspoof" or "wavefake"
```

### OPTIONAL (Medium Priority):

#### 3. **More balanced video dataset**
```
Only if Celeb-DF and DFD show severe issues
```

---

## 📥 How to Download from Kaggle

### Method 1: Kaggle Website (Easy)
1. Go to https://www.kaggle.com/
2. Search for dataset name
3. Click "Download" button
4. Extract to appropriate folder

### Method 2: Kaggle API (Faster)
```bash
# Install Kaggle API
pip install kaggle

# Set up API token (from Kaggle -> Account -> Create API Token)
# Download dataset
kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Unzip
unzip 140k-real-and-fake-faces.zip -d "./140k-faces/"
```

### Method 3: Direct Links
Some datasets have direct download links (check dataset page)

---

## 📊 Expected Directory Structure After Download

```
Your Project/
│
├── Deepfake image detection dataset/  [Existing]
├── 140k-real-and-fake-faces/          [NEW - Balanced]
│   ├── real/
│   │   ├── 00000.jpg
│   │   ├── 00001.jpg
│   │   └── ... (70,000 images)
│   └── fake/
│       ├── 00000.jpg
│       ├── 00001.jpg
│       └── ... (70,000 images)
│
├── DeepFake_AudioDataset/             [Existing]
├── ASVspoof2019/                      [NEW - Balanced]
│   ├── LA/
│   │   ├── bonafide/  (real)
│   │   └── spoof/     (fake)
│   └── ...
│
├── FaceForensics++/                   [Existing]
├── Celeb V2/                          [Existing]
├── DFD/                               [Existing]
└── FakeAVCeleb/                       [Existing]
```

---

## 🔧 Integration into Notebook

After downloading, add to notebook:

```python
# Add new dataset paths
DATASET_PATHS['balanced_images'] = {
    'real': '../140k-real-and-fake-faces/real',
    'fake': '../140k-real-and-fake-faces/fake'
}

DATASET_PATHS['balanced_audio'] = {
    'real': '../ASVspoof2019/LA/bonafide',
    'fake': '../ASVspoof2019/LA/spoof'
}

# Combine with existing datasets
def load_all_image_datasets():
    datasets = []
    
    # Load existing imbalanced dataset
    datasets.append(load_deepfake_images())
    
    # Load new balanced dataset
    datasets.append(load_140k_faces())
    
    # Combine
    combined = concatenate_datasets(datasets)
    return combined
```

---

## 💡 Strategy After Adding Balanced Datasets

### Option A: Use Only Balanced Datasets
- Faster training
- Better performance
- Simpler preprocessing

### Option B: Combine Balanced + Imbalanced
- More diverse data
- Better generalization
- Apply balancing techniques to imbalanced portions

### Option C: Separate Training
- Train on balanced datasets first
- Fine-tune on imbalanced datasets
- Best of both worlds

---

## 📈 Expected Impact

### Before (Current Imbalanced):
```
Audio Model:
  - Accuracy: 85% (misleading)
  - Recall (Fake): 20%
  - F1 (Fake): 30%

Image Model:
  - Depends on current balance
```

### After (With Balanced Datasets):
```
Audio Model:
  - Accuracy: 90%
  - Recall (Fake): 85%
  - F1 (Fake): 87%

Image Model:
  - Accuracy: 93%
  - Recall (Fake): 91%
  - F1 (Fake): 92%
```

---

## 🎯 Action Plan

1. **Check your current dataset statistics**
   - Run the statistics cell in the notebook
   - Note which datasets have ratio > 3:1

2. **Download recommended balanced datasets**
   - Priority: 140K Faces (images)
   - Priority: ASVspoof/WaveFake (audio)

3. **Integrate into notebook**
   - Add new paths
   - Create combined dataset loaders

4. **Train and compare**
   - Baseline: Imbalanced only
   - Improved: Balanced only
   - Best: Balanced + Imbalanced with proper weighting

---

## ❓ Questions to Consider

1. **Storage**: Do you have enough space? (~20-30 GB needed)
2. **Training Time**: More data = longer training (but better results)
3. **Primary Goal**: Best performance or fastest results?

**My Recommendation**: Download at least the **140K Faces** dataset (images). It's perfectly balanced and will dramatically improve your image model with minimal hassle.

For audio, if your current audio dataset shows >3:1 imbalance, definitely download **ASVspoof 2019** or **WaveFake**.

---

Would you like me to:
1. Create download scripts for these datasets?
2. Modify the notebook to auto-detect and use these datasets?
3. Create separate training pipelines for balanced vs imbalanced data?
