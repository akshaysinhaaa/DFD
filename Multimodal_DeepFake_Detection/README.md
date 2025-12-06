# Multimodal DeepFake Detection - Novel Research Framework

## 📋 Project Overview
This project implements a comprehensive multimodal deepfake detection system using state-of-the-art deep learning architectures. The framework explores multiple approaches from individual modality baselines to sophisticated multimodal fusion techniques.

## 🗂️ Notebook Structure

### Phase 1: Individual Modality Baselines
1. **01_Image_Deepfake_Baseline.ipynb** - Image-based detection using CLIP, DINOv2, ConvNeXt, EfficientNet
2. **02_Audio_Deepfake_Baseline.ipynb** - Audio-based detection using Wav2Vec2, HuBERT, Whisper
3. **03_Video_Deepfake_Baseline.ipynb** - Video-based detection using VideoMAE, TimeSformer, I3D

### Phase 2: Multimodal Fusion Approaches
4. **04_EarlyFusion_Multimodal.ipynb** - Early fusion of feature embeddings
5. **05_LateFusion_Multimodal.ipynb** - Late fusion with decision-level combination
6. **06_CrossModal_Attention.ipynb** - Transformer-based cross-modal attention
7. **07_Contrastive_Multimodal.ipynb** - CLIP-style contrastive learning

### Phase 3: Novel Architectures
8. **08_AudioVisual_LipSync_Detector.ipynb** - Lip-synchronization verification module
9. **09_Temporal_Consistency_Module.ipynb** - Temporal consistency analysis across frames
10. **10_Hierarchical_Classifier.ipynb** - Hierarchical classification (Binary → Multi-class → Localization)

### Phase 4: Comprehensive System
11. **11_Complete_Multimodal_System.ipynb** - Integrated system with all features

### Phase 5: Analysis
12. **12_Model_Comparison_Analysis.ipynb** - Comprehensive model benchmarking and comparison

## 📊 Datasets Used
- **Images**: Deepfake image detection dataset (train/test splits with real/fake labels)
- **Audio**: DeepFake Audio Dataset (KAGGLE - real and fake voice samples)
- **Video**: DFD (DeepFake Detection) dataset with manipulated and original sequences

## 🛠️ Hardware Requirements
- GPU: NVIDIA RTX A6000 (48GB VRAM)
- CUDA-enabled PyTorch

## 📦 Key Dependencies
- PyTorch 2.9.1+
- transformers (HuggingFace)
- timm (PyTorch Image Models)
- librosa (audio processing)
- opencv-python (video processing)
- scikit-learn
- matplotlib, seaborn

## 🚀 Getting Started
Run notebooks sequentially from 01 to 12 to reproduce the complete research pipeline.

## 📄 License
Academic Research Project

## 👤 Author
Research Project for Novel Multimodal DeepFake Detection
