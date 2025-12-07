"""
Novel Multimodal Deepfake Detection Architecture
=================================================

A complete implementation with:
- Cross-modal transformer fusion
- Domain-adversarial training (GRL)
- Multi-encoder architecture (Visual, Audio, Text, Metadata)
- Adaptive memory management for RTX A6000 (48GB VRAM)

Requirements:
------------
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
timm>=0.9.0
open_clip_torch>=2.20.0
sentence-transformers>=2.2.0
opencv-python>=4.8.0
decord>=0.6.0
librosa>=0.10.0
soundfile>=0.12.0
bitsandbytes>=0.41.0  # Optional
accelerate>=0.20.0     # Optional

Usage:
------
# Training on dataset:
python multimodal_deepfake_detector.py --data_root /path/to/data --epochs 10 --batch_size 2

# Demo run (synthetic data):
python multimodal_deepfake_detector.py --demo

# Resume from checkpoint:
python multimodal_deepfake_detector.py --data_root /path/to/data --resume checkpoint.pth

# Use small model config for lower VRAM:
python multimodal_deepfake_detector.py --data_root /path/to/data --model_config small

Author: Research Team
Target GPU: NVIDIA RTX A6000 (48GB VRAM)
"""

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import cv2
import librosa
import soundfile as sf
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import torchvision.transforms as transforms

# Vision models
import timm
try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("Warning: open_clip not available, using timm only")

# Audio models
import torchaudio
from torchaudio.transforms import Resample

# NLP models
from transformers import (
    AutoModel, AutoTokenizer,
    Wav2Vec2Model, Wav2Vec2Processor,
)

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("Warning: sentence-transformers not available")

warnings.filterwarnings('ignore')

# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ModelConfig:
    """Configuration for model architecture and hyperparameters"""
    
    # Model size preset
    preset: str = "large"  # "large" or "small"
    
    # Model dimensions
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 4
    dropout: float = 0.1
    
    # Vision encoder
    vision_backbone: str = "vit_base_patch16_224"  # timm model name
    vision_pretrained: bool = True
    freeze_vision: bool = True
    
    # Audio encoder
    audio_backbone: str = "facebook/wav2vec2-large-960h"
    freeze_audio: bool = True
    
    # Text encoder
    text_backbone: str = "sentence-transformers/all-MiniLM-L6-v2"
    freeze_text: bool = True
    
    # Training
    batch_size: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 10
    gradient_accumulation_steps: int = 4
    alpha_domain: float = 0.5  # Domain adversarial loss weight
    
    # Data processing
    k_frames: int = 5  # Number of frames to sample from video
    k_audio_chunks: int = 5  # Number of 1s audio chunks
    sample_rate: int = 16000
    image_size: int = 224
    max_text_tokens: int = 256
    
    @classmethod
    def from_preset(cls, preset: str, gpu_memory_gb: float):
        """Create config from preset based on available GPU memory"""
        
        if preset == "large" and gpu_memory_gb >= 40:
            return cls(
                preset="large",
                vision_backbone="vit_base_patch16_224",
                audio_backbone="facebook/wav2vec2-large-960h",
                text_backbone="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=2,
            )
        else:
            # Fallback to small config
            print(f"Using SMALL model config (GPU memory: {gpu_memory_gb:.1f}GB)")
            return cls(
                preset="small",
                vision_backbone="resnet50",
                audio_backbone="facebook/wav2vec2-base",
                text_backbone="sentence-transformers/all-MiniLM-L6-v2",
                batch_size=4,
                d_model=256,
                n_heads=4,
                n_layers=2,
            )

# =============================================================================
# Gradient Reversal Layer
# =============================================================================

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient Reversal Layer from:
    'Domain-Adversarial Training of Neural Networks'
    Reverses gradients during backward pass for domain adaptation.
    """
    
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)
    
    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class GradientReversalLayer(nn.Module):
    """Wrapper for gradient reversal"""
    
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)
    
    def set_alpha(self, alpha):
        self.alpha = alpha

print("Configuration and GRL classes defined!")

# =============================================================================
# Encoders
# =============================================================================

class VisualEncoder(nn.Module):
    """
    Visual encoder for images/video frames.
    Extracts per-frame token embeddings using pretrained vision models.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load backbone
        if "vit" in config.vision_backbone.lower():
            self.backbone = timm.create_model(
                config.vision_backbone,
                pretrained=config.vision_pretrained,
                num_classes=0  # Remove classification head
            )
            self.feature_dim = self.backbone.num_features
        elif "resnet" in config.vision_backbone.lower():
            self.backbone = timm.create_model(
                config.vision_backbone,
                pretrained=config.vision_pretrained,
                num_classes=0
            )
            self.feature_dim = self.backbone.num_features
        else:
            raise ValueError(f"Unsupported vision backbone: {config.vision_backbone}")
        
        # Freeze backbone if specified
        if config.freeze_vision:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Projection to common dimension
        self.projection = nn.Linear(self.feature_dim, config.d_model)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (batch, num_frames, C, H, W) or (batch, C, H, W)
        
        Returns:
            tokens: Tensor of shape (batch, num_tokens, d_model)
            available: Boolean indicating if visual data is available
        """
        if images is None or images.numel() == 0:
            return None, False
        
        # Handle single images vs video frames
        if images.ndim == 4:
            # Single image: (batch, C, H, W)
            batch_size = images.size(0)
            num_frames = 1
            images = images.unsqueeze(1)  # (batch, 1, C, H, W)
        else:
            # Video frames: (batch, num_frames, C, H, W)
            batch_size, num_frames = images.size(0), images.size(1)
        
        # Reshape to process all frames
        images_flat = images.view(batch_size * num_frames, *images.shape[2:])
        
        # Extract features
        with torch.set_grad_enabled(not self.config.freeze_vision):
            features = self.backbone(images_flat)  # (batch*num_frames, feature_dim)
        
        # Project to common dimension
        tokens = self.projection(features)  # (batch*num_frames, d_model)
        
        # Reshape back to (batch, num_frames, d_model)
        tokens = tokens.view(batch_size, num_frames, -1)
        
        return tokens, True


class AudioEncoder(nn.Module):
    """
    Audio encoder using Wav2Vec2 or similar pretrained models.
    Extracts audio tokens from waveforms.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load Wav2Vec2 model
        try:
            self.backbone = Wav2Vec2Model.from_pretrained(config.audio_backbone)
            self.processor = Wav2Vec2Processor.from_pretrained(config.audio_backbone)
            self.feature_dim = self.backbone.config.hidden_size
            
            # Freeze backbone if specified
            if config.freeze_audio:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Projection to common dimension
            self.projection = nn.Linear(self.feature_dim, config.d_model)
            self.available = True
            
        except Exception as e:
            print(f"Warning: Could not load audio model: {e}")
            print("Using fallback CNN encoder")
            self.available = False
            self._build_fallback_encoder(config)
    
    def _build_fallback_encoder(self, config):
        """Build simple CNN encoder for audio spectrograms"""
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=10, stride=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(8),
            nn.Conv1d(64, 128, kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(32)
        )
        self.projection = nn.Linear(128 * 32, config.d_model)
        self.feature_dim = 128 * 32
    
    def forward(self, waveforms):
        """
        Args:
            waveforms: Tensor of shape (batch, num_chunks, samples) or (batch, samples)
        
        Returns:
            tokens: Tensor of shape (batch, num_tokens, d_model)
            available: Boolean indicating if audio data is available
        """
        if waveforms is None or waveforms.numel() == 0:
            return None, False
        
        # Handle single waveform vs chunks
        if waveforms.ndim == 2:
            batch_size = waveforms.size(0)
            num_chunks = 1
            waveforms = waveforms.unsqueeze(1)  # (batch, 1, samples)
        else:
            batch_size, num_chunks = waveforms.size(0), waveforms.size(1)
        
        # Reshape to process all chunks
        waveforms_flat = waveforms.view(batch_size * num_chunks, -1)
        
        # Extract features
        if self.available:
            with torch.set_grad_enabled(not self.config.freeze_audio):
                outputs = self.backbone(waveforms_flat)
                features = outputs.last_hidden_state.mean(dim=1)  # Pool over time
        else:
            # Fallback CNN
            waveforms_flat = waveforms_flat.unsqueeze(1)  # Add channel dim
            features = self.backbone(waveforms_flat)
            features = features.view(batch_size * num_chunks, -1)
        
        # Project to common dimension
        tokens = self.projection(features)  # (batch*num_chunks, d_model)
        
        # Reshape back
        tokens = tokens.view(batch_size, num_chunks, -1)
        
        return tokens, True


class TextEncoder(nn.Module):
    """
    Text encoder for transcripts using sentence transformers or similar.
    Extracts text embeddings from transcripts.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load text model
        try:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                self.backbone = SentenceTransformer(config.text_backbone)
                self.feature_dim = self.backbone.get_sentence_embedding_dimension()
            else:
                # Fallback to distilbert
                self.backbone = AutoModel.from_pretrained('distilbert-base-uncased')
                self.tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
                self.feature_dim = 768
            
            # Freeze backbone if specified
            if config.freeze_text:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            # Projection to common dimension
            self.projection = nn.Linear(self.feature_dim, config.d_model)
            self.available = True
            
        except Exception as e:
            print(f"Warning: Could not load text model: {e}")
            self.available = False
            self.feature_dim = config.d_model
            self.projection = nn.Identity()
    
    def forward(self, texts):
        """
        Args:
            texts: List of strings or None
        
        Returns:
            tokens: Tensor of shape (batch, 1, d_model) - pooled text embedding
            available: Boolean indicating if text data is available
        """
        if texts is None or len(texts) == 0:
            return None, False
        
        batch_size = len(texts)
        
        # Extract features
        if self.available:
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                with torch.set_grad_enabled(not self.config.freeze_text):
                    embeddings = self.backbone.encode(
                        texts, 
                        convert_to_tensor=True,
                        show_progress_bar=False
                    )
            else:
                # Fallback: use tokenizer + model
                inputs = self.tokenizer(
                    texts, 
                    return_tensors='pt', 
                    padding=True, 
                    truncation=True,
                    max_length=self.config.max_text_tokens
                ).to(next(self.backbone.parameters()).device)
                
                with torch.set_grad_enabled(not self.config.freeze_text):
                    outputs = self.backbone(**inputs)
                    embeddings = outputs.last_hidden_state[:, 0, :]  # CLS token
        else:
            # Return zeros if not available
            device = next(self.projection.parameters()).device
            embeddings = torch.zeros(batch_size, self.feature_dim, device=device)
        
        # Project to common dimension
        tokens = self.projection(embeddings)  # (batch, d_model)
        
        # Add sequence dimension
        tokens = tokens.unsqueeze(1)  # (batch, 1, d_model)
        
        return tokens, True


class MetadataEncoder(nn.Module):
    """
    Metadata encoder for categorical features.
    Encodes metadata like uploader, platform, date, etc.
    """
    
    def __init__(self, config: ModelConfig, 
                 n_uploaders=100, n_platforms=10, n_date_buckets=12, n_likes_buckets=10):
        super().__init__()
        self.config = config
        
        # Categorical embeddings
        self.uploader_emb = nn.Embedding(n_uploaders, 64)
        self.platform_emb = nn.Embedding(n_platforms, 32)
        self.date_emb = nn.Embedding(n_date_buckets, 32)
        self.likes_emb = nn.Embedding(n_likes_buckets, 32)
        
        # MLP to project to common dimension
        total_dim = 64 + 32 + 32 + 32
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
    
    def forward(self, metadata):
        """
        Args:
            metadata: Dict with keys 'uploader', 'platform', 'date', 'likes' (LongTensor)
        
        Returns:
            tokens: Tensor of shape (batch, 1, d_model)
            available: Boolean indicating if metadata is available
        """
        if metadata is None or len(metadata) == 0:
            return None, False
        
        # Get embeddings for each field
        embs = []
        if 'uploader' in metadata:
            embs.append(self.uploader_emb(metadata['uploader']))
        if 'platform' in metadata:
            embs.append(self.platform_emb(metadata['platform']))
        if 'date' in metadata:
            embs.append(self.date_emb(metadata['date']))
        if 'likes' in metadata:
            embs.append(self.likes_emb(metadata['likes']))
        
        if len(embs) == 0:
            return None, False
        
        # Concatenate and project
        combined = torch.cat(embs, dim=-1)
        tokens = self.mlp(combined)
        
        # Add sequence dimension
        tokens = tokens.unsqueeze(1)  # (batch, 1, d_model)
        
        return tokens, True


print("Encoder classes defined!")

# =============================================================================
# Cross-Modal Fusion Transformer
# =============================================================================

class CrossModalFusionTransformer(nn.Module):
    """
    Cross-modal fusion using Transformer encoder.
    Fuses tokens from all modalities using self-attention.
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Modality embeddings (learned)
        self.modality_embeddings = nn.Embedding(4, config.d_model)  # 4 modalities
        
        # CLS token for pooling
        self.cls_token = nn.Parameter(torch.randn(1, 1, config.d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.n_layers,
            norm=nn.LayerNorm(config.d_model)
        )
        
        # Modality IDs
        self.VISUAL_ID = 0
        self.AUDIO_ID = 1
        self.TEXT_ID = 2
        self.META_ID = 3
    
    def forward(self, visual_tokens=None, audio_tokens=None, 
                text_tokens=None, meta_tokens=None, attention_mask=None):
        """
        Args:
            visual_tokens: (batch, n_visual, d_model) or None
            audio_tokens: (batch, n_audio, d_model) or None
            text_tokens: (batch, n_text, d_model) or None
            meta_tokens: (batch, n_meta, d_model) or None
            attention_mask: (batch, total_tokens) - True for valid tokens
        
        Returns:
            fused_vector: (batch, d_model) - pooled representation
            all_tokens: (batch, total_tokens, d_model) - all output tokens
        """
        batch_size = (visual_tokens.size(0) if visual_tokens is not None 
                     else audio_tokens.size(0) if audio_tokens is not None
                     else text_tokens.size(0) if text_tokens is not None
                     else meta_tokens.size(0))
        
        device = (visual_tokens.device if visual_tokens is not None
                 else audio_tokens.device if audio_tokens is not None
                 else text_tokens.device if text_tokens is not None
                 else meta_tokens.device)
        
        # Collect all tokens
        all_tokens = []
        modality_ids = []
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        all_tokens.append(cls_tokens)
        # CLS doesn't need modality embedding
        
        # Add visual tokens
        if visual_tokens is not None:
            n_visual = visual_tokens.size(1)
            visual_mod_emb = self.modality_embeddings(
                torch.full((batch_size, n_visual), self.VISUAL_ID, 
                          dtype=torch.long, device=device)
            )
            visual_tokens = visual_tokens + visual_mod_emb
            all_tokens.append(visual_tokens)
        
        # Add audio tokens
        if audio_tokens is not None:
            n_audio = audio_tokens.size(1)
            audio_mod_emb = self.modality_embeddings(
                torch.full((batch_size, n_audio), self.AUDIO_ID,
                          dtype=torch.long, device=device)
            )
            audio_tokens = audio_tokens + audio_mod_emb
            all_tokens.append(audio_tokens)
        
        # Add text tokens
        if text_tokens is not None:
            n_text = text_tokens.size(1)
            text_mod_emb = self.modality_embeddings(
                torch.full((batch_size, n_text), self.TEXT_ID,
                          dtype=torch.long, device=device)
            )
            text_tokens = text_tokens + text_mod_emb
            all_tokens.append(text_tokens)
        
        # Add metadata tokens
        if meta_tokens is not None:
            n_meta = meta_tokens.size(1)
            meta_mod_emb = self.modality_embeddings(
                torch.full((batch_size, n_meta), self.META_ID,
                          dtype=torch.long, device=device)
            )
            meta_tokens = meta_tokens + meta_mod_emb
            all_tokens.append(meta_tokens)
        
        # Concatenate all tokens
        if len(all_tokens) == 0:
            raise ValueError("At least one modality must be provided")
        
        combined_tokens = torch.cat(all_tokens, dim=1)  # (batch, total_tokens, d_model)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, combined_tokens.size(1),
                dtype=torch.bool, device=device
            )
        
        # Convert mask for transformer (True = mask out)
        src_key_padding_mask = ~attention_mask
        
        # Apply transformer
        output_tokens = self.transformer(
            combined_tokens,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # Extract CLS token as fused representation
        fused_vector = output_tokens[:, 0, :]  # (batch, d_model)
        
        return fused_vector, output_tokens


# =============================================================================
# Domain Discriminator
# =============================================================================

class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for adversarial training.
    Classifies the source domain of the input.
    """
    
    def __init__(self, d_model, n_domains, dropout=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, n_domains)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, d_model) - features from encoder
        
        Returns:
            logits: (batch, n_domains) - domain classification logits
        """
        return self.network(x)


# =============================================================================
# Classifier
# =============================================================================

class ClassifierMLP(nn.Module):
    """
    Binary classifier for fake/real detection.
    """
    
    def __init__(self, d_model, dropout=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Binary classification (no sigmoid, use BCEWithLogitsLoss)
        )
    
    def forward(self, x):
        """
        Args:
            x: (batch, d_model) - fused features
        
        Returns:
            logits: (batch, 1) - raw logits for fake/real
        """
        return self.network(x)


print("Fusion transformer, discriminator, and classifier defined!")

# =============================================================================
# Complete Multimodal Model
# =============================================================================

class MultimodalDeepfakeDetector(nn.Module):
    """
    Complete multimodal deepfake detection model with domain-adversarial training.
    """
    
    def __init__(self, config: ModelConfig, n_domains=5):
        super().__init__()
        self.config = config
        
        # Encoders
        self.visual_encoder = VisualEncoder(config)
        self.audio_encoder = AudioEncoder(config)
        self.text_encoder = TextEncoder(config)
        self.meta_encoder = MetadataEncoder(config)
        
        # Fusion
        self.fusion = CrossModalFusionTransformer(config)
        
        # Gradient Reversal Layer
        self.grl = GradientReversalLayer(alpha=config.alpha_domain)
        
        # Domain discriminator
        self.domain_discriminator = DomainDiscriminator(
            config.d_model, n_domains, config.dropout
        )
        
        # Classifier
        self.classifier = ClassifierMLP(config.d_model, config.dropout)
    
    def forward(self, images=None, audio=None, text=None, metadata=None,
                return_domain_logits=True):
        """
        Forward pass through the model.
        
        Args:
            images: (batch, num_frames, C, H, W) or None
            audio: (batch, num_chunks, samples) or None
            text: List of strings or None
            metadata: Dict of categorical features or None
            return_domain_logits: Whether to compute domain logits
        
        Returns:
            dict with keys:
                - 'logits': (batch, 1) - fake/real classification logits
                - 'domain_logits': (batch, n_domains) - domain classification logits
                - 'fused_vector': (batch, d_model) - fused representation
        """
        # Encode each modality
        visual_tokens, visual_avail = self.visual_encoder(images) if images is not None else (None, False)
        audio_tokens, audio_avail = self.audio_encoder(audio) if audio is not None else (None, False)
        text_tokens, text_avail = self.text_encoder(text) if text is not None else (None, False)
        meta_tokens, meta_avail = self.meta_encoder(metadata) if metadata is not None else (None, False)
        
        # Fuse modalities
        fused_vector, all_tokens = self.fusion(
            visual_tokens=visual_tokens if visual_avail else None,
            audio_tokens=audio_tokens if audio_avail else None,
            text_tokens=text_tokens if text_avail else None,
            meta_tokens=meta_tokens if meta_avail else None
        )
        
        # Classification
        class_logits = self.classifier(fused_vector)
        
        # Domain classification with GRL
        domain_logits = None
        if return_domain_logits:
            reversed_features = self.grl(fused_vector)
            domain_logits = self.domain_discriminator(reversed_features)
        
        return {
            'logits': class_logits,
            'domain_logits': domain_logits,
            'fused_vector': fused_vector
        }
    
    def set_grl_alpha(self, alpha):
        """Update GRL alpha for domain adaptation scheduling"""
        self.grl.set_alpha(alpha)


# =============================================================================
# Dataset Classes
# =============================================================================

class GenericMultimodalDataset(Dataset):
    """
    Generic dataset that handles multiple data sources and modalities.
    Automatically detects dataset types and creates unified interface.
    """
    
    def __init__(self, data_root, config: ModelConfig, split='train'):
        self.data_root = Path(data_root)
        self.config = config
        self.split = split
        self.samples = []
        
        # Scan for datasets
        self._scan_datasets()
        
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _scan_datasets(self):
        """Scan data_root for known dataset types"""
        
        # Look for image datasets
        for img_dir in self.data_root.glob("**/train" if self.split == "train" else "**/test"):
            if any(x in str(img_dir).lower() for x in ['image', 'fake', 'real', 'deepfake']):
                self._add_image_dataset(img_dir)
        
        # Look for audio datasets
        for audio_dir in self.data_root.glob("**/AUDIO"):
            if 'REAL' in str(audio_dir) or 'FAKE' in str(audio_dir):
                self._add_audio_dataset(audio_dir.parent)
        
        # Look for video datasets (DFD, DFDC, etc.)
        for video_dir in self.data_root.glob("**/dfd_faces"):
            self._add_video_dataset(video_dir)
    
    def _add_image_dataset(self, img_dir):
        """Add image samples from directory"""
        fake_dir = img_dir / 'fake'
        real_dir = img_dir / 'real'
        
        # Add fake images
        if fake_dir.exists():
            for img_file in fake_dir.glob('*.jpg') or fake_dir.glob('*.png'):
                self.samples.append({
                    'type': 'image',
                    'path': str(img_file),
                    'label': 1,  # fake
                    'domain': self._get_domain_id(str(img_file)),
                    'modalities': {'image': True, 'audio': False, 'text': False, 'meta': False}
                })
        
        # Add real images
        if real_dir.exists():
            for img_file in real_dir.glob('*.jpg') or real_dir.glob('*.png'):
                self.samples.append({
                    'type': 'image',
                    'path': str(img_file),
                    'label': 0,  # real
                    'domain': self._get_domain_id(str(img_file)),
                    'modalities': {'image': True, 'audio': False, 'text': False, 'meta': False}
                })
    
    def _add_audio_dataset(self, audio_root):
        """Add audio samples"""
        fake_dir = audio_root / 'FAKE'
        real_dir = audio_root / 'REAL'
        
        if fake_dir.exists():
            for audio_file in fake_dir.glob('*.wav'):
                self.samples.append({
                    'type': 'audio',
                    'path': str(audio_file),
                    'label': 1,
                    'domain': self._get_domain_id(str(audio_file)),
                    'modalities': {'image': False, 'audio': True, 'text': False, 'meta': False}
                })
        
        if real_dir.exists():
            for audio_file in real_dir.glob('*.wav'):
                self.samples.append({
                    'type': 'audio',
                    'path': str(audio_file),
                    'label': 0,
                    'domain': self._get_domain_id(str(audio_file)),
                    'modalities': {'image': False, 'audio': True, 'text': False, 'meta': False}
                })
    
    def _add_video_dataset(self, video_root):
        """Add video samples (using extracted frames)"""
        for split_dir in (video_root / self.split).iterdir():
            if split_dir.is_dir():
                label = 1 if 'fake' in split_dir.name.lower() else 0
                for frame_file in split_dir.glob('*.jpg'):
                    self.samples.append({
                        'type': 'video',
                        'path': str(frame_file),
                        'label': label,
                        'domain': self._get_domain_id(str(frame_file)),
                        'modalities': {'image': True, 'audio': False, 'text': False, 'meta': False}
                    })
    
    def _get_domain_id(self, path):
        """Extract domain ID from path"""
        path_lower = path.lower()
        if 'dfd' in path_lower or 'deepfake' in path_lower:
            return 0
        elif 'kaggle' in path_lower or 'image' in path_lower:
            return 1
        elif 'audio' in path_lower:
            return 2
        elif 'dfdc' in path_lower:
            return 3
        else:
            return 4  # unknown
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load data based on type
        if sample['type'] == 'image':
            image = self._load_image(sample['path'])
            return {
                'image': image,
                'audio': None,
                'text': None,
                'metadata': None,
                'label': sample['label'],
                'domain': sample['domain']
            }
        elif sample['type'] == 'audio':
            audio = self._load_audio(sample['path'])
            return {
                'image': None,
                'audio': audio,
                'text': None,
                'metadata': None,
                'label': sample['label'],
                'domain': sample['domain']
            }
        elif sample['type'] == 'video':
            # Load as image for now (frames already extracted)
            image = self._load_image(sample['path'])
            return {
                'image': image,
                'audio': None,
                'text': None,
                'metadata': None,
                'label': sample['label'],
                'domain': sample['domain']
            }
    
    def _load_image(self, path):
        """Load and preprocess image"""
        try:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (self.config.image_size, self.config.image_size))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            # Normalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std
            return img
        except Exception as e:
            print(f"Error loading image {path}: {e}")
            return torch.zeros(3, self.config.image_size, self.config.image_size)
    
    def _load_audio(self, path):
        """Load and preprocess audio"""
        try:
            waveform, sr = librosa.load(path, sr=self.config.sample_rate, duration=10)
            # Pad or truncate to fixed length
            target_length = self.config.sample_rate * 10
            if len(waveform) < target_length:
                waveform = np.pad(waveform, (0, target_length - len(waveform)))
            else:
                waveform = waveform[:target_length]
            return torch.from_numpy(waveform).float()
        except Exception as e:
            print(f"Error loading audio {path}: {e}")
            return torch.zeros(self.config.sample_rate * 10)


def collate_fn(batch):
    """Custom collate function for variable-length modalities"""
    
    images, audios, texts, metadatas = [], [], [], []
    labels, domains = [], []
    
    for item in batch:
        if item['image'] is not None:
            images.append(item['image'])
        if item['audio'] is not None:
            audios.append(item['audio'])
        if item['text'] is not None:
            texts.append(item['text'])
        if item['metadata'] is not None:
            metadatas.append(item['metadata'])
        
        labels.append(item['label'])
        domains.append(item['domain'])
    
    # Stack tensors
    batch_images = torch.stack(images) if len(images) > 0 else None
    batch_audios = torch.stack(audios) if len(audios) > 0 else None
    batch_texts = texts if len(texts) > 0 else None
    batch_metadata = metadatas if len(metadatas) > 0 else None
    
    batch_labels = torch.tensor(labels, dtype=torch.float32)
    batch_domains = torch.tensor(domains, dtype=torch.long)
    
    return {
        'images': batch_images,
        'audio': batch_audios,
        'text': batch_texts,
        'metadata': batch_metadata,
        'labels': batch_labels,
        'domains': batch_domains
    }


print("Model and dataset classes defined!")

# =============================================================================
# Training & Evaluation
# =============================================================================

def train_epoch(model, dataloader, optimizer, scaler, config, epoch, device):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_cls_loss = 0
    total_domain_loss = 0
    correct = 0
    total = 0
    
    # Update GRL alpha (gradually increase from 0 to alpha_domain)
    progress = epoch / config.epochs
    alpha = config.alpha_domain * (2 / (1 + np.exp(-10 * progress)) - 1)
    model.set_grl_alpha(alpha)
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{config.epochs}')
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        images = batch['images'].to(device) if batch['images'] is not None else None
        audio = batch['audio'].to(device) if batch['audio'] is not None else None
        text = batch['text']  # Keep as list of strings
        metadata = batch['metadata']
        labels = batch['labels'].to(device)
        domains = batch['domains'].to(device)
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(
                images=images,
                audio=audio,
                text=text,
                metadata=metadata,
                return_domain_logits=True
            )
            
            # Classification loss
            cls_loss = F.binary_cross_entropy_with_logits(
                outputs['logits'].squeeze(-1),
                labels
            )
            
            # Domain loss
            if outputs['domain_logits'] is not None:
                domain_loss = F.cross_entropy(
                    outputs['domain_logits'],
                    domains
                )
            else:
                domain_loss = torch.tensor(0.0, device=device)
            
            # Total loss
            loss = cls_loss + alpha * domain_loss
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        # Statistics
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_domain_loss += domain_loss.item()
        
        preds = (torch.sigmoid(outputs['logits']) > 0.5).float()
        correct += (preds.squeeze(-1) == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': total_loss / (batch_idx + 1),
            'cls_loss': total_cls_loss / (batch_idx + 1),
            'dom_loss': total_domain_loss / (batch_idx + 1),
            'acc': 100. * correct / total,
            'alpha': alpha
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'cls_loss': total_cls_loss / len(dataloader),
        'domain_loss': total_domain_loss / len(dataloader),
        'accuracy': 100. * correct / total
    }


def evaluate(model, dataloader, config, device):
    """Evaluate the model"""
    model.eval()
    
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            # Move to device
            images = batch['images'].to(device) if batch['images'] is not None else None
            audio = batch['audio'].to(device) if batch['audio'] is not None else None
            text = batch['text']
            metadata = batch['metadata']
            labels = batch['labels'].to(device)
            
            # Forward pass
            with autocast():
                outputs = model(
                    images=images,
                    audio=audio,
                    text=text,
                    metadata=metadata,
                    return_domain_logits=False
                )
                
                loss = F.binary_cross_entropy_with_logits(
                    outputs['logits'].squeeze(-1),
                    labels
                )
            
            total_loss += loss.item()
            
            preds = (torch.sigmoid(outputs['logits']) > 0.5).float()
            correct += (preds.squeeze(-1) == labels).sum().item()
            total += labels.size(0)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = 100. * correct / total
    precision = precision_score(all_labels, all_preds, zero_division=0) * 100
    recall = recall_score(all_labels, all_preds, zero_division=0) * 100
    f1 = f1_score(all_labels, all_preds, zero_division=0) * 100
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }


def save_checkpoint(model, optimizer, scaler, epoch, config, path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'config': config
    }
    torch.save(checkpoint, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(model, optimizer, scaler, path, device):
    """Load model checkpoint"""
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    epoch = checkpoint['epoch']
    print(f"Checkpoint loaded from {path} (epoch {epoch})")
    return epoch


def train(config, data_root, resume=None, device='cuda'):
    """Main training function"""
    
    # Create datasets
    print("Loading datasets...")
    train_dataset = GenericMultimodalDataset(data_root, config, split='train')
    val_dataset = GenericMultimodalDataset(data_root, config, split='test')
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("Building model...")
    model = MultimodalDeepfakeDetector(config, n_domains=5).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs
    )
    
    # Mixed precision scaler
    scaler = GradScaler()
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if resume is not None:
        start_epoch = load_checkpoint(model, optimizer, scaler, resume, device)
    
    # Training loop
    print(f"\nStarting training for {config.epochs} epochs...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    best_val_acc = 0.0
    
    for epoch in range(start_epoch, config.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, scaler, config, epoch, device
        )
        
        # Validate
        val_metrics = evaluate(model, val_loader, config, device)
        
        # Update scheduler
        scheduler.step()
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{config.epochs}")
        print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%, "
              f"P: {val_metrics['precision']:.2f}%, R: {val_metrics['recall']:.2f}%, "
              f"F1: {val_metrics['f1']:.2f}%")
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, scaler, epoch, config,
                'best_multimodal_model.pth'
            )
            print(f"New best model! Accuracy: {best_val_acc:.2f}%")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_checkpoint(
                model, optimizer, scaler, epoch, config,
                f'checkpoint_epoch_{epoch+1}.pth'
            )
    
    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")


print("Training and evaluation functions defined!")

# =============================================================================
# Demo Function
# =============================================================================

def run_demo():
    """
    Run a demo with synthetic data to verify the model works.
    Creates toy samples and runs one training step.
    """
    print("\n" + "="*70)
    print("RUNNING DEMO WITH SYNTHETIC DATA")
    print("="*70 + "\n")
    
    # Detect GPU memory and select config
    if torch.cuda.is_available():
        gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Detected GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {gpu_memory_gb:.2f} GB")
        device = torch.device('cuda')
    else:
        gpu_memory_gb = 0
        print("No GPU detected, using CPU")
        device = torch.device('cpu')
    
    # Create config
    config = ModelConfig.from_preset("large" if gpu_memory_gb >= 40 else "small", gpu_memory_gb)
    print(f"\nUsing {config.preset.upper()} model configuration")
    print(f"Model dimension: {config.d_model}")
    print(f"Transformer layers: {config.n_layers}")
    print(f"Attention heads: {config.n_heads}")
    
    # Create toy dataset
    print("\nCreating synthetic samples...")
    
    class ToyDataset(Dataset):
        def __init__(self, config, n_samples=4):
            self.config = config
            self.n_samples = n_samples
        
        def __len__(self):
            return self.n_samples
        
        def __getitem__(self, idx):
            # Create synthetic data
            if idx % 2 == 0:
                # Fake sample with image and audio
                image = torch.randn(3, config.image_size, config.image_size)
                audio = torch.randn(config.sample_rate * 10)
                label = 1.0  # fake
            else:
                # Real sample with only image
                image = torch.randn(3, config.image_size, config.image_size)
                audio = None
                label = 0.0  # real
            
            return {
                'image': image,
                'audio': audio,
                'text': None,
                'metadata': None,
                'label': label,
                'domain': idx % 3  # Rotate through 3 domains
            }
    
    # Create dataloader
    toy_dataset = ToyDataset(config, n_samples=4)
    toy_loader = DataLoader(
        toy_dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    print(f"Created {len(toy_dataset)} synthetic samples")
    
    # Create model
    print("\nBuilding model...")
    try:
        model = MultimodalDeepfakeDetector(config, n_domains=3).to(device)
        print(f"Model created successfully!")
        
        # Count parameters
        n_params = sum(p.numel() for p in model.parameters())
        n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total parameters: {n_params:,}")
        print(f"Trainable parameters: {n_trainable:,}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nGPU Out of Memory! Switching to SMALL config...")
            torch.cuda.empty_cache()
            config = ModelConfig.from_preset("small", 0)
            model = MultimodalDeepfakeDetector(config, n_domains=3).to(device)
            print("Model created with SMALL config")
        else:
            raise e
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    scaler = GradScaler()
    
    # Run one training step
    print("\nRunning one training step...")
    model.train()
    
    for batch in toy_loader:
        images = batch['images'].to(device) if batch['images'] is not None else None
        audio = batch['audio'].to(device) if batch['audio'] is not None else None
        labels = batch['labels'].to(device)
        domains = batch['domains'].to(device)
        
        print(f"\nBatch shapes:")
        if images is not None:
            print(f"  Images: {images.shape}")
        if audio is not None:
            print(f"  Audio: {audio.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Domains: {domains.shape}")
        
        # Forward pass
        with autocast():
            outputs = model(
                images=images,
                audio=audio,
                text=None,
                metadata=None,
                return_domain_logits=True
            )
            
            cls_loss = F.binary_cross_entropy_with_logits(
                outputs['logits'].squeeze(-1),
                labels
            )
            
            domain_loss = F.cross_entropy(
                outputs['domain_logits'],
                domains
            )
            
            loss = cls_loss + config.alpha_domain * domain_loss
        
        print(f"\nLosses:")
        print(f"  Classification loss: {cls_loss.item():.4f}")
        print(f"  Domain loss: {domain_loss.item():.4f}")
        print(f"  Total loss: {loss.item():.4f}")
        
        # Backward pass
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        print("\nBackward pass completed successfully!")
        
        # Only run one batch for demo
        break
    
    # Save checkpoint
    checkpoint_path = 'demo_checkpoint.pth'
    save_checkpoint(model, optimizer, scaler, 0, config, checkpoint_path)
    
    print("\n" + "="*70)
    print("DEMO COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nCheckpoint saved to: {checkpoint_path}")
    print(f"Model is ready for training on real data.")
    print(f"\nTo train on your data:")
    print(f"  python multimodal_deepfake_detector.py --data_root /path/to/data --epochs 10")
    print("\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    """Main entry point with argument parsing"""
    parser = argparse.ArgumentParser(
        description='Novel Multimodal Deepfake Detection with Domain-Adversarial Training'
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default=None,
                       help='Root directory containing datasets')
    parser.add_argument('--demo', action='store_true',
                       help='Run demo with synthetic data')
    
    # Model arguments
    parser.add_argument('--model_config', type=str, default='auto',
                       choices=['auto', 'large', 'small'],
                       help='Model size configuration (auto selects based on GPU memory)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--alpha_domain', type=float, default=0.5,
                       help='Weight for domain adversarial loss')
    parser.add_argument('--k_frames', type=int, default=5,
                       help='Number of frames to sample from video')
    parser.add_argument('--k_audio_chunks', type=int, default=5,
                       help='Number of audio chunks to sample')
    
    # Checkpoint arguments
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume training')
    
    # Device arguments
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    # Run demo if specified
    if args.demo:
        run_demo()
        return
    
    # Check data_root is provided
    if args.data_root is None:
        print("Error: --data_root must be provided (or use --demo for synthetic data)")
        print("\nUsage:")
        print("  python multimodal_deepfake_detector.py --data_root /path/to/data --epochs 10")
        print("  python multimodal_deepfake_detector.py --demo")
        return
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
        gpu_memory_gb = 0
    else:
        device = torch.device(args.device)
        if torch.cuda.is_available():
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        else:
            gpu_memory_gb = 0
    
    # Create config
    if args.model_config == 'auto':
        preset = 'large' if gpu_memory_gb >= 40 else 'small'
    else:
        preset = args.model_config
    
    config = ModelConfig.from_preset(preset, gpu_memory_gb)
    
    # Override config with command-line arguments
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    config.epochs = args.epochs
    config.learning_rate = args.lr
    config.alpha_domain = args.alpha_domain
    config.k_frames = args.k_frames
    config.k_audio_chunks = args.k_audio_chunks
    
    # Print configuration
    print("\n" + "="*70)
    print("MULTIMODAL DEEPFAKE DETECTION - TRAINING")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model preset: {config.preset.upper()}")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Transformer layers: {config.n_layers}")
    print(f"  Attention heads: {config.n_heads}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Alpha domain: {config.alpha_domain}")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {gpu_memory_gb:.2f} GB")
    print(f"  Data root: {args.data_root}")
    print("="*70 + "\n")
    
    # Start training
    try:
        train(config, args.data_root, resume=args.resume, device=device)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
    except Exception as e:
        print(f"\n\nError during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
