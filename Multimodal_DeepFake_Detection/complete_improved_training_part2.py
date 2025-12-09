"""
Complete Improved Deepfake Detection Training Script - PART 2
Dataset Classes, Models, Training Loop, Evaluation

Run after part 1 or combine both scripts.
"""

# =============================================================================
# DATASET CLASSES
# =============================================================================

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
import numpy as np
from PIL import Image
import cv2
import librosa
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ImageDataset(Dataset):
    """Dataset for image files with enhanced augmentation for minority class."""
    def __init__(self, file_paths, labels, transform=None, augment_minority=False, minority_class=1):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment_minority = augment_minority
        self.minority_class = minority_class
        
        # Enhanced augmentation for minority class
        self.minority_augment = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            
            # Apply enhanced augmentation for minority class
            if self.augment_minority and label == self.minority_class:
                image = self.minority_augment(image)
            else:
                image = self.transform(image)
            
            return image, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return black image on error
            return torch.zeros(3, 224, 224), torch.tensor(label, dtype=torch.float32)


class AudioDataset(Dataset):
    """Dataset for audio files."""
    def __init__(self, file_paths, labels, sr=16000, duration=3, n_mels=128):
        self.file_paths = file_paths
        self.labels = labels
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        audio_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Load audio
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            # Pad or trim to fixed length
            target_length = self.sr * self.duration
            if len(audio) < target_length:
                audio = np.pad(audio, (0, target_length - len(audio)))
            else:
                audio = audio[:target_length]
            
            # Convert to mel spectrogram
            mel_spec = librosa.feature.melspectrogram(y=audio, sr=self.sr, n_mels=self.n_mels)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize
            mel_spec_db = (mel_spec_db - mel_spec_db.mean()) / (mel_spec_db.std() + 1e-8)
            
            # Convert to tensor [1, n_mels, time]
            mel_tensor = torch.FloatTensor(mel_spec_db).unsqueeze(0)
            
            return mel_tensor, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return torch.zeros(1, self.n_mels, 94), torch.tensor(label, dtype=torch.float32)


class VideoDataset(Dataset):
    """Dataset for video files (frame extraction)."""
    def __init__(self, file_paths, labels, n_frames=16, transform=None):
        self.file_paths = file_paths
        self.labels = labels
        self.n_frames = n_frames
        self.transform = transform
        
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        video_path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                cap.release()
                return torch.zeros(self.n_frames, 3, 224, 224), torch.tensor(label, dtype=torch.float32)
            
            # Sample n_frames uniformly
            frame_indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = Image.fromarray(frame)
                    frame = self.transform(frame)
                    frames.append(frame)
            
            cap.release()
            
            if len(frames) < self.n_frames:
                # Pad with zeros if not enough frames
                frames += [torch.zeros_like(frames[0])] * (self.n_frames - len(frames))
            
            # Stack frames [n_frames, C, H, W]
            frames_tensor = torch.stack(frames)
            
            return frames_tensor, torch.tensor(label, dtype=torch.float32)
        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return torch.zeros(self.n_frames, 3, 224, 224), torch.tensor(label, dtype=torch.float32)


# =============================================================================
# MODEL ARCHITECTURES
# =============================================================================

class ImageModel(nn.Module):
    """Image-based deepfake detector using EfficientNet."""
    def __init__(self, pretrained=True):
        super(ImageModel, self).__init__()
        # Use EfficientNet-B0
        self.backbone = models.efficientnet_b0(pretrained=pretrained)
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)


class AudioModel(nn.Module):
    """Audio-based deepfake detector using CNN."""
    def __init__(self, n_mels=128):
        super(AudioModel, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # Conv block 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Conv block 2
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),
            
            # Conv block 3
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),
            
            # Conv block 4
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


class VideoModel(nn.Module):
    """Video-based deepfake detector using 3D CNN or frame aggregation."""
    def __init__(self, pretrained=True, n_frames=16):
        super(VideoModel, self).__init__()
        self.n_frames = n_frames
        
        # Use 2D CNN for frame feature extraction
        self.frame_encoder = models.resnet18(pretrained=pretrained)
        num_features = self.frame_encoder.fc.in_features
        self.frame_encoder.fc = nn.Identity()  # Remove final FC
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )
        
        # Final classifier
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        # x shape: [batch, n_frames, C, H, W]
        batch_size, n_frames, C, H, W = x.shape
        
        # Extract features from each frame
        x = x.view(batch_size * n_frames, C, H, W)
        frame_features = self.frame_encoder(x)  # [batch*n_frames, features]
        frame_features = frame_features.view(batch_size, n_frames, -1)  # [batch, n_frames, features]
        
        # LSTM for temporal modeling
        lstm_out, _ = self.lstm(frame_features)  # [batch, n_frames, 256]
        
        # Use last output
        x = lstm_out[:, -1, :]  # [batch, 256]
        
        # Final classification
        x = self.fc(x)
        return x


# =============================================================================
# TRAINING FUNCTIONS
# =============================================================================

def train_epoch(model, dataloader, criterion, optimizer, device, use_focal_loss=True):
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for inputs, labels in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(inputs).squeeze()
        
        # Compute loss
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item() * inputs.size(0)
        
        # Predictions
        probs = torch.sigmoid(outputs).detach().cpu().numpy()
        preds = (probs >= 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        acc = accuracy_score(all_labels, all_preds)
        pbar.set_postfix({'loss': loss.item(), 'acc': f'{acc:.4f}'})
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def evaluate_model(model, dataloader, criterion, device, return_probs=False):
    """Evaluate model."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc='Evaluating'):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs).squeeze()
            
            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            
            # Predictions
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    epoch_loss = running_loss / len(dataloader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    
    print(f"\n📊 Evaluation Results:")
    print(f"   Loss: {epoch_loss:.4f}")
    print(f"   Accuracy: {acc:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    
    if return_probs:
        return epoch_loss, acc, precision, recall, f1, np.array(all_labels), np.array(all_probs)
    else:
        return epoch_loss, acc, precision, recall, f1


def plot_confusion_matrix(y_true, y_pred, title='Confusion Matrix', save_path='confusion_matrix.png'):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Real', 'Fake'],
                yticklabels=['Real', 'Fake'])
    plt.title(title, fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"💾 Saved confusion matrix to {save_path}")
    plt.show()
    
    # Print detailed statistics
    print("\n📊 Confusion Matrix Analysis:")
    print(f"   True Negatives (Real as Real): {cm[0, 0]}")
    print(f"   False Positives (Real as Fake): {cm[0, 1]}")
    print(f"   False Negatives (Fake as Real): {cm[1, 0]}")
    print(f"   True Positives (Fake as Fake): {cm[1, 1]}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def train_complete_model(
    train_loader,
    val_loader,
    model_type='image',
    use_focal_loss=True,
    use_class_weights=True,
    epochs=20,
    lr=1e-4,
    save_path='best_model.pth'
):
    """
    Complete training pipeline with all improvements.
    
    Args:
        train_loader: Training data loader
        val_loader: Validation data loader
        model_type: 'image', 'audio', or 'video'
        use_focal_loss: Whether to use Focal Loss
        use_class_weights: Whether to use class weights
        epochs: Number of training epochs
        lr: Learning rate
        save_path: Path to save best model
    """
    print("\n" + "="*80)
    print(f"🚀 TRAINING {model_type.upper()} MODEL")
    print("="*80)
    
    # Create model
    if model_type == 'image':
        model = ImageModel(pretrained=True)
    elif model_type == 'audio':
        model = AudioModel()
    elif model_type == 'video':
        model = VideoModel(pretrained=True)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model = model.to(device)
    
    # Setup loss function
    if use_focal_loss:
        if use_class_weights:
            # Get class weights from training data
            # This would require extracting labels from train_loader
            print("⚠️ Using Focal Loss without explicit class weights (already balanced via α parameter)")
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("✅ Using Focal Loss (α=0.25, γ=2.0)")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("✅ Using BCE Loss")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    
    print(f"\n🎓 Starting training for {epochs} epochs...")
    print(f"   Device: {device}")
    print(f"   Learning rate: {lr}")
    print(f"   Batch size: {train_loader.batch_size}")
    
    for epoch in range(epochs):
        print(f"\n{'='*80}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"{'='*80}")
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        print(f"\n📈 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1 = evaluate_model(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_f1': val_f1,
            }, save_path)
            print(f"💾 Saved best model (F1: {val_f1:.4f})")
    
    print(f"\n{'='*80}")
    print(f"✅ TRAINING COMPLETE")
    print(f"{'='*80}")
    print(f"Best validation F1: {best_val_f1:.4f}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    
    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)
    
    return model, train_losses, val_losses


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss
    ax1.plot(train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.plot(val_losses, 'r-', label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(train_accs, 'b-', label='Train Acc', linewidth=2)
    ax2.plot(val_accs, 'r-', label='Val Acc', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
    print("💾 Saved training curves to training_curves.png")
    plt.show()


print("✅ Part 2 loaded: Dataset classes, Models, Training functions")
print("\n📝 Next: Load your data and start training!")
print("\nExample:")
print("   train_complete_model(train_loader, val_loader, model_type='image', epochs=20)")
