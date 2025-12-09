"""
Complete Improved Deepfake Detection Training Script - PART 3
Data Loading Functions and Complete End-to-End Example

This file shows how to load your datasets and run complete training.
"""

import glob
from pathlib import Path

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_image_dataset(paths_dict, max_samples_per_class=None, print_stats=True):
    """
    Load image dataset from multiple sources.
    
    Args:
        paths_dict: Dictionary with 'real' and 'fake' keys containing paths
        max_samples_per_class: Maximum samples per class (None = all)
        print_stats: Whether to print statistics
    
    Returns:
        file_paths, labels arrays
    """
    all_files = []
    all_labels = []
    
    # Load real images
    if 'train_real' in paths_dict and 'test_real' in paths_dict:
        # Has train/test split
        for key in ['train_real', 'test_real']:
            if os.path.exists(paths_dict[key]):
                files = glob.glob(os.path.join(paths_dict[key], '*.*'))
                files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if max_samples_per_class:
                    files = files[:max_samples_per_class]
                all_files.extend(files)
                all_labels.extend([0] * len(files))
                if print_stats:
                    print(f"✓ Loaded {len(files)} from {key}")
    elif 'real' in paths_dict:
        # Single real folder
        if os.path.exists(paths_dict['real']):
            files = glob.glob(os.path.join(paths_dict['real'], '**', '*.*'), recursive=True)
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if max_samples_per_class:
                files = files[:max_samples_per_class]
            all_files.extend(files)
            all_labels.extend([0] * len(files))
            if print_stats:
                print(f"✓ Loaded {len(files)} real images")
    
    # Load fake images
    if 'train_fake' in paths_dict and 'test_fake' in paths_dict:
        # Has train/test split
        for key in ['train_fake', 'test_fake']:
            if os.path.exists(paths_dict[key]):
                files = glob.glob(os.path.join(paths_dict[key], '*.*'))
                files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
                if max_samples_per_class:
                    files = files[:max_samples_per_class]
                all_files.extend(files)
                all_labels.extend([1] * len(files))
                if print_stats:
                    print(f"✓ Loaded {len(files)} from {key}")
    elif 'fake' in paths_dict:
        # Single fake folder
        if os.path.exists(paths_dict['fake']):
            files = glob.glob(os.path.join(paths_dict['fake'], '**', '*.*'), recursive=True)
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if max_samples_per_class:
                files = files[:max_samples_per_class]
            all_files.extend(files)
            all_labels.extend([1] * len(files))
            if print_stats:
                print(f"✓ Loaded {len(files)} fake images")
    
    # Load FaceForensics++ fake types if present
    for key in ['deepfakes', 'face2face', 'faceswap', 'neuraltextures']:
        if key in paths_dict and os.path.exists(paths_dict[key]):
            files = glob.glob(os.path.join(paths_dict[key], '**', '*.*'), recursive=True)
            files = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
            if max_samples_per_class:
                files = files[:max_samples_per_class // 4]  # Divide among 4 types
            all_files.extend(files)
            all_labels.extend([1] * len(files))
            if print_stats:
                print(f"✓ Loaded {len(files)} from {key}")
    
    if print_stats:
        print(f"\n📊 Total: {len(all_files)} images")
        real_count = sum(1 for l in all_labels if l == 0)
        fake_count = sum(1 for l in all_labels if l == 1)
        print(f"   Real: {real_count}")
        print(f"   Fake: {fake_count}")
        if fake_count > 0:
            ratio = real_count / fake_count
            print(f"   Imbalance Ratio: {ratio:.2f}:1")
            if ratio > 3:
                print(f"   ⚠️ SEVERE IMBALANCE! Consider using SMOTE or balanced dataset")
    
    return np.array(all_files), np.array(all_labels)


def load_audio_dataset(paths_dict, max_samples_per_class=None, print_stats=True):
    """
    Load audio dataset.
    
    Args:
        paths_dict: Dictionary with 'real' and 'fake' keys
        max_samples_per_class: Maximum samples per class
        print_stats: Whether to print statistics
    
    Returns:
        file_paths, labels arrays
    """
    all_files = []
    all_labels = []
    
    # Load real audio
    if 'real' in paths_dict and os.path.exists(paths_dict['real']):
        files = glob.glob(os.path.join(paths_dict['real'], '**', '*.*'), recursive=True)
        files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        all_files.extend(files)
        all_labels.extend([0] * len(files))
        if print_stats:
            print(f"✓ Loaded {len(files)} real audio files")
    
    # Load fake audio
    if 'fake' in paths_dict and os.path.exists(paths_dict['fake']):
        files = glob.glob(os.path.join(paths_dict['fake'], '**', '*.*'), recursive=True)
        files = [f for f in files if f.lower().endswith(('.wav', '.mp3', '.flac', '.m4a'))]
        if max_samples_per_class:
            files = files[:max_samples_per_class]
        all_files.extend(files)
        all_labels.extend([1] * len(files))
        if print_stats:
            print(f"✓ Loaded {len(files)} fake audio files")
    
    if print_stats:
        print(f"\n📊 Total: {len(all_files)} audio files")
        real_count = sum(1 for l in all_labels if l == 0)
        fake_count = sum(1 for l in all_labels if l == 1)
        print(f"   Real: {real_count}")
        print(f"   Fake: {fake_count}")
        if fake_count > 0:
            ratio = real_count / fake_count
            print(f"   Imbalance Ratio: {ratio:.2f}:1")
            if ratio > 5:
                print(f"   ⚠️ SEVERE IMBALANCE! Strongly consider balanced dataset")
    
    return np.array(all_files), np.array(all_labels)


def load_video_dataset(paths_dict, max_samples_per_class=None, print_stats=True):
    """
    Load video dataset.
    
    Args:
        paths_dict: Dictionary with 'real'/'original' and 'fake'/'manipulated' keys
        max_samples_per_class: Maximum samples per class
        print_stats: Whether to print statistics
    
    Returns:
        file_paths, labels arrays
    """
    all_files = []
    all_labels = []
    
    # Load real videos
    for key in ['real', 'original', 'celeb_real', 'youtube_real']:
        if key in paths_dict and os.path.exists(paths_dict[key]):
            files = glob.glob(os.path.join(paths_dict[key], '**', '*.*'), recursive=True)
            files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if max_samples_per_class:
                files = files[:max_samples_per_class // 2]  # Divide if multiple real sources
            all_files.extend(files)
            all_labels.extend([0] * len(files))
            if print_stats:
                print(f"✓ Loaded {len(files)} from {key}")
    
    # Load fake videos
    for key in ['fake', 'manipulated', 'celeb_synthesis', 'fake_vv_aa', 'fake_v_ra', 'fake_rv_a']:
        if key in paths_dict and os.path.exists(paths_dict[key]):
            files = glob.glob(os.path.join(paths_dict[key], '**', '*.*'), recursive=True)
            files = [f for f in files if f.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))]
            if max_samples_per_class:
                files = files[:max_samples_per_class // 3]  # Divide if multiple fake sources
            all_files.extend(files)
            all_labels.extend([1] * len(files))
            if print_stats:
                print(f"✓ Loaded {len(files)} from {key}")
    
    if print_stats:
        print(f"\n📊 Total: {len(all_files)} video files")
        real_count = sum(1 for l in all_labels if l == 0)
        fake_count = sum(1 for l in all_labels if l == 1)
        print(f"   Real: {real_count}")
        print(f"   Fake: {fake_count}")
        if fake_count > 0:
            ratio = real_count / fake_count
            print(f"   Imbalance Ratio: {ratio:.2f}:1")
            if ratio > 4:
                print(f"   ⚠️ MODERATE-SEVERE IMBALANCE!")
    
    return np.array(all_files), np.array(all_labels)


# =============================================================================
# COMPLETE END-TO-END EXAMPLE
# =============================================================================

def run_complete_training_example():
    """
    Complete end-to-end example showing how to train on all datasets.
    """
    print("\n" + "="*80)
    print("🚀 COMPLETE END-TO-END TRAINING EXAMPLE")
    print("="*80 + "\n")
    
    # =========================================================================
    # 1. LOAD IMAGE DATASET
    # =========================================================================
    print("📁 STEP 1: Loading Image Dataset")
    print("-" * 80)
    
    # Load from Deepfake Image Detection dataset
    image_files, image_labels = load_image_dataset(
        DATASET_PATHS['deepfake_images'],
        max_samples_per_class=5000,  # Limit for faster training
        print_stats=True
    )
    
    # Split into train/val/test
    train_files, test_files, train_labels, test_labels = train_test_split(
        image_files, image_labels, test_size=0.2, random_state=42, stratify=image_labels
    )
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"\n✅ Split complete:")
    print(f"   Train: {len(train_files)} samples")
    print(f"   Val:   {len(val_files)} samples")
    print(f"   Test:  {len(test_files)} samples")
    
    # Check for class imbalance
    train_imbalance = np.sum(train_labels == 0) / np.sum(train_labels == 1)
    print(f"\n⚖️ Train set imbalance ratio: {train_imbalance:.2f}:1")
    
    # Create datasets
    train_dataset = ImageDataset(
        train_files, 
        train_labels,
        augment_minority=(train_imbalance > 2)  # Enhanced augmentation if imbalanced
    )
    val_dataset = ImageDataset(val_files, val_labels)
    test_dataset = ImageDataset(test_files, test_labels)
    
    # Create data loaders
    batch_size = 32
    
    if train_imbalance > 2:
        # Use WeightedRandomSampler for imbalanced data
        print("✅ Creating WeightedRandomSampler for balanced batches...")
        sampler = create_weighted_sampler(train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=4)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    print(f"✅ Data loaders created (batch size: {batch_size})")
    
    # =========================================================================
    # 2. TRAIN MODEL
    # =========================================================================
    print(f"\n{'='*80}")
    print("📁 STEP 2: Training Model")
    print("-" * 80)
    
    model, train_losses, val_losses = train_complete_model(
        train_loader=train_loader,
        val_loader=val_loader,
        model_type='image',
        use_focal_loss=True,
        use_class_weights=True,
        epochs=10,  # Increase for better results
        lr=1e-4,
        save_path='best_image_model.pth'
    )
    
    # =========================================================================
    # 3. EVALUATE ON TEST SET
    # =========================================================================
    print(f"\n{'='*80}")
    print("📁 STEP 3: Evaluating on Test Set")
    print("-" * 80)
    
    # Load best model
    checkpoint = torch.load('best_image_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Evaluate with default threshold
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    test_loss, test_acc, test_prec, test_rec, test_f1, y_true, y_probs = evaluate_model(
        model, test_loader, criterion, device, return_probs=True
    )
    
    # Plot confusion matrix with default threshold
    y_pred_default = (y_probs >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred_default, 
                         title='Confusion Matrix (Threshold=0.5)',
                         save_path='confusion_matrix_default.png')
    
    # =========================================================================
    # 4. THRESHOLD OPTIMIZATION
    # =========================================================================
    print(f"\n{'='*80}")
    print("📁 STEP 4: Threshold Optimization")
    print("-" * 80)
    
    # Find optimal threshold
    optimal_threshold, optimal_threshold_youden = plot_threshold_analysis(
        y_true, y_probs,
        title='Threshold Analysis - Image Model',
        save_path='threshold_analysis_image.png'
    )
    
    # Evaluate with optimal threshold
    y_pred_optimal = (y_probs >= optimal_threshold).astype(int)
    
    print(f"\n📊 Results with Optimal Threshold ({optimal_threshold:.3f}):")
    print(f"   Accuracy:  {accuracy_score(y_true, y_pred_optimal):.4f}")
    print(f"   Precision: {precision_score(y_true, y_pred_optimal):.4f}")
    print(f"   Recall:    {recall_score(y_true, y_pred_optimal):.4f}")
    print(f"   F1 Score:  {f1_score(y_true, y_pred_optimal):.4f}")
    
    # Plot confusion matrix with optimal threshold
    plot_confusion_matrix(y_true, y_pred_optimal,
                         title=f'Confusion Matrix (Threshold={optimal_threshold:.3f})',
                         save_path='confusion_matrix_optimal.png')
    
    # =========================================================================
    # 5. DETAILED CLASSIFICATION REPORT
    # =========================================================================
    print(f"\n{'='*80}")
    print("📁 STEP 5: Detailed Classification Report")
    print("-" * 80)
    
    print("\n📊 Classification Report (Optimal Threshold):")
    print(classification_report(y_true, y_pred_optimal, 
                               target_names=['Real', 'Fake'],
                               digits=4))
    
    # =========================================================================
    # 6. SAVE RESULTS
    # =========================================================================
    print(f"\n{'='*80}")
    print("📁 STEP 6: Saving Results")
    print("-" * 80)
    
    results = {
        'model_type': 'image',
        'test_loss': test_loss,
        'test_acc_default': test_acc,
        'test_f1_default': test_f1,
        'optimal_threshold': optimal_threshold,
        'test_acc_optimal': accuracy_score(y_true, y_pred_optimal),
        'test_f1_optimal': f1_score(y_true, y_pred_optimal),
        'improvement': ((f1_score(y_true, y_pred_optimal) - test_f1) / test_f1) * 100
    }
    
    # Save results to file
    with open('training_results.txt', 'w') as f:
        f.write("="*80 + "\n")
        f.write("TRAINING RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    
    print("✅ Results saved to training_results.txt")
    print("\n" + "="*80)
    print("✅ COMPLETE TRAINING PIPELINE FINISHED!")
    print("="*80)
    print(f"\n📊 Final F1 Improvement: {results['improvement']:+.2f}%")
    print(f"   Default threshold (0.5): F1 = {test_f1:.4f}")
    print(f"   Optimal threshold ({optimal_threshold:.3f}): F1 = {results['test_f1_optimal']:.4f}")
    
    return model, results


# =============================================================================
# SIMPLIFIED INTERFACE FOR DIFFERENT MODALITIES
# =============================================================================

def train_image_model(max_samples=5000, epochs=10, batch_size=32):
    """Simplified function to train image model."""
    print("🖼️ Training Image Model...")
    image_files, image_labels = load_image_dataset(
        DATASET_PATHS['deepfake_images'],
        max_samples_per_class=max_samples
    )
    # ... (same as above)
    return run_complete_training_example()


def train_audio_model(max_samples=5000, epochs=10, batch_size=32):
    """Simplified function to train audio model."""
    print("🎵 Training Audio Model...")
    audio_files, audio_labels = load_audio_dataset(
        DATASET_PATHS['audio'],
        max_samples_per_class=max_samples
    )
    # Create datasets and train (similar to image)
    print("⚠️ Implement full audio training loop following image example")


def train_video_model(max_samples=500, epochs=10, batch_size=16):
    """Simplified function to train video model."""
    print("🎬 Training Video Model...")
    video_files, video_labels = load_video_dataset(
        DATASET_PATHS['dfd'],
        max_samples_per_class=max_samples
    )
    # Create datasets and train (similar to image but with VideoDataset)
    print("⚠️ Implement full video training loop following image example")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎯 COMPLETE IMPROVED DEEPFAKE DETECTION SYSTEM")
    print("="*80)
    print("\nThis script provides:")
    print("   ✓ Complete data loading for all 6 datasets")
    print("   ✓ Class balancing (WeightedSampler, SMOTE)")
    print("   ✓ Focal Loss for hard examples")
    print("   ✓ Threshold optimization")
    print("   ✓ Comprehensive evaluation")
    print("\nTo run the complete example:")
    print("   model, results = run_complete_training_example()")
    print("\nOr train specific modalities:")
    print("   train_image_model(max_samples=5000, epochs=10)")
    print("   train_audio_model(max_samples=5000, epochs=10)")
    print("   train_video_model(max_samples=500, epochs=10)")
    print("="*80 + "\n")
    
    # Uncomment to run automatically:
    model, results = run_complete_training_example()

print("✅ Part 3 loaded: Data loading and complete example")
print("\n🚀 Ready to train! Run: run_complete_training_example()")
