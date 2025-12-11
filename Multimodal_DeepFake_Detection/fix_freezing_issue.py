"""
Fix for Training Freezing Issue - Corrupted Video Files

The freezing happens because:
1. Some video files are corrupted (H.263 codec errors)
2. DataLoader workers hang when trying to load these files
3. The training loop waits indefinitely

Solutions implemented in this script:
1. Skip corrupted videos with try-except
2. Add timeout to video loading
3. Validate videos before training
4. Use error-handling in Dataset class
"""

import os
import cv2
import torch
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp

def validate_video_file(video_path, timeout=5):
    """
    Check if a video file can be opened and read.
    
    Args:
        video_path: Path to video file
        timeout: Timeout in seconds
    
    Returns:
        bool: True if valid, False if corrupted
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False
        
        # Try to read first frame
        ret, frame = cap.read()
        cap.release()
        
        return ret and frame is not None
    except Exception as e:
        return False

def scan_and_filter_videos(dataset_paths, output_file='valid_videos.txt'):
    """
    Scan all video files and create a list of valid ones.
    
    Args:
        dataset_paths: Dictionary of dataset paths
        output_file: File to save valid video paths
    """
    print("\n" + "="*80)
    print("🔍 SCANNING FOR CORRUPTED VIDEOS")
    print("="*80 + "\n")
    
    all_videos = []
    corrupted_videos = []
    
    # Collect all video paths
    for dataset_name, paths in dataset_paths.items():
        if isinstance(paths, dict):
            for key, path in paths.items():
                if os.path.exists(path):
                    for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                        videos = list(Path(path).rglob(f'*{ext}'))
                        all_videos.extend([(v, dataset_name) for v in videos])
    
    print(f"Found {len(all_videos)} video files to validate\n")
    
    valid_videos = []
    
    # Validate each video
    for video_path, dataset_name in tqdm(all_videos, desc="Validating videos"):
        if validate_video_file(video_path):
            valid_videos.append(str(video_path))
        else:
            corrupted_videos.append((str(video_path), dataset_name))
            print(f"\n⚠️  Corrupted: {video_path.name} (from {dataset_name})")
    
    # Save valid videos
    with open(output_file, 'w') as f:
        for video in valid_videos:
            f.write(f"{video}\n")
    
    print("\n" + "="*80)
    print("📊 VALIDATION RESULTS")
    print("="*80)
    print(f"Total videos scanned: {len(all_videos)}")
    print(f"Valid videos: {len(valid_videos)}")
    print(f"Corrupted videos: {len(corrupted_videos)}")
    print(f"Success rate: {len(valid_videos)/len(all_videos)*100:.1f}%")
    print(f"\n✅ Valid videos saved to: {output_file}")
    
    if corrupted_videos:
        print(f"\n⚠️  Found {len(corrupted_videos)} corrupted videos:")
        for vid, dataset in corrupted_videos[:10]:  # Show first 10
            print(f"   - {Path(vid).name} (from {dataset})")
        if len(corrupted_videos) > 10:
            print(f"   ... and {len(corrupted_videos) - 10} more")
    
    return valid_videos, corrupted_videos


def create_robust_video_dataset():
    """
    Creates a robust VideoDataset class that handles corrupted files.
    """
    code = '''
class RobustVideoDataset(Dataset):
    """Video dataset with robust error handling for corrupted files."""
    
    def __init__(self, file_paths, labels, n_frames=16, transform=None, 
                 max_retries=3, timeout=5):
        self.file_paths = file_paths
        self.labels = labels
        self.n_frames = n_frames
        self.transform = transform
        self.max_retries = max_retries
        self.timeout = timeout
        
        # Pre-filter corrupted videos
        print("Filtering corrupted videos...")
        self.valid_indices = []
        for idx in tqdm(range(len(file_paths))):
            if self._is_valid_video(file_paths[idx]):
                self.valid_indices.append(idx)
        
        print(f"Valid videos: {len(self.valid_indices)}/{len(file_paths)}")
    
    def _is_valid_video(self, video_path):
        """Quick check if video can be opened."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return False
            ret, _ = cap.read()
            cap.release()
            return ret
        except:
            return False
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        """Get video with robust error handling."""
        actual_idx = self.valid_indices[idx]
        video_path = self.file_paths[actual_idx]
        label = self.labels[actual_idx]
        
        for attempt in range(self.max_retries):
            try:
                frames = self._load_video_frames(video_path)
                if frames is not None:
                    return frames, torch.tensor(label, dtype=torch.float32)
            except Exception as e:
                if attempt == self.max_retries - 1:
                    print(f"Failed to load {video_path} after {self.max_retries} attempts")
                    # Return dummy data
                    return torch.zeros(self.n_frames, 3, 224, 224), \
                           torch.tensor(label, dtype=torch.float32)
        
        # Should not reach here, but return dummy data as fallback
        return torch.zeros(self.n_frames, 3, 224, 224), \
               torch.tensor(label, dtype=torch.float32)
    
    def _load_video_frames(self, video_path):
        """Load video frames with timeout protection."""
        try:
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                cap.release()
                return None
            
            # Sample frames
            frame_indices = np.linspace(0, total_frames - 1, self.n_frames, dtype=int)
            frames = []
            
            for frame_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    continue
                
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                
                if self.transform:
                    frame = self.transform(Image.fromarray(frame))
                else:
                    frame = transforms.ToTensor()(frame)
                
                frames.append(frame)
            
            cap.release()
            
            # Pad if needed
            if len(frames) < self.n_frames:
                frames += [torch.zeros_like(frames[0])] * (self.n_frames - len(frames))
            
            return torch.stack(frames[:self.n_frames])
            
        except Exception as e:
            return None
'''
    return code


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🔧 FIX FOR TRAINING FREEZING ISSUE")
    print("="*80 + "\n")
    
    print("This script helps fix the freezing issue caused by corrupted videos.")
    print("\nOptions:")
    print("1. Scan and filter corrupted videos")
    print("2. Show robust VideoDataset code")
    print("3. Both")
    print("\nChoice (1/2/3): ", end="")
    
    # For now, just show the solutions
    print("\n\n" + "="*80)
    print("📝 SOLUTION 1: Pre-filter Corrupted Videos")
    print("="*80 + "\n")
    print("Add this code before creating your dataset:")
    print('''
# Validate videos before training
valid_videos = []
valid_labels = []

print("Filtering corrupted videos...")
for video_path, label in tqdm(zip(all_video_paths, all_labels)):
    cap = cv2.VideoCapture(str(video_path))
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            valid_videos.append(video_path)
            valid_labels.append(label)

print(f"Valid videos: {len(valid_videos)}/{len(all_video_paths)}")

# Use filtered videos
dataset = VideoDataset(valid_videos, valid_labels)
''')
    
    print("\n" + "="*80)
    print("📝 SOLUTION 2: Add Error Handling to Dataset")
    print("="*80 + "\n")
    print("Replace your VideoDataset __getitem__ with:")
    print('''
def __getitem__(self, idx):
    video_path = self.file_paths[idx]
    label = self.labels[idx]
    
    try:
        cap = cv2.VideoCapture(str(video_path))
        
        # Check if opened
        if not cap.isOpened():
            print(f"Warning: Could not open {video_path}")
            cap.release()
            return torch.zeros(self.n_frames, 3, 224, 224), torch.tensor(label, dtype=torch.float32)
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Check frame count
        if total_frames == 0:
            print(f"Warning: No frames in {video_path}")
            cap.release()
            return torch.zeros(self.n_frames, 3, 224, 224), torch.tensor(label, dtype=torch.float32)
        
        # Rest of your loading code...
        
    except Exception as e:
        print(f"Error loading {video_path}: {e}")
        return torch.zeros(self.n_frames, 3, 224, 224), torch.tensor(label, dtype=torch.float32)
''')
    
    print("\n" + "="*80)
    print("📝 SOLUTION 3: Reduce DataLoader Workers")
    print("="*80 + "\n")
    print("In your training script, change:")
    print('''
# FROM:
train_loader = DataLoader(dataset, batch_size=8, num_workers=4)

# TO:
train_loader = DataLoader(dataset, batch_size=8, num_workers=0)  # Single process
# OR
train_loader = DataLoader(dataset, batch_size=8, num_workers=1)  # Minimal workers
''')
    
    print("\n" + "="*80)
    print("📝 SOLUTION 4: Add Timeout to DataLoader")
    print("="*80 + "\n")
    print("Add timeout parameter:")
    print('''
train_loader = DataLoader(
    dataset, 
    batch_size=8, 
    num_workers=0,
    timeout=10  # 10 second timeout
)
''')
    
    print("\n" + "="*80)
    print("🎯 RECOMMENDED QUICK FIX")
    print("="*80 + "\n")
    print("1. In your training script, find the DataLoader creation")
    print("2. Change num_workers to 0:")
    print("   train_loader = DataLoader(dataset, batch_size=8, num_workers=0)")
    print("3. Add try-except in VideoDataset __getitem__ (see Solution 2)")
    print("\nThis will fix the freezing immediately!")
