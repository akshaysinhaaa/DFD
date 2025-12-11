"""
Quick Fix Script - Automatically patch the training script to fix freezing issue.

This script will:
1. Find all DataLoader instances in train_from_notebook_14_clean.py
2. Change num_workers from any value to 0
3. Create a backup of the original file
4. Apply the fix

Usage:
    python apply_quick_fix.py
"""

import re
import shutil
from datetime import datetime

def apply_dataloader_fix(script_path='train_from_notebook_14_clean.py'):
    """
    Automatically fix the DataLoader num_workers parameter.
    """
    print("="*80)
    print("🔧 APPLYING QUICK FIX FOR FREEZING ISSUE")
    print("="*80)
    print()
    
    # Read the script
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ Error: {script_path} not found!")
        print(f"   Make sure you're in the Multimodal_DeepFake_Detection directory")
        return False
    
    # Create backup
    backup_path = f"{script_path}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copy2(script_path, backup_path)
    print(f"✅ Backup created: {backup_path}")
    
    # Find and replace DataLoader with num_workers
    pattern = r'DataLoader\((.*?),\s*num_workers\s*=\s*\d+\s*(.*?)\)'
    matches = re.finditer(pattern, content, re.DOTALL)
    
    count = 0
    for match in matches:
        count += 1
        print(f"\n📍 Found DataLoader #{count}:")
        print(f"   Original: ...num_workers={match.group(0).split('num_workers=')[1].split(',')[0].split(')')[0]}...")
    
    # Replace all occurrences
    new_content = re.sub(
        r'(DataLoader\(.*?),\s*num_workers\s*=\s*\d+\s*(.*?\))',
        r'\1, num_workers=0\2',
        content,
        flags=re.DOTALL
    )
    
    # Also handle cases where num_workers might be the last parameter
    new_content = re.sub(
        r'(DataLoader\(.*?),\s*num_workers\s*=\s*\d+\s*\)',
        r'\1, num_workers=0)',
        new_content,
        flags=re.DOTALL
    )
    
    # Write the fixed version
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(new_content)
    
    print(f"\n✅ Applied fix: Changed {count} DataLoader instances to num_workers=0")
    print(f"✅ Fixed script saved: {script_path}")
    print()
    print("="*80)
    print("🎉 FIX APPLIED SUCCESSFULLY!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Restart your training script")
    print("2. Training should now continue past 9% without freezing")
    print("3. If you need to restore: copy from backup file")
    print()
    
    return True


if __name__ == "__main__":
    import os
    
    print()
    print("This script will fix the freezing issue by setting num_workers=0")
    print("in all DataLoader instances.")
    print()
    
    # Check if we're in the right directory
    if not os.path.exists('train_from_notebook_14_clean.py'):
        print("❌ Error: train_from_notebook_14_clean.py not found!")
        print()
        print("Please run this script from the Multimodal_DeepFake_Detection directory:")
        print("  cd Multimodal_DeepFake_Detection")
        print("  python apply_quick_fix.py")
        print()
    else:
        apply_dataloader_fix()
