#!/usr/bin/env python3
"""
Script to move all test_*.py files to a tests directory
and update imports if needed.
"""
import os
import shutil
from pathlib import Path
import sys

def move_tests():
    """Move all test_*.py files to the tests directory."""
    # Create tests directory if it doesn't exist
    test_dir = Path('tests')
    test_dir.mkdir(exist_ok=True)
    
    # Create __init__.py in the tests directory
    init_file = test_dir / "__init__.py"
    if not init_file.exists():
        with open(init_file, 'w') as f:
            f.write("# Tests package\n")
    
    # Find all test files
    test_files = list(Path('.').glob('test_*.py'))
    
    print(f"Found {len(test_files)} test files")
    
    # Move each file
    for file_path in test_files:
        dest_path = test_dir / file_path.name
        
        # Skip if file is already in tests directory
        if str(file_path).startswith('tests/'):
            continue
        
        # Skip if file already exists in destination
        if dest_path.exists():
            print(f"Skipping {file_path}: already exists in tests directory")
            continue
        
        try:
            # Copy first, then delete (safer than direct move)
            shutil.copy2(file_path, dest_path)
            print(f"Moved {file_path} to {dest_path}")
            
            # Delete original file only after successful copy
            os.remove(file_path)
        except Exception as e:
            print(f"Error moving {file_path}: {e}")
    
    print("\nDone! All test files have been moved to the tests directory.")
    print("You may need to update imports in the test files.")

if __name__ == "__main__":
    move_tests() 