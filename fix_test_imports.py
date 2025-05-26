#!/usr/bin/env python3
"""
Script to fix imports in test files after moving them to the tests directory.
This adds parent directory to the path in each test file.
"""
import os
import re
from pathlib import Path

def fix_imports():
    """Fix imports in test files."""
    # Check if tests directory exists
    test_dir = Path('tests')
    if not test_dir.exists() or not test_dir.is_dir():
        print("Tests directory not found. Run move_tests.py first.")
        return
    
    # Find all test files
    test_files = list(test_dir.glob('test_*.py'))
    
    print(f"Found {len(test_files)} test files to fix")
    
    # Add parent directory to path in each test file
    for file_path in test_files:
        # Read file content
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check if import fix is already in place
        if "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))" in content:
            print(f"Skipping {file_path}: import fix already in place")
            continue
        
        # Check for other import fixes that might be in place
        if "sys.path.append" in content and "os.path.dirname" in content:
            print(f"Skipping {file_path}: has custom path manipulation")
            continue
        
        # Determine where to insert the import fix
        # Add after other imports or at the top
        import_block_end = 0
        
        # Find the import statements block
        import_matches = list(re.finditer(r'^(?:import|from)\s+\w+', content, re.MULTILINE))
        if import_matches:
            # Get the position after the last import statement
            last_import_match = import_matches[-1]
            import_line = content[last_import_match.start():content.find('\n', last_import_match.start()) + 1]
            import_block_end = content.find(import_line) + len(import_line)
        
        # Add the import fix
        new_content = content[:import_block_end]
        
        # Check if import of os and sys are present
        if not re.search(r'^import\s+os', content, re.MULTILINE):
            new_content += "import os\n"
        if not re.search(r'^import\s+sys', content, re.MULTILINE):
            new_content += "import sys\n"
        
        # Add the path fix
        new_content += "\n# Add parent directory to path\n"
        new_content += "sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))\n\n"
        
        # Add the rest of the content
        new_content += content[import_block_end:]
        
        # Write modified content back to file
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        print(f"Fixed imports in {file_path}")
    
    print("\nDone! All test files have been updated.")
    print("Now run run_all_tests.py to verify everything works.")

if __name__ == "__main__":
    fix_imports() 