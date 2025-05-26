#!/usr/bin/env python3
"""
Run a specific test file with detailed debugging output.
"""
import os
import sys
import subprocess
import argparse
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_specific_test(test_file_path, verbose=False):
    """
    Run a specific test file and display detailed output.
    
    Args:
        test_file_path: Path to the test file
        verbose: Whether to run in verbose mode
    """
    test_file = Path(test_file_path)
    
    if not test_file.exists():
        logger.error(f"Test file {test_file} does not exist")
        return False
    
    logger.info(f"Running test: {test_file}")
    
    # Add parent directory to sys.path for tests in tests/ directory
    if "tests" in str(test_file):
        parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(str(test_file))))
        sys.path.insert(0, parent_dir)
        logger.info(f"Added {parent_dir} to sys.path")
    
    # Run the test with unittest directly to get detailed output
    cmd = [sys.executable, "-m", "unittest", str(test_file)]
    if verbose:
        cmd.append("-v")
    
    logger.info(f"Running command: {' '.join(cmd)}")
    
    try:
        # Run the test in a subprocess
        process = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout
        )
        
        # Log full output
        if process.stdout:
            for line in process.stdout.splitlines():
                if line.strip():
                    logger.info(f"Output: {line}")
        
        # Log any errors
        if process.stderr:
            for line in process.stderr.splitlines():
                if line.strip():
                    logger.error(f"Error: {line}")
        
        if process.returncode == 0:
            logger.info(f"✅ Test passed: {test_file}")
            return True
        else:
            logger.error(f"❌ Test failed with code {process.returncode}: {test_file}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ Test timed out after 60 seconds: {test_file}")
        return False
    except Exception as e:
        logger.error(f"💥 Error running test: {str(e)}")
        return False

def main():
    """Run the specific test file."""
    parser = argparse.ArgumentParser(description="Run a specific test file with detailed output")
    parser.add_argument("test_file", help="Path to the test file to run")
    parser.add_argument("-v", "--verbose", action="store_true", help="Run in verbose mode")
    
    args = parser.parse_args()
    
    success = run_specific_test(args.test_file, args.verbose)
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 