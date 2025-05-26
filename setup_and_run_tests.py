#!/usr/bin/env python3
"""
Script to setup tests and run them in sequence.
This:
1. Moves test files to tests directory
2. Fixes imports in test files
3. Runs all tests
"""
import os
import sys
import subprocess
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup_runner')

def run_script(script_name, description):
    """Run a Python script and return success status."""
    logger.info(f"Running {description}...")
    
    if not Path(script_name).exists():
        logger.error(f"Script {script_name} not found!")
        return False
    
    try:
        result = subprocess.run(
            [sys.executable, script_name],
            check=True,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        # Log the output
        for line in result.stdout.splitlines():
            if line.strip():
                logger.info(f"{script_name}: {line.strip()}")
        
        logger.info(f"✅ {description} completed successfully")
        return True
    except subprocess.TimeoutExpired:
        logger.error(f"⏱️ {description} timed out")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ {description} failed with exit code {e.returncode}")
        
        # Log the error output
        for line in e.stderr.splitlines():
            if line.strip():
                logger.error(f"{script_name} error: {line.strip()}")
        
        return False
    except Exception as e:
        logger.error(f"💥 Error running {description}: {str(e)}")
        return False

def main():
    """Run all setup and test steps in sequence."""
    logger.info("Starting Paperboy test setup and execution")
    
    # 1. Move test files to tests directory
    move_success = run_script('move_tests_to_folder.py', 'Moving test files')
    if not move_success:
        logger.warning("⚠️ Moving tests failed, but continuing...")
    
    # 2. Fix imports in test files
    fix_success = run_script('fix_test_imports.py', 'Fixing test imports')
    if not fix_success:
        logger.warning("⚠️ Fixing imports failed, but continuing...")
    
    # 3. Run all tests
    test_success = run_script('run_all_tests.py', 'Running all tests')
    
    # Summary
    logger.info("\n--- SETUP AND TEST SUMMARY ---")
    logger.info(f"Move tests:      {'✅ Success' if move_success else '❌ Failed'}")
    logger.info(f"Fix imports:     {'✅ Success' if fix_success else '❌ Failed'}")
    logger.info(f"Run all tests:   {'✅ Success' if test_success else '❌ Failed'}")
    
    # Overall status
    if test_success:
        logger.info("\n✅ Setup and tests completed successfully")
        return 0
    else:
        logger.error("\n🔥 Setup and tests completed with issues")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 