#!/usr/bin/env python3
"""
Script to run all tests for the Paperboy pipeline.
This runs the unit tests and also performs a full pipeline test with mock data.
"""
import os
import sys
import unittest
import subprocess
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_runner')

def run_unit_tests():
    """Run all the unit tests for the pipeline"""
    logger.info("Running unit tests...")
    
    # Use test discovery from the tests directory
    test_dir = Path('tests')
    if not test_dir.exists() or not test_dir.is_dir():
        logger.error("❌ Tests directory not found! Create it with proper tests.")
        return False
    
    logger.info(f"Discovering tests in {test_dir} directory")
    
    # Create test suite using loader discovery
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Track statistics
    total_tests = result.testsRun
    passed_tests = total_tests - len(result.errors) - len(result.failures)
    failed_tests = len(result.failures)
    error_tests = len(result.errors)
    skipped_tests = len(result.skipped)
    
    # Print summary
    logger.info("\n--- TEST SUMMARY ---")
    logger.info(f"Total tests run:    {total_tests}")
    logger.info(f"Passed:             {passed_tests}")
    logger.info(f"Failed:             {failed_tests}")
    logger.info(f"Errors:             {error_tests}")
    logger.info(f"Skipped:            {skipped_tests}")
    
    # Track failed test names
    if failed_tests > 0:
        logger.error("\nFailed tests:")
        for failure in result.failures:
            logger.error(f"  - {failure[0]}")
    
    # Track error test names
    if error_tests > 0:
        logger.error("\nTests with errors:")
        for error in result.errors:
            logger.error(f"  - {error[0]}")
    
    # Overall success/failure
    success = failed_tests == 0 and error_tests == 0
    
    if success:
        logger.info("\n✅ All unit tests passed!")
    else:
        logger.error(f"\n❌ {failed_tests + error_tests} tests failed or had errors")
    
    return success

def run_mock_pipeline():
    """Run the pipeline with mock data"""
    logger.info("Running mock pipeline test...")
    
    # Check if the mock pipeline script exists
    if not Path('run_pipeline_with_mocks.py').exists():
        logger.error("❌ run_pipeline_with_mocks.py not found")
        return False
    
    try:
        # Run the mock pipeline with 5 articles
        result = subprocess.run(
            [sys.executable, 'run_pipeline_with_mocks.py', '--count', '5'],
            check=True,
            capture_output=True,
            text=True,
            timeout=60  # 60 second timeout for the full pipeline
        )
        
        # Log the output
        for line in result.stdout.splitlines():
            if line.strip():
                logger.info(f"Pipeline: {line.strip()}")
        
        logger.info("✅ Mock pipeline test completed successfully")
        return True
    except subprocess.TimeoutExpired:
        logger.error("⏱️ Mock pipeline test timed out after 60 seconds")
        return False
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Mock pipeline test failed with exit code {e.returncode}")
        
        # Log the error output
        for line in e.stderr.splitlines():
            if line.strip():
                logger.error(f"Pipeline error: {line.strip()}")
        
        return False

def check_streamlit_installation():
    """Check if Streamlit is installed"""
    logger.info("Checking Streamlit installation...")
    
    try:
        # Try to run streamlit --version
        result = subprocess.run(
            ['streamlit', '--version'],
            check=True,
            capture_output=True,
            text=True,
            timeout=10  # 10 second timeout
        )
        
        version = result.stdout.strip()
        logger.info(f"✅ Streamlit is installed: {version}")
        return True
    except subprocess.TimeoutExpired:
        logger.warning("⏱️ Streamlit version check timed out")
        return False
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.warning("⚠️ Streamlit is not installed or not in PATH")
        logger.info("You can install it with: pip install streamlit")
        return False

def main():
    """Run all tests"""
    start_time = time.time()
    logger.info("Starting Paperboy pipeline tests")
    
    # Create tests directory if it doesn't exist
    test_dir = Path('tests')
    if not test_dir.exists():
        logger.info("Creating tests directory")
        test_dir.mkdir()
        logger.info(f"Created directory: {test_dir}")
        # Create __init__.py file in tests directory
        init_file = test_dir / "__init__.py"
        with open(init_file, 'w') as f:
            f.write("# Tests package\n")
        logger.info("Created __init__.py in tests directory")
    
    # Track test results
    results = {}
    
    # Check Streamlit installation
    results['streamlit'] = check_streamlit_installation()
    
    # Run unit tests
    results['unit_tests'] = run_unit_tests()
    
    # Run mock pipeline
    results['mock_pipeline'] = run_mock_pipeline()
    
    # Calculate total run time
    end_time = time.time()
    run_time = end_time - start_time
    
    # Print summary
    logger.info("\n--- TEST SUMMARY ---")
    logger.info(f"Streamlit installed: {'✅ Yes' if results['streamlit'] else '⚠️ No'}")
    logger.info(f"Unit tests: {'✅ Passed' if results['unit_tests'] else '❌ Failed'}")
    logger.info(f"Mock pipeline: {'✅ Passed' if results['mock_pipeline'] else '❌ Failed'}")
    logger.info(f"Total run time: {run_time:.2f} seconds")
    
    # Overall result
    all_passed = all(v is True for v in [results['unit_tests'], results['mock_pipeline']])
    if all_passed:
        logger.info("\n✅ ALL TESTS PASSED")
    else:
        failed_count = sum(1 for v in [results['unit_tests'], results['mock_pipeline']] if v is False)
        logger.error(f"\n🔥 Issues found: {failed_count}")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main()) 