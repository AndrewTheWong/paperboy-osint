#!/usr/bin/env python3
"""
Test script for the preprocessing pipeline.
"""
import unittest
import sys
import os
import logging
from pathlib import Path

# Add project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_preprocessing_pipeline')

class TestPreprocessingPipeline(unittest.TestCase):
    """Test suite for the preprocessing pipeline."""
    
    @unittest.skip("Missing dependencies for preprocessing pipeline")
    def test_pipeline_imports(self):
        """Test that the preprocessing modules can be imported."""
        try:
            from pipelines import preprocessing_pipeline
            logger.info("Successfully imported preprocessing_pipeline module")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import preprocessing_pipeline module: {e}")
    
    @unittest.skip("Missing dependencies for preprocessing pipeline")
    def test_preprocessing_functions(self):
        """Test the preprocessing functions."""
        try:
            from pipelines.preprocessing_pipeline import preprocess_text
            
            # Test with a simple example
            test_text = "Taiwan conducts military exercises amid regional tensions"
            processed_text = preprocess_text(test_text)
            
            # Check that some basic preprocessing was done
            self.assertIsInstance(processed_text, str, "Processed text should be a string")
            self.assertNotEqual(test_text, processed_text, "Preprocessing should modify the text")
            
            logger.info("Preprocessing functions working correctly")
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import preprocessing functions: {e}")
        except Exception as e:
            logger.error(f"Error testing preprocessing: {str(e)}")
            self.fail(f"Failed to test preprocessing: {e}")

if __name__ == "__main__":
    unittest.main() 