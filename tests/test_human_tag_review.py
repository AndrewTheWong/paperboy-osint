#!/usr/bin/env python3
"""
Test script for the human tag review UI.
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
logger = logging.getLogger('test_human_tag_review')

class TestHumanTagReview(unittest.TestCase):
    """Test suite for the human tag review UI."""
    
    def test_imports(self):
        """Test that the UI module can be imported."""
        try:
            from ui.human_tag_review import main
            logger.info("Successfully imported human_tag_review.main")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import human_tag_review module: {e}")
    
    def test_tag_utils_import(self):
        """Test that the tag utils are imported correctly in the UI."""
        try:
            from ui.human_tag_review import KEYWORD_MAP
            logger.info("Successfully imported KEYWORD_MAP from human_tag_review")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import KEYWORD_MAP from human_tag_review: {e}")
    
    def test_tag_conversion(self):
        """Test the tag conversion functions."""
        try:
            from ui.human_tag_review import tag_to_flat, flat_to_tag
            
            # Create a test case
            test_flat_tag = "Military & Security – military"
            test_raw_tag = "military"
            
            # Test flat_to_tag
            self.assertEqual(flat_to_tag(test_flat_tag), test_raw_tag)
            
            # Test with unknown tag
            self.assertEqual(flat_to_tag("Unknown Category – test"), "test")
            
            logger.info("Tag conversion functions working correctly")
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import tag conversion functions: {e}")
        except Exception as e:
            logger.error(f"Error testing tag conversion: {str(e)}")
            self.fail(f"Failed to test tag conversion: {e}")

if __name__ == "__main__":
    unittest.main() 