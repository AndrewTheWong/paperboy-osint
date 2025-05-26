#!/usr/bin/env python3
"""
Test script for the Streamlit UI.
"""
import unittest
import sys
import os
import importlib
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_streamlit_ui')

class TestStreamlitUI(unittest.TestCase):
    """Test suite for the Streamlit UI."""

    def test_imports(self):
        """Test that all necessary modules can be imported."""
        try:
            import streamlit
            logger.info("Successfully imported streamlit")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"Import error for streamlit: {str(e)}")
            self.skipTest(f"Streamlit not installed: {e}")
    
    def test_tag_review_imports(self):
        """Test that the human_tag_review module can be imported."""
        try:
            # Add project root to path
            project_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(project_root))
            
            # Import the module
            from ui.human_tag_review import main, load_articles, tag_to_flat, flat_to_tag
            logger.info("Successfully imported human_tag_review module")
            
            # Test that tag utils was imported properly
            from ui.human_tag_review import KEYWORD_MAP
            self.assertIsNotNone(KEYWORD_MAP)
            logger.info("Successfully imported KEYWORD_MAP from tag_utils")
            
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import human_tag_review module: {e}")
    
    def test_tag_conversion(self):
        """Test the tag conversion functions."""
        try:
            # Add project root to path
            project_root = Path(__file__).resolve().parent.parent
            sys.path.insert(0, str(project_root))
            
            # Import the functions
            from ui.human_tag_review import tag_to_flat, flat_to_tag, ALL_TAGS
            
            # Test tag_to_flat with a known tag
            flat_tag = tag_to_flat("military")
            self.assertIsNotNone(flat_tag)
            self.assertEqual(flat_tag, "Military & Security – military")
            
            # Test flat_to_tag with a known flat tag
            raw_tag = flat_to_tag("Military & Security – military")
            self.assertEqual(raw_tag, "military")
            
            # Test conversion round-trip for all tags
            for category, tags in ALL_TAGS.items():
                for tag in tags:
                    flat = tag_to_flat(tag)
                    self.assertIsNotNone(flat)
                    self.assertTrue(flat.startswith(category))
                    
                    # Convert back
                    raw = flat_to_tag(flat)
                    self.assertEqual(raw, tag)
            
            logger.info("Tag conversion functions working correctly")
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import tag conversion functions: {e}")
        except Exception as e:
            logger.error(f"Error testing tag conversion: {str(e)}")
            self.fail(f"Failed to test tag conversion: {e}")

if __name__ == "__main__":
    unittest.main() 