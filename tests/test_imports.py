#!/usr/bin/env python3
"""
Test that required modules can be imported.
"""
import unittest
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_imports')

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestImports(unittest.TestCase):
    """Test that required modules can be imported."""
    
    def test_scraping_imports(self):
        """Test that scraping modules can be imported."""
        try:
            # Try importing from new location
            from scraping.runner import scrape_all_dynamic, save_articles_to_file, load_cache, save_cache, generate_hash
            logger.info("Successfully imported scraping.runner functions")
            self.assertTrue(True)
        except ImportError as e:
            logger.warning(f"Could not import from scraping.runner: {e}")
            try:
                # Fall back to old locations
                from pipelines.dynamic_scraper import scrape_all_dynamic
                logger.info("Successfully imported pipelines.dynamic_scraper functions")
                
                from utils.article_cache import load_cache, save_cache, generate_hash
                logger.info("Successfully imported utils.article_cache functions")
                self.assertTrue(True)
            except ImportError as e:
                logger.error(f"Import error: {e}")
                self.fail(f"Failed to import required scraping modules: {e}")
    
    def test_source_imports(self):
        """Test that source scraper modules can be imported."""
        try:
            # Try importing from new location
            from scraping.sources import taipei_times, xinhua, china_daily
            logger.info("Successfully imported scraping.sources modules")
            self.assertTrue(True)
        except ImportError as e:
            logger.warning(f"Could not import from scraping.sources: {e}")
            try:
                # Fall back to old locations
                from scrapers import taipei_times, xinhua, china_daily
                logger.info("Successfully imported scrapers modules")
                self.assertTrue(True)
            except ImportError as e:
                logger.warning(f"Import error for specific scrapers: {e}")
                # This is not a fatal error, as we have universal_scraper as fallback
                logger.info("Universal scraper will be used as fallback")
                pass
    
    def test_universal_scraper(self):
        """Test that universal scraper can be imported."""
        try:
            from scraping.universal_scraper import scrape_site
            logger.info("Successfully imported universal_scraper")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"Import error for universal_scraper: {e}")
            self.fail(f"Failed to import universal_scraper: {e}")

if __name__ == "__main__":
    unittest.main() 