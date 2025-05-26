#!/usr/bin/env python3
"""
Test script for the dynamic scraper.
"""
import unittest
import sys
import os
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_dynamic_scraper')

class TestDynamicScraper(unittest.TestCase):
    """Test suite for the dynamic scraper module."""

    def test_imports(self):
        """Test that all necessary modules can be imported."""
        try:
            from pipelines.dynamic_scraper import scrape_all_dynamic
            logger.info("Successfully imported dynamic_scraper functions")
            self.assertTrue(True)
        except ImportError as e:
            logger.error(f"Import error: {str(e)}")
            self.fail(f"Failed to import dynamic_scraper module: {e}")
    
    def test_config_file(self):
        """Test that the config file exists and can be loaded."""
        try:
            import json
            config_path = Path('config/sources_config.json')
            self.assertTrue(config_path.exists(), "Config file does not exist")
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                logger.info(f"Successfully loaded config with {len(config)} sources")
                self.assertTrue(len(config) > 0, "Config file is empty")
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Config file error: {str(e)}")
            self.fail(f"Failed to load config file: {e}")

    def test_scrape_function(self):
        """Test that the scrape_all_dynamic function returns articles."""
        try:
            from pipelines.dynamic_scraper import scrape_all_dynamic
            articles = scrape_all_dynamic()
            
            self.assertIsNotNone(articles, "scrape_all_dynamic returned None")
            self.assertIsInstance(articles, list, "scrape_all_dynamic did not return a list")
            
            # Should return at least the sample articles even if no scraping occurs
            self.assertTrue(len(articles) > 0, "No articles were returned")
            
            # Check article structure
            for article in articles:
                self.assertIsInstance(article, dict, "Article is not a dictionary")
                self.assertIn('title', article, "Article has no title")
                self.assertIn('url', article, "Article has no URL")
                self.assertIn('source', article, "Article has no source")
                self.assertIn('scraped_at', article, "Article has no scraped_at timestamp")
        
        except Exception as e:
            logger.error(f"Error testing scrape function: {str(e)}")
            self.fail(f"Failed to test scrape function: {e}")

if __name__ == "__main__":
    unittest.main() 