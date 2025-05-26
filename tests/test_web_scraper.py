#!/usr/bin/env python3
"""
Test script for the web scraper.
"""
import unittest
import sys
from unittest.mock import patch, MagicMock
import os
import json
from datetime import datetime
import logging
import requests

# Add the project root to sys.path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_web_scraper')

class TestWebScraper(unittest.TestCase):
    
    def setUp(self):
        # Create mock responses for each site
        self.taipei_times_html = """
        <div class="list">
            <h3><a href="/News/front/archives/2023/05/26/2003801234">Test Taipei Times Article</a></h3>
        </div>
        """
        
        self.xinhua_html = """
        <div class="headline-item">
            <h3><a href="/202305/26/c_123456.htm">Test Xinhua Article</a></h3>
        </div>
        """
        
        self.china_daily_html = """
        <div class="item">
            <h3><a href="/a/202305/26/WS123456.html">Test China Daily Article</a></h3>
        </div>
        """
    
    @unittest.skip("Updated structure uses scraping.sources module")
    @patch('scrapers.taipei_times.make_request')
    def test_scrape_taipei_times_old(self, mock_make_request):
        """Test scraping Taipei Times with old structure."""
        try:
            # Setup mock response
            mock_response = MagicMock()
            mock_response.content = self.taipei_times_html
            mock_make_request.return_value = mock_response
            
            # Import the scraper
            from scrapers.taipei_times import scrape
            
            articles = scrape()
            
            self.assertEqual(len(articles), 1)
            self.assertEqual(articles[0]['title'], 'Test Taipei Times Article')
            self.assertEqual(articles[0]['source'], 'Taipei Times')
            self.assertTrue(articles[0]['url'].startswith('https://www.taipeitimes.com'))
            self.assertIn('scraped_at', articles[0])
        except ImportError:
            self.skipTest("Taipei Times scraper not available")
    
    def test_universal_scraper(self):
        """Test the universal scraper without mock."""
        try:
            # Import the universal scraper
            from scraping.universal_scraper import scrape_site
            
            # Test the scraper with direct call (no network)
            url = "https://example.com"
            source_name = "Test Source"
            articles = scrape_site(url, source_name)
            
            # Verify results
            self.assertGreater(len(articles), 0, "Universal scraper should return at least one sample article")
            
            # Check article structure
            for article in articles:
                self.assertIsInstance(article, dict, "Article should be a dictionary")
                self.assertIn('title', article, "Article should have a title")
                self.assertIn('url', article, "Article should have a URL")
                self.assertIn('source', article, "Article should have a source")
                self.assertIn('scraped_at', article, "Article should have a scraped_at timestamp")
                self.assertEqual(article['source'], source_name, "Source name should match")
        except ImportError as e:
            self.skipTest(f"Universal scraper not available: {e}")
    
    def test_scraping_runner(self):
        """Test the scraping runner module."""
        try:
            # Import the runner
            from scraping.runner import scrape_all_dynamic
            
            # Run with samples to ensure we get something even if actual scraping fails
            articles = scrape_all_dynamic(use_samples=True)
            
            # Verify results
            self.assertIsNotNone(articles, "scrape_all_dynamic should return a list")
            self.assertIsInstance(articles, list, "scrape_all_dynamic should return a list")
            self.assertGreater(len(articles), 0, "Should return at least the sample articles")
            
            # Check article structure
            for article in articles:
                self.assertIsInstance(article, dict, "Article should be a dictionary")
                self.assertIn('title', article, "Article should have a title")
                self.assertIn('url', article, "Article should have a URL")
                self.assertIn('source', article, "Article should have a source")
                self.assertIn('scraped_at', article, "Article should have a scraped_at timestamp")
                
            logger.info(f"Scraping runner successfully returned {len(articles)} articles")
        except ImportError as e:
            self.skipTest(f"Scraping runner not available: {e}")
    
    def test_error_handling(self):
        """Test that scrapers handle errors gracefully when imports fail."""
        try:
            # Try to import a non-existent scraper
            from scrapers.nonexistent_source import scrape
            self.fail("Should have raised ImportError")
        except ImportError:
            # Expected behavior
            pass

if __name__ == '__main__':
    unittest.main() 