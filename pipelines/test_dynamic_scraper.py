#!/usr/bin/env python3
"""
Test script for the dynamic scraper.
"""
import sys
import os
import logging
from pathlib import Path

# Add the project root to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('test_dynamic_scraper')

def test_imports():
    """Test that all necessary modules can be imported."""
    try:
        from utils.article_cache import load_cache, save_cache, generate_hash
        logger.info("Successfully imported article_cache functions")
        
        from pipelines.dynamic_scraper import scrape_all_dynamic
        logger.info("Successfully imported dynamic_scraper functions")
        
        # Test importing scraper modules
        from scrapers.taipei_times import scrape as scrape_taipei_times
        logger.info("Successfully imported taipei_times scraper")
        
        from scrapers.xinhua import scrape as scrape_xinhua
        logger.info("Successfully imported xinhua scraper")
        
        from scrapers.china_daily import scrape as scrape_china_daily
        logger.info("Successfully imported china_daily scraper")
        
        return True
    except ImportError as e:
        logger.error(f"Import error: {str(e)}")
        return False

def test_config_file():
    """Test that the config file exists and can be loaded."""
    try:
        import json
        config_path = Path('config/sources_config.json')
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
            logger.info(f"Successfully loaded config with {len(config)} sources")
            return True
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Config file error: {str(e)}")
        return False

def main():
    """Run all tests."""
    logger.info("Starting dynamic scraper tests")
    
    # Check paths
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Python path: {sys.path}")
    
    # Run tests
    imports_ok = test_imports()
    config_ok = test_config_file()
    
    # Print summary
    if imports_ok and config_ok:
        logger.info("All tests passed!")
        return 0
    else:
        logger.error("Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 