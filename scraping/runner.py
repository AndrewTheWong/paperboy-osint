#!/usr/bin/env python3
"""
Dynamic scraper runner for Paperboy OSINT pipeline.
Loads source config and dynamically imports scrapers.
"""
import json
import logging
import importlib
import os
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scraping_runner')

# Define sample articles for testing when real scraping isn't possible
SAMPLE_ARTICLES = [
    {
        "title": "Taiwan conducts military exercises amid regional tensions",
        "url": "https://example.com/sample/1",
        "source": "Sample News",
        "scraped_at": datetime.utcnow().isoformat(),
        "language": "en"
    },
    {
        "title": "China announces new trade policies for cross-strait relations",
        "url": "https://example.com/sample/2",
        "source": "Sample News",
        "scraped_at": datetime.utcnow().isoformat(),
        "language": "en"
    },
    {
        "title": "US reaffirms commitment to Taiwan's defense",
        "url": "https://example.com/sample/3",
        "source": "Sample News",
        "scraped_at": datetime.utcnow().isoformat(),
        "language": "en"
    }
]

def load_config(config_path: Union[str, Path] = None) -> List[Dict[str, Any]]:
    """
    Load the source configuration from JSON file.
    
    Args:
        config_path: Path to the config file (defaults to scraping/config/sources_config.json)
        
    Returns:
        List of source configurations
    """
    if config_path is None:
        # Try both possible locations for config
        paths = [
            Path('scraping/config/sources_config.json'),
            Path('config/sources_config.json')
        ]
        
        for path in paths:
            if path.exists():
                config_path = path
                break
        
        if config_path is None:
            logger.error("Config file not found in expected locations")
            return []
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        logger.info(f"Loaded {len(config)} sources from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return []

def import_scraper(module_path: str) -> Optional[callable]:
    """
    Dynamically import a scraper module.
    
    Args:
        module_path: Dotted path to the scraper module (e.g., "scraping.sources.taipei_times.scrape")
        
    Returns:
        Scraper function or None if import failed
    """
    try:
        module_name, function_name = module_path.rsplit('.', 1)
        module = importlib.import_module(module_name)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import {module_path}: {str(e)}")
        return None

def scrape_all_dynamic(max_sources: int = None, use_samples: bool = False) -> List[Dict[str, Any]]:
    """
    Scrape articles from multiple sources dynamically based on config.
    
    Args:
        max_sources: Maximum number of sources to scrape (None for all)
        use_samples: Whether to include sample articles if real scraping fails
        
    Returns:
        List of article dictionaries
    """
    # Load source configurations
    config = load_config()
    
    if not config:
        logger.warning("No sources configured, using sample articles")
        return SAMPLE_ARTICLES
    
    # Limit number of sources if specified
    if max_sources is not None:
        config = config[:max_sources]
    
    # Collect all articles
    all_articles = []
    
    # Keep track of success/failure counts
    success_count = 0
    failure_count = 0
    
    # Try each source
    for source in config:
        source_name = source.get('name')
        source_url = source.get('url')
        scraper_path = source.get('function')
        language = source.get('language', 'en')
        
        logger.info(f"Trying to scrape {source_name} from {source_url}")
        
        try:
            # First, try the specified scraper
            articles = []
            if scraper_path:
                # Check if the path is in old format (scrapers.*) or new format (scraping.sources.*)
                if scraper_path.startswith('scrapers.'):
                    # Try to adapt to new structure
                    new_path = scraper_path.replace('scrapers.', 'scraping.sources.')
                    scraper_func = import_scraper(new_path)
                    if scraper_func:
                        articles = scraper_func()
                    else:
                        # Try the original path as fallback
                        scraper_func = import_scraper(scraper_path)
                        if scraper_func:
                            articles = scraper_func()
                else:
                    # Use path as-is
                    scraper_func = import_scraper(scraper_path)
                    if scraper_func:
                        articles = scraper_func()
            
            # If the specific scraper failed or no articles were returned, try the universal scraper
            if not articles:
                logger.info(f"Specific scraper failed for {source_name}, trying universal scraper")
                try:
                    from scraping.universal_scraper import scrape_site
                    articles = scrape_site(source_url, source_name, language)
                except ImportError:
                    logger.error("Failed to import universal_scraper module")
            
            # Add articles to collection
            if articles:
                logger.info(f"Successfully scraped {len(articles)} articles from {source_name}")
                all_articles.extend(articles)
                success_count += 1
            else:
                logger.warning(f"No articles scraped from {source_name}")
                failure_count += 1
        
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {str(e)}")
            failure_count += 1
    
    # Add sample articles if we didn't get any real ones and samples are requested
    if not all_articles and use_samples:
        logger.warning("No articles scraped, adding sample articles")
        all_articles = SAMPLE_ARTICLES
    
    # Log summary
    logger.info(f"Scraping complete: {success_count} sources successful, {failure_count} failed")
    logger.info(f"Total articles scraped: {len(all_articles)}")
    
    return all_articles

def save_articles_to_file(articles: List[Dict[str, Any]], filepath: Union[str, Path]) -> Path:
    """
    Save articles to a JSON file.
    
    Args:
        articles: List of article dictionaries
        filepath: Path to save the articles
        
    Returns:
        Path to the saved file
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(articles)} articles to {filepath}")
    return filepath

def load_articles_from_file(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        filepath: Path to the JSON file
        
    Returns:
        List of article dictionaries
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        logger.error(f"File not found: {filepath}")
        return []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        articles = json.load(f)
    
    logger.info(f"Loaded {len(articles)} articles from {filepath}")
    return articles

def generate_hash(url: str) -> str:
    """
    Generate a hash for an article URL to use as a cache key.
    
    Args:
        url: The article URL
        
    Returns:
        Hash string
    """
    import hashlib
    return hashlib.md5(url.encode('utf-8')).hexdigest()

def load_cache(cache_file: Union[str, Path] = None) -> Dict[str, Any]:
    """
    Load the article cache from file.
    
    Args:
        cache_file: Path to the cache file
        
    Returns:
        Dictionary of cached articles
    """
    if cache_file is None:
        cache_file = Path('data/article_cache.json')
    else:
        cache_file = Path(cache_file)
    
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, 'r', encoding='utf-8') as f:
            cache = json.load(f)
        return cache
    except Exception as e:
        logger.error(f"Error loading cache: {str(e)}")
        return {}

def save_cache(cache: Dict[str, Any], cache_file: Union[str, Path] = None) -> None:
    """
    Save the article cache to file.
    
    Args:
        cache: Dictionary of cached articles
        cache_file: Path to the cache file
    """
    if cache_file is None:
        cache_file = Path('data/article_cache.json')
    else:
        cache_file = Path(cache_file)
    
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(cache_file, 'w', encoding='utf-8') as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"Error saving cache: {str(e)}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run the dynamic scraper")
    parser.add_argument("--max-sources", type=int, default=None, 
                        help="Maximum number of sources to scrape")
    parser.add_argument("--output", default="data/articles.json",
                        help="Output file path for scraped articles")
    parser.add_argument("--samples", action="store_true",
                        help="Include sample articles if real scraping fails")
    
    args = parser.parse_args()
    
    # Run the scraper
    articles = scrape_all_dynamic(args.max_sources, args.samples)
    
    # Save results
    if articles:
        save_articles_to_file(articles, args.output)
        print(f"Saved {len(articles)} articles to {args.output}")
    else:
        print("No articles scraped") 