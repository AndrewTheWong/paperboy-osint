#!/usr/bin/env python3
"""
Main script to run the dynamic scraper.
"""
import json
import importlib
import time
import random
import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('scraper_runner')

# Define paths
CACHE_PATH = Path("data/article_cache.json")

def load_cache():
    """Load the article hash cache from disk."""
    if CACHE_PATH.exists():
        with open(CACHE_PATH, "r", encoding="utf-8") as f:
            return set(json.load(f))
    return set()

def save_cache(cache_set):
    """Save the set of article hashes to disk."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(list(cache_set), f, ensure_ascii=False, indent=2)

def generate_hash(article):
    """Generate a unique hash for an article based on its URL and title."""
    import hashlib
    key = f"{article['url']}-{article['title']}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()

def load_config():
    """Load the scraper configuration."""
    config_path = Path('config/sources_config.json')
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Failed to load config file: {str(e)}")
        return []

def import_scraper_function(function_path):
    """Dynamically import a scraper function."""
    try:
        module_path, function_name = function_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        return getattr(module, function_name)
    except (ImportError, AttributeError) as e:
        logger.error(f"Failed to import {function_path}: {str(e)}")
        return None

def scrape_all_dynamic():
    """Load config, call each scraper module, and return deduplicated articles."""
    logger.info("Starting dynamic scraper")
    start_time = time.time()
    
    # Load the article cache for deduplication
    article_cache = load_cache()
    logger.info(f"Loaded cache with {len(article_cache)} existing article hashes")
    
    # Load the scraper configurations
    scraper_configs = load_config()
    if not scraper_configs:
        logger.error("No scraper configurations found")
        return []
    
    logger.info(f"Loaded {len(scraper_configs)} scraper configurations")
    
    # Track new articles and hashes
    all_new_articles = []
    new_hashes = set()
    stats = defaultdict(int)
    
    # Process each scraper
    for config in scraper_configs:
        source_name = config.get('name', 'Unknown')
        function_path = config.get('function')
        
        if not function_path:
            logger.warning(f"Missing function path for {source_name}, skipping")
            continue
        
        logger.info(f"Processing {source_name} using {function_path}")
        
        # Import the scraper function
        scraper_function = import_scraper_function(function_path)
        if not scraper_function:
            logger.error(f"Failed to import scraper for {source_name}, skipping")
            continue
        
        try:
            # Call the scraper function
            articles = scraper_function()
            
            if not articles:
                logger.warning(f"No articles returned from {source_name}")
                continue
                
            logger.info(f"Retrieved {len(articles)} articles from {source_name}")
            
            # Process each article for deduplication
            new_source_articles = []
            for article in articles:
                # Generate hash for deduplication
                article_hash = generate_hash(article)
                
                # Skip if already in cache (seen before)
                if article_hash in article_cache:
                    stats['skipped'] += 1
                    continue
                    
                # Add to new articles and update cache
                new_source_articles.append(article)
                new_hashes.add(article_hash)
                stats['new'] += 1
            
            # Add new articles to the combined list
            all_new_articles.extend(new_source_articles)
            logger.info(f"Added {len(new_source_articles)} new articles from {source_name}")
            
            # Statistics for this source
            stats[source_name] = len(new_source_articles)
            
        except Exception as e:
            logger.error(f"Error processing {source_name}: {str(e)}")
        
        # Add a delay between requests to different sources
        time.sleep(random.uniform(1.0, 3.0))
    
    # Update the cache with new hashes
    article_cache.update(new_hashes)
    save_cache(article_cache)
    logger.info(f"Updated cache with {len(new_hashes)} new article hashes")
    
    # Report statistics
    runtime = time.time() - start_time
    logger.info(f"Dynamic scraper completed in {runtime:.2f} seconds")
    logger.info(f"Articles: {stats['new']} new, {stats['skipped']} duplicates skipped")
    
    return all_new_articles

def save_articles_to_file(articles, filename=None):
    """Save the scraped articles to a JSON file."""
    if not filename:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"data/articles_{timestamp}.json"
    
    file_path = Path(filename)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved {len(articles)} articles to {file_path}")
    return file_path

if __name__ == "__main__":
    # Run the dynamic scraper
    articles = scrape_all_dynamic()
    
    # Group articles by source for reporting
    sources = defaultdict(list)
    for article in articles:
        sources[article['source']].append(article)
    
    # Print summary
    print(f"\nTotal articles scraped: {len(articles)}")
    print("\nArticles by source:")
    for source, source_articles in sources.items():
        print(f"- {source}: {len(source_articles)} articles")
    
    # Print sample articles (first from each source)
    print("\nSample articles:")
    seen_sources = set()
    for article in articles:
        source = article['source']
        if source not in seen_sources:
            seen_sources.add(source)
            print(f"\n{article['source']} - {article['title'][:100]}...")
            print(f"URL: {article['url']}")
    
    # Save to file
    if articles:
        output_file = save_articles_to_file(articles)
        print(f"\nFull results saved to {output_file}")
    else:
        print("\nNo articles were scraped.") 