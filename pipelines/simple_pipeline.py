#!/usr/bin/env python3
"""
Simple Pipeline: Scraping -> Translation -> Tagging -> Supabase Storage
"""
import logging
import sys
import os
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('simple_pipeline')

def run_simple_pipeline():
    """Run a simple version of the pipeline."""
    logger.info("Starting Simple Pipeline")
    
    try:
        # Step 1: Scrape articles
        from pipelines.scrapers.runner import scrape_all_dynamic
        logger.info("Step 1: Scraping articles...")
        articles = scrape_all_dynamic(max_sources=2, use_samples=True)
        logger.info(f"Scraped {len(articles)} articles")
        
        # Step 2: Tag articles
        from pipelines.tagging.article_tagging import ArticleKeywordTagger
        logger.info("Step 2: Tagging articles...")
        tagger = ArticleKeywordTagger()
        
        for i, article in enumerate(articles):
            text = article.get('title', '') + ' ' + article.get('text', '')
            if text.strip():
                tags, _ = tagger.extract_tags_with_confidence(text)
                article['tags'] = list(tags.keys()) if tags else []
            else:
                article['tags'] = []
        
        logger.info("Step 3: Storing articles...")
        from pipelines.scrapers.supabase_storage import upload_articles_to_supabase
        uploaded = upload_articles_to_supabase(articles)
        logger.info(f"Uploaded {uploaded} articles to Supabase")
        
        logger.info("Pipeline completed successfully")
        return articles
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return []

if __name__ == "__main__":
    run_simple_pipeline()