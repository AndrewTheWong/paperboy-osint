#!/usr/bin/env python3
"""
Async Scraper Worker for Paperboy Backend
Uses the async scraper service as a Celery task
"""

import logging
from celery import Celery
from typing import List, Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('app.celery_config')

@celery_app.task(bind=True, max_retries=3)
def run_async_scraper(self, sources: Optional[List[Dict[str, Any]]] = None, max_articles_per_source: int = 10):
    """
    Run the async scraper as a Celery task
    
    Args:
        sources: List of source configurations (uses default if None)
        max_articles_per_source: Maximum articles to scrape per source
        
    Returns:
        Dict with scraping results
    """
    try:
        logger.info(f"🚀 Starting async scraper task with {len(sources) if sources else 'default'} sources")
        
        # Import the async scraper service
        from app.services.scraper_service import run_scraper, TAIWAN_STRAIT_SOURCES
        
        # Use default sources if none provided
        if sources is None:
            sources = TAIWAN_STRAIT_SOURCES
        
        # Run the async scraper
        import asyncio
        result = asyncio.run(run_scraper(sources, max_articles_per_source))
        
        logger.info(f"✅ Async scraper completed: {result['total_stored']} articles stored")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in async scraper task: {e}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def run_continuous_scraper(self):
    """
    Run the scraper continuously in a loop
    
    Returns:
        Dict with continuous scraping results
    """
    try:
        logger.info("🔄 Starting continuous async scraper")
        
        # Import the async scraper service
        from app.services.scraper_service import run_scraper, TAIWAN_STRAIT_SOURCES
        
        # Run the async scraper with default sources
        import asyncio
        result = asyncio.run(run_scraper(TAIWAN_STRAIT_SOURCES, max_articles_per_source=10))
        
        logger.info(f"✅ Continuous scraper cycle completed: {result['total_stored']} articles stored")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in continuous scraper task: {e}")
        raise self.retry(countdown=60, max_retries=3) 