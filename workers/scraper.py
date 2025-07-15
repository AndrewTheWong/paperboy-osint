#!/usr/bin/env python3
"""
Scraper Tasks for Paperboy Backend (Redesigned)
Scrapes articles using RobustAsyncScraper and writes directly to Supabase articles table.
"""

import logging
import asyncio
from datetime import datetime
from celery import shared_task, Celery
from services.async_scraper import RobustAsyncScraper
from db.supabase_client_v2 import store_raw_articles_batch

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('config.celery_config')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _run_async_with_proper_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@shared_task(bind=True)
def run_async_scraper(self, sources=None, max_articles_per_source=10):
    """
    Legacy async scraper task for compatibility.
    Now delegates to run_supabase_scraper.
    """
    logger.info("🔄 run_async_scraper called - delegating to run_supabase_scraper")
    return run_supabase_scraper.delay()

@shared_task(bind=True)
def run_continuous_scraper(self, sources=None, max_articles_per_source=10):
    """
    Legacy continuous scraper task for compatibility.
    Now delegates to run_supabase_scraper.
    """
    logger.info("🔄 run_continuous_scraper called - delegating to run_supabase_scraper")
    return run_supabase_scraper.delay()

@shared_task(bind=True)
def run_supabase_scraper(self, sources_path="sources/master_sources.json", max_sources=20, batch_size=100):
    """
    Scrape articles and write directly to Supabase articles table.
    Processes sources in batches of 100 for better performance.
    Args:
        sources_path: Path to sources file
        max_sources: Maximum number of sources to process (for testing)
        batch_size: Number of sources to process in each batch
    Returns:
        dict: Scraping results and performance metrics
    """
    try:
        logger.info(f"🚀 Starting Supabase scraper with sources: {sources_path}")
        scraper = RobustAsyncScraper()
        all_sources = RobustAsyncScraper.load_sources_from_file(sources_path)
        
        # Limit sources for testing if specified
        if max_sources and max_sources < len(all_sources):
            sources = all_sources[:max_sources]
            logger.info(f"📋 Using {len(sources)} sources (limited from {len(all_sources)} total)")
        else:
            sources = all_sources
            logger.info(f"📋 Loaded {len(sources)} sources")
        
        async def scrape_batch_async(batch_sources):
            """Scrape a batch of sources asynchronously"""
            return await scraper.scrape_sources(batch_sources)
        
        async def process_all_sources():
            start = datetime.now()
            all_articles = []
            total_batches = (len(sources) + batch_size - 1) // batch_size
            
            for i in range(0, len(sources), batch_size):
                batch_sources = sources[i:i + batch_size]
                batch_num = i // batch_size + 1
                logger.info(f"🔄 Processing batch {batch_num}/{total_batches}: {len(batch_sources)} sources")
                
                # Scrape batch asynchronously
                batch_articles = await scrape_batch_async(batch_sources)
                all_articles.extend(batch_articles)
                
                logger.info(f"✅ Batch {batch_num} completed: {len(batch_articles)} articles")
            
            scrape_time = (datetime.now() - start).total_seconds()
            logger.info(f"⏱️ Scraping {len(all_articles)} articles took {scrape_time:.2f}s")
            
            # Batch insert all articles
            t0 = datetime.now()
            stored = store_raw_articles_batch(all_articles, batch_size=100)
            t1 = datetime.now()
            supabase_time = (t1-t0).total_seconds()
            logger.info(f"⏱️ Supabase batch write: {supabase_time:.2f}s for {stored} articles")
            
            total_time = (datetime.now() - start).total_seconds()
            rate = stored / total_time if total_time > 0 else 0
            
            return {
                'status': 'success',
                'articles_scraped': len(all_articles),
                'articles_stored': stored,
                'duration_seconds': total_time,
                'articles_per_second': rate,
                'scrape_time': scrape_time,
                'supabase_batch_time': supabase_time,
                'sources_processed': len(sources),
                'batches_processed': total_batches
            }
        
        result = _run_async_with_proper_loop(process_all_sources())
        logger.info(f"✅ Scraped and stored {result['articles_stored']} articles in {result['duration_seconds']:.2f}s ({result['articles_per_second']:.2f}/sec)")
        if result['articles_per_second'] < 2.0:
            logger.warning(f"⚠️ Performance below target: {result['articles_per_second']:.2f} articles/sec (target: 2.0)")
        else:
            logger.info(f"🎯 Performance target met: {result['articles_per_second']:.2f} articles/sec")
        return result
    except Exception as e:
        logger.error(f"❌ Supabase scraper failed: {e}")
        return {'status': 'error', 'error': str(e)} 