#!/usr/bin/env python3
"""
Scraper Tasks for Paperboy Backend
Uses the unified high-speed scraper for 2+ articles/second performance
"""

import logging
from celery import shared_task
from services.scraper import run_scraper, UnifiedHighSpeedScraper

logger = logging.getLogger(__name__)

@shared_task(bind=True)
def run_async_scraper(self, sources=None, max_articles_per_source=10):
    """
    Run the unified high-speed scraper asynchronously
    
    Args:
        sources: List of source dictionaries (optional, uses default if None)
        max_articles_per_source: Maximum articles to scrape per source
    
    Returns:
        dict: Scraping results with performance metrics
    """
    try:
        logger.info(f"🚀 Starting unified high-speed scraper task with {len(sources) if sources else 'default'} sources")
        
        # Run the scraper
        import asyncio
        result = asyncio.run(run_scraper(sources, max_articles_per_source))
        
        # Log performance metrics
        articles_per_second = result.get('articles_per_second', 0)
        total_queued = result.get('total_queued', 0)
        duration = result.get('duration_seconds', 0)
        
        logger.info(f"✅ Unified scraper completed: {total_queued} articles passed to preprocessing")
        logger.info(f"📊 Performance: {articles_per_second:.2f} articles/second in {duration:.1f}s")
        
        # Check if performance target was met
        if articles_per_second >= 2.0:
            logger.info(f"🎯 Performance target met: {articles_per_second:.2f} articles/second")
        else:
            logger.warning(f"⚠️ Performance below target: {articles_per_second:.2f} articles/second (target: 2.0)")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Scraper task failed: {e}")
        raise

@shared_task(bind=True)
def run_continuous_scraper(self, sources=None, max_articles_per_source=10, interval_minutes=30):
    """
    Run continuous scraping with periodic intervals
    
    Args:
        sources: List of source dictionaries
        max_articles_per_source: Maximum articles per source
        interval_minutes: Minutes between scraping runs
    
    Returns:
        dict: Results from the latest scraping run
    """
    try:
        logger.info(f"🔄 Starting continuous scraper (interval: {interval_minutes} minutes)")
        
        # Run one scraping cycle
        import asyncio
        result = asyncio.run(run_scraper(sources, max_articles_per_source))
        
        # Log continuous scraping metrics
        articles_per_second = result.get('articles_per_second', 0)
        total_queued = result.get('total_queued', 0)
        
        logger.info(f"✅ Continuous scraper cycle completed: {total_queued} articles passed to preprocessing")
        logger.info(f"📊 Cycle performance: {articles_per_second:.2f} articles/second")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Continuous scraper failed: {e}")
        raise

@shared_task(bind=True)
def run_stress_test_scraper(self, sources=None, max_articles_per_source=50):
    """
    Run stress test with high article count to test performance
    
    Args:
        sources: List of source dictionaries
        max_articles_per_source: High article count for stress testing
    
    Returns:
        dict: Stress test results with detailed performance metrics
    """
    try:
        logger.info(f"🔥 Starting stress test scraper with {max_articles_per_source} articles per source")
        
        # Run stress test
        import asyncio
        result = asyncio.run(run_scraper(sources, max_articles_per_source))
        
        # Detailed performance analysis
        articles_per_second = result.get('articles_per_second', 0)
        total_scraped = result.get('total_scraped', 0)
        total_queued = result.get('total_queued', 0)
        request_count = result.get('request_count', 0)
        duration = result.get('duration_seconds', 0)
        
        logger.info(f"🔥 Stress test completed:")
        logger.info(f"   📊 Articles/second: {articles_per_second:.2f}")
        logger.info(f"   📈 Total scraped: {total_scraped}")
        logger.info(f"   📦 Total passed to preprocessing: {total_queued}")
        logger.info(f"   🌐 HTTP requests: {request_count}")
        logger.info(f"   ⏱️ Duration: {duration:.1f}s")
        
        # Performance assessment
        if articles_per_second >= 5.0:
            logger.info("🚀 Excellent performance! System can handle high load")
        elif articles_per_second >= 2.0:
            logger.info("✅ Good performance! Meets target requirements")
        else:
            logger.warning("⚠️ Performance below target - consider optimization")
        
        return result
        
    except Exception as e:
        logger.error(f"❌ Stress test failed: {e}")
        raise 