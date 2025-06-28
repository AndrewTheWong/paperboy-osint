"""
News Ingestion Module
Unified scraper for all news sources with advanced paywall handling.
"""

from .NewsScraper_new import EnhancedNewsScraper as NewsScraper, EnhancedScrapingConfig as ScrapingConfig
from .runner import run_news_ingestion_pipeline

__all__ = ['NewsScraper', 'ScrapingConfig', 'run_news_ingestion_pipeline'] 