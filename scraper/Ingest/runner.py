#!/usr/bin/env python3
"""
News Ingestion Pipeline Runner
Complete pipeline for news scraping, processing, and storage.
"""

import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from .NewsScraper_new import EnhancedNewsScraper as NewsScraper, EnhancedScrapingConfig as ScrapingConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_ingestion_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NewsIngestionPipeline:
    """Complete news ingestion pipeline."""
    
    def __init__(self, config: ScrapingConfig = None):
        self.config = config or ScrapingConfig()
        self.scraper = NewsScraper(self.config)
        self.pipeline_stats = {
            'start_time': None,
            'end_time': None,
            'total_sources': 0,
            'successful_sources': 0,
            'total_articles_scraped': 0,
            'total_articles_saved': 0,
            'total_articles_uploaded': 0,
            'failed_sources': [],
            'errors': []
        }
    
    async def run_full_pipeline(self, 
                               categories: List[str] = None,
                               save_to_file: bool = True,
                               upload_to_supabase: bool = True) -> Dict[str, Any]:
        """Run the complete news ingestion pipeline."""
        logger.info("Starting news ingestion pipeline")
        self.pipeline_stats['start_time'] = datetime.now()
        
        try:
            # Step 1: Scrape articles from all sources
            logger.info("Step 1: Scraping articles from news sources")
            articles = await self.scrape_articles(categories)
            
            if not articles:
                logger.warning("No articles scraped. Pipeline ending early.")
                return self._generate_pipeline_report(articles)
            
            # Step 2: Process and clean articles
            logger.info("Step 2: Processing and cleaning articles")
            processed_articles = self.process_articles(articles)
            
            # Step 3: Save to file
            saved_file = None
            if save_to_file:
                logger.info("Step 3: Saving articles to file")
                saved_file = self.scraper.save_results(processed_articles)
                self.pipeline_stats['total_articles_saved'] = len(processed_articles)
            
            # Step 4: Upload to Supabase
            upload_stats = (0, 0, 0)
            if upload_to_supabase:
                logger.info("Step 4: Uploading articles to Supabase")
                upload_stats = await self.scraper.upload_to_supabase(processed_articles)
                self.pipeline_stats['total_articles_uploaded'] = sum(upload_stats[:2])
            
            # Step 5: Generate final report
            self.pipeline_stats['end_time'] = datetime.now()
            pipeline_report = self._generate_pipeline_report(processed_articles, saved_file, upload_stats)
            
            logger.info("News ingestion pipeline completed successfully")
            return pipeline_report
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.pipeline_stats['errors'].append(str(e))
            self.pipeline_stats['end_time'] = datetime.now()
            return self._generate_pipeline_report([], None, (0, 0, 0))
    
    async def scrape_articles(self, categories: List[str] = None) -> List[Dict[str, Any]]:
        """Scrape articles from configured news sources."""
        try:
            articles = await self.scraper.scrape_all_sources(categories)
            
            # Update pipeline stats
            self.pipeline_stats['total_sources'] = self.scraper.session_stats['total_sources']
            self.pipeline_stats['successful_sources'] = self.scraper.session_stats['successful_sources']
            self.pipeline_stats['total_articles_scraped'] = len(articles)
            self.pipeline_stats['failed_sources'] = self.scraper.session_stats['failed_sources']
            
            return articles
            
        except Exception as e:
            logger.error(f"Error during article scraping: {e}")
            self.pipeline_stats['errors'].append(f"Scraping error: {str(e)}")
            return []
    
    def process_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean scraped articles."""
        processed_articles = []
        
        for article in articles:
            try:
                # Clean and validate article data
                processed_article = self._clean_article(article)
                
                # Add additional metadata
                processed_article.update({
                    'processed_at': datetime.now().isoformat(),
                    'pipeline_version': '1.0',
                    'status': 'processed'
                })
                
                processed_articles.append(processed_article)
                
            except Exception as e:
                logger.warning(f"Error processing article {article.get('url', 'unknown')}: {e}")
                self.pipeline_stats['errors'].append(f"Processing error: {str(e)}")
        
        logger.info(f"Processed {len(processed_articles)} articles")
        return processed_articles
    
    def _clean_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and validate individual article data."""
        cleaned_article = article.copy()
        
        # Ensure required fields exist
        required_fields = ['url', 'source', 'title', 'content']
        for field in required_fields:
            if field not in cleaned_article:
                cleaned_article[field] = ''
        
        # Clean text fields
        for field in ['title', 'content']:
            if cleaned_article.get(field):
                cleaned_article[field] = self._clean_text(cleaned_article[field])
        
        # Validate URL
        if not cleaned_article['url'].startswith(('http://', 'https://')):
            cleaned_article['url'] = f"https://{cleaned_article['url']}"
        
        # Ensure dates are properly formatted
        if 'publish_date' not in cleaned_article or not cleaned_article['publish_date']:
            cleaned_article['publish_date'] = datetime.now().isoformat()
        
        # Add missing metadata fields
        if 'language' not in cleaned_article:
            cleaned_article['language'] = 'en'
        
        if 'region' not in cleaned_article:
            cleaned_article['region'] = 'Unknown'
        
        if 'category' not in cleaned_article:
            cleaned_article['category'] = 'news'
        
        return cleaned_article
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text content."""
        if not text:
            return ''
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove common web artifacts
        web_artifacts = [
            'Subscribe to continue reading',
            'Sign up for free newsletters',
            'This article requires a subscription',
            'Click here to subscribe',
            'Advertisement',
            'Loading...',
            'JavaScript is disabled'
        ]
        
        for artifact in web_artifacts:
            text = text.replace(artifact, '')
        
        # Limit length
        if len(text) > self.config.max_text_length:
            text = text[:self.config.max_text_length] + '...'
        
        return text.strip()
    
    def _generate_pipeline_report(self, 
                                 articles: List[Dict[str, Any]], 
                                 saved_file: Optional[str] = None,
                                 upload_stats: tuple = (0, 0, 0)) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution report."""
        duration = None
        if self.pipeline_stats['start_time'] and self.pipeline_stats['end_time']:
            duration = (self.pipeline_stats['end_time'] - self.pipeline_stats['start_time']).total_seconds()
        
        report = {
            'pipeline_execution': {
                'start_time': self.pipeline_stats['start_time'].isoformat() if self.pipeline_stats['start_time'] else None,
                'end_time': self.pipeline_stats['end_time'].isoformat() if self.pipeline_stats['end_time'] else None,
                'duration_seconds': duration,
                'status': 'completed' if not self.pipeline_stats['errors'] else 'completed_with_errors'
            },
            'scraping_results': {
                'total_sources_attempted': self.pipeline_stats['total_sources'],
                'successful_sources': self.pipeline_stats['successful_sources'],
                'failed_sources': len(self.pipeline_stats['failed_sources']),
                'failed_source_list': self.pipeline_stats['failed_sources'],
                'total_articles_scraped': len(articles)
            },
            'processing_results': {
                'articles_processed': len(articles),
                'processing_errors': len([e for e in self.pipeline_stats['errors'] if 'Processing error' in e])
            },
            'storage_results': {
                'saved_to_file': saved_file is not None,
                'file_path': saved_file,
                'uploaded_to_supabase': sum(upload_stats[:2]) > 0,
                'supabase_new': upload_stats[0],
                'supabase_updated': upload_stats[1],
                'supabase_errors': upload_stats[2]
            },
            'articles_by_source': self._generate_source_breakdown(articles),
            'articles_by_category': self._generate_category_breakdown(articles),
            'errors': self.pipeline_stats['errors']
        }
        
        return report
    
    def _generate_source_breakdown(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate breakdown of articles by source."""
        source_counts = {}
        for article in articles:
            source = article.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
        return source_counts
    
    def _generate_category_breakdown(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Generate breakdown of articles by category."""
        category_counts = {}
        for article in articles:
            category = article.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
        return category_counts
    
    def save_pipeline_report(self, report: Dict[str, Any]) -> str:
        """Save pipeline execution report to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"pipeline_report_{timestamp}.json"
        
        # Ensure reports directory exists
        Path("reports").mkdir(exist_ok=True)
        filepath = Path("reports") / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"Pipeline report saved to {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Could not save pipeline report: {e}")
            return ""

# Convenience functions for common pipeline configurations

async def run_news_ingestion_pipeline(
    categories: List[str] = None,
    max_workers: int = 10,
    articles_per_source: int = 20,
    enable_paywall_bypass: bool = True,
    save_to_file: bool = True,
    upload_to_supabase: bool = True
) -> Dict[str, Any]:
    """
    Run the complete news ingestion pipeline with specified configuration.
    
    Args:
        categories: List of source categories to scrape (None for all)
        max_workers: Maximum concurrent workers for scraping
        articles_per_source: Maximum articles to scrape per source
        enable_paywall_bypass: Whether to enable paywall bypass mechanisms
        save_to_file: Whether to save results to JSON file
        upload_to_supabase: Whether to upload results to Supabase
    
    Returns:
        Dict containing pipeline execution report
    """
    config = ScrapingConfig(
        max_workers=max_workers,
        articles_per_source=articles_per_source,
        enable_paywall_bypass=enable_paywall_bypass
    )
    
    pipeline = NewsIngestionPipeline(config)
    report = await pipeline.run_full_pipeline(
        categories=categories,
        save_to_file=save_to_file,
        upload_to_supabase=upload_to_supabase
    )
    
    # Save the pipeline report
    pipeline.save_pipeline_report(report)
    
    return report

async def run_quick_scraping(categories: List[str] = None) -> Dict[str, Any]:
    """Quick scraping with minimal configuration for testing."""
    return await run_news_ingestion_pipeline(
        categories=categories,
        max_workers=5,
        articles_per_source=10,
        enable_paywall_bypass=False,
        save_to_file=True,
        upload_to_supabase=False
    )

async def run_full_production_pipeline() -> Dict[str, Any]:
    """Full production pipeline with all features enabled."""
    return await run_news_ingestion_pipeline(
        categories=None,  # All categories
        max_workers=15,
        articles_per_source=50,
        enable_paywall_bypass=True,
        save_to_file=True,
        upload_to_supabase=True
    )

async def run_specific_categories(categories: List[str]) -> Dict[str, Any]:
    """Run pipeline for specific source categories only."""
    return await run_news_ingestion_pipeline(
        categories=categories,
        max_workers=10,
        articles_per_source=30,
        enable_paywall_bypass=True,
        save_to_file=True,
        upload_to_supabase=True
    )

# Main execution for testing
async def main():
    """Main execution function for testing the pipeline."""
    logger.info("Starting news ingestion pipeline test")
    
    # Run quick test
    report = await run_quick_scraping(['us_major_outlets'])
    
    logger.info("Pipeline test completed")
    logger.info(f"Scraped {report['scraping_results']['total_articles_scraped']} articles")
    
    return report

if __name__ == "__main__":
    asyncio.run(main()) 