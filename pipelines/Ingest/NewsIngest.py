#!/usr/bin/env python3
"""
Comprehensive News Ingestion Pipeline
Fast parallel processing with scraping, translation, tagging, geotagging, and embeddings.
Optimized for Taiwan Strait monitoring with 1000+ sources.
"""

import asyncio
import aiohttp
import json
import logging
import multiprocessing
import random
import time
import concurrent.futures
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse, quote
import re
import os
import sys

# Core dependencies
try:
    from bs4 import BeautifulSoup
    HAS_BS4 = True
except ImportError:
    HAS_BS4 = False

try:
    from supabase import create_client, Client
    HAS_SUPABASE = True
except ImportError:
    HAS_SUPABASE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_ingest.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('NewsIngest')

@dataclass
class IngestConfig:
    """Configuration for news ingestion pipeline."""
    max_workers: int = 16
    max_retries: int = 2
    timeout: int = 15
    articles_per_source: int = 10
    enable_translation: bool = True
    enable_tagging: bool = True
    enable_geotagging: bool = True
    enable_embedding: bool = True
    use_parallel_processing: bool = True
    batch_size: int = 50
    max_content_length: int = 15000
    headless_chrome: bool = True

class TaiwanStraitSourceManager:
    """Manages expanded Taiwan Strait monitoring sources."""
    
    def __init__(self):
        self.taiwan_strait_sources = self._create_comprehensive_sources()
    
    def _create_comprehensive_sources(self) -> Dict[str, Dict[str, Any]]:
        """Create comprehensive Taiwan Strait monitoring sources."""
        return {
            # Taiwan Official Sources
            'taiwan_presidential_office': {
                'name': 'Taiwan Presidential Office',
                'urls': ['https://english.president.gov.tw/News', 'https://english.president.gov.tw/'],
                'priority': 'high',
                'region': 'Taiwan',
                'language': 'en'
            },
            'taiwan_mnd': {
                'name': 'Taiwan MND',
                'urls': ['https://www.mnd.gov.tw/english/PublishTable.aspx', 'https://www.mnd.gov.tw/english/'],
                'priority': 'high',
                'region': 'Taiwan',
                'language': 'en'
            },
            'taiwan_mofa': {
                'name': 'Taiwan MOFA',
                'urls': ['https://en.mofa.gov.tw/News.aspx', 'https://en.mofa.gov.tw/'],
                'priority': 'high',
                'region': 'Taiwan',
                'language': 'en'
            },
            # Taiwan Media
            'taipei_times': {
                'name': 'Taipei Times',
                'urls': ['http://www.taipeitimes.com/News/front', 'http://www.taipeitimes.com/News/taiwan'],
                'priority': 'high',
                'region': 'Taiwan',
                'language': 'en'
            },
            'focus_taiwan': {
                'name': 'Focus Taiwan',
                'urls': ['https://focustaiwan.tw/politics', 'https://focustaiwan.tw/cross-strait', 'https://focustaiwan.tw/'],
                'priority': 'high',
                'region': 'Taiwan',
                'language': 'en'
            },
            'taiwan_news': {
                'name': 'Taiwan News',
                'urls': ['https://www.taiwannews.com.tw/en/news/politics', 'https://www.taiwannews.com.tw/en/news/defense'],
                'priority': 'high',
                'region': 'Taiwan',
                'language': 'en'
            },
            # Chinese Official Sources
            'china_mofa': {
                'name': 'China MOFA',
                'urls': ['https://www.fmprc.gov.cn/eng/xwfw_665399/', 'https://www.fmprc.gov.cn/eng/'],
                'priority': 'high',
                'region': 'China',
                'language': 'en'
            },
            'xinhua_english': {
                'name': 'Xinhua English',
                'urls': ['http://www.xinhuanet.com/english/', 'http://www.xinhuanet.com/english/asiapacific/'],
                'priority': 'high',
                'region': 'China',
                'language': 'en'
            },
            'china_daily': {
                'name': 'China Daily',
                'urls': ['https://www.chinadaily.com.cn/china', 'https://www.chinadaily.com.cn/world'],
                'priority': 'high',
                'region': 'China',
                'language': 'en'
            },
            # US Defense & Intelligence
            'pentagon': {
                'name': 'Pentagon',
                'urls': ['https://www.defense.gov/News/', 'https://www.defense.gov/News/Releases/'],
                'priority': 'high',
                'region': 'US',
                'language': 'en'
            },
            'state_dept': {
                'name': 'US State Department',
                'urls': ['https://www.state.gov/briefings/', 'https://www.state.gov/'],
                'priority': 'high',
                'region': 'US',
                'language': 'en'
            },
            # US Media
            'reuters_asia': {
                'name': 'Reuters Asia',
                'urls': ['https://www.reuters.com/world/asia-pacific/', 'https://www.reuters.com/world/china/'],
                'priority': 'high',
                'region': 'US',
                'language': 'en'
            },
            'bloomberg_asia': {
                'name': 'Bloomberg Asia',
                'urls': ['https://www.bloomberg.com/asia', 'https://www.bloomberg.com/news/world'],
                'priority': 'high',
                'region': 'US',
                'language': 'en'
            },
            'wsj_asia': {
                'name': 'WSJ Asia',
                'urls': ['https://www.wsj.com/news/world/asia', 'https://www.wsj.com/news/world'],
                'priority': 'high',
                'region': 'US',
                'language': 'en'
            },
            'nyt_asia': {
                'name': 'NYT Asia',
                'urls': ['https://www.nytimes.com/section/world/asia', 'https://www.nytimes.com/section/world'],
                'priority': 'high',
                'region': 'US',
                'language': 'en'
            },
            'cnn_asia': {
                'name': 'CNN Asia',
                'urls': ['https://www.cnn.com/world/asia', 'https://www.cnn.com/world'],
                'priority': 'medium',
                'region': 'US',
                'language': 'en'
            },
            # Japanese Sources
            'nikkei_asia': {
                'name': 'Nikkei Asia',
                'urls': ['https://asia.nikkei.com/Politics', 'https://asia.nikkei.com/Spotlight/Taiwan'],
                'priority': 'high',
                'region': 'Japan',
                'language': 'en'
            },
            'japan_times': {
                'name': 'Japan Times',
                'urls': ['https://www.japantimes.co.jp/news/world/', 'https://www.japantimes.co.jp/news/'],
                'priority': 'medium',
                'region': 'Japan',
                'language': 'en'
            },
            # Regional Asian Sources
            'straits_times': {
                'name': 'Straits Times',
                'urls': ['https://www.straitstimes.com/asia', 'https://www.straitstimes.com/world'],
                'priority': 'high',
                'region': 'Singapore',
                'language': 'en'
            },
            'south_china_morning_post': {
                'name': 'SCMP',
                'urls': ['https://www.scmp.com/news/china', 'https://www.scmp.com/news/asia'],
                'priority': 'high',
                'region': 'Hong Kong',
                'language': 'en'
            },
            # Defense & Security Specialized
            'defense_news': {
                'name': 'Defense News',
                'urls': ['https://www.defensenews.com/asia-pacific/', 'https://www.defensenews.com/global/'],
                'priority': 'high',
                'region': 'US',
                'language': 'en'
            },
            'breaking_defense': {
                'name': 'Breaking Defense',
                'urls': ['https://breakingdefense.com/category/indo-pacific/', 'https://breakingdefense.com/'],
                'priority': 'medium',
                'region': 'US',
                'language': 'en'
            }
        }
    
    def get_all_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get all Taiwan Strait monitoring sources."""
        return self.taiwan_strait_sources
    
    def get_high_priority_sources(self) -> Dict[str, Dict[str, Any]]:
        """Get only high priority sources for fast scanning."""
        return {k: v for k, v in self.taiwan_strait_sources.items() if v.get('priority') == 'high'}

class FastContentExtractor:
    """Optimized content extraction with site-specific patterns."""
    
    def __init__(self):
        self.site_patterns = {
            'reuters.com': {
                'title': 'h1[data-testid="Heading"], h1.article-headline',
                'content': '[data-testid="ArticleBody"] p, .article-body p',
            },
            'bbc.com': {
                'title': 'h1[data-testid="headline"], h1',
                'content': '[data-component="text-block"] p, .story-body p',
            },
            'default': {
                'title': 'h1, .title, .headline',
                'content': 'article p, .content p, .article-body p, .story-body p',
            }
        }
    
    def extract_fast(self, html: str, url: str) -> Dict[str, str]:
        """Fast content extraction."""
        if not HAS_BS4:
            return {'title': '', 'content': '', 'date': ''}
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            domain = urlparse(url).netloc.lower()
            
            patterns = self.site_patterns.get('default')
            for site, site_patterns in self.site_patterns.items():
                if site in domain:
                    patterns = site_patterns
                    break
            
            # Extract title
            title = ''
            for selector in patterns['title'].split(','):
                elem = soup.select_one(selector.strip())
                if elem and elem.get_text(strip=True):
                    title = elem.get_text(strip=True)[:500]
                    break
            
            # Extract content
            content_parts = []
            for selector in patterns['content'].split(','):
                elements = soup.select(selector.strip())
                if elements:
                    content_parts = [p.get_text(strip=True) for p in elements[:15] if p.get_text(strip=True)]
                    break
            
            content = ' '.join(content_parts)[:5000] if content_parts else ''
            
            return {
                'title': title,
                'content': content,
                'date': ''
            }
        except Exception as e:
            logger.error(f"Content extraction failed for {url}: {e}")
            return {'title': '', 'content': '', 'date': ''}

class ParallelArticleProcessor:
    """Parallel processing engine for article pipeline."""
    
    def __init__(self, config: IngestConfig):
        self.config = config
        self.extractor = FastContentExtractor()
        
        # Initialize processors
        self.processors = {}
        if config.enable_embedding:
            try:
                sys.path.append(str(Path(__file__).parent / "NewsArticles"))
                from processors.embedding_processor import EmbeddingProcessor
                self.processors['embedder'] = EmbeddingProcessor()
            except Exception as e:
                logger.warning(f"Embedding processor not available: {e}")
        
        if config.enable_geotagging:
            try:
                from processors.geographic_tagger import GeographicTagger
                self.processors['geo_tagger'] = GeographicTagger()
            except Exception as e:
                logger.warning(f"Geographic tagger not available: {e}")
        
        if config.enable_tagging:
            try:
                from processors.article_tagger import UnifiedArticleTagger
                self.processors['tagger'] = UnifiedArticleTagger()
            except Exception as e:
                logger.warning(f"Article tagger not available: {e}")
        
        if config.enable_translation:
            try:
                from processors.translator import TranslationProcessor
                self.processors['translator'] = TranslationProcessor()
            except Exception as e:
                logger.warning(f"Translator not available: {e}")
    
    async def scrape_url(self, session: aiohttp.ClientSession, url: str, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape articles from a URL."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            }
            
            timeout = aiohttp.ClientTimeout(total=self.config.timeout)
            
            async with session.get(url, headers=headers, timeout=timeout) as response:
                if response.status != 200:
                    return []
                
                html = await response.text()
                articles = self.extract_articles_from_page(html, url, source_config)
                
                logger.info(f"Scraped {len(articles)} articles from {url}")
                return articles
                
        except Exception as e:
            logger.error(f"Error scraping {url}: {e}")
            return []
    
    def extract_articles_from_page(self, html: str, base_url: str, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract article links and basic info from a page."""
        if not HAS_BS4:
            return []
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            articles = []
            
            # Extract article URLs
            article_links = set()
            link_selectors = [
                'a[href*="/news/"]', 'a[href*="/article/"]', 'a[href*="/story/"]',
                'a[href*="/politics/"]', 'a[href*="/world/"]',
                'a.headline-link', 'a.story-link'
            ]
            
            for selector in link_selectors:
                links = soup.select(selector)
                for link in links[:20]:  # Limit per selector
                    href = link.get('href')
                    if href:
                        if href.startswith('/'):
                            href = urljoin(base_url, href)
                        if self.is_valid_article_url(href):
                            article_links.add(href)
            
            # Process each article link
            for article_url in list(article_links)[:self.config.articles_per_source]:
                try:
                    article_data = {
                        'url': article_url,
                        'source': source_config.get('name', 'Unknown'),
                        'source_region': source_config.get('region', 'Unknown'),
                        'source_language': source_config.get('language', 'en'),
                        'scraped_at': datetime.now().isoformat(),
                        'title': '',
                        'content': '',
                        'date': ''
                    }
                    articles.append(article_data)
                except Exception as e:
                    continue
            
            return articles
            
        except Exception as e:
            logger.error(f"Error extracting articles from {base_url}: {e}")
            return []
    
    def is_valid_article_url(self, url: str) -> bool:
        """Check if URL is a valid article URL."""
        if not url or len(url) < 10:
            return False
        
        skip_patterns = [
            '/tag/', '/category/', '/author/', '/search/',
            '.pdf', '.jpg', '.png', '.gif',
            'mailto:', 'javascript:',
            '/subscribe', '/login'
        ]
        
        url_lower = url.lower()
        for pattern in skip_patterns:
            if pattern in url_lower:
                return False
        
        return url.startswith(('http://', 'https://'))
    
    def process_article_content(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process individual article with all pipeline components."""
        try:
            # Skip if no content to process
            if not article.get('content') and not article.get('title'):
                return article
            
            text = f"{article.get('title', '')} {article.get('content', '')}"
            
            # Translation
            if self.config.enable_translation and 'translator' in self.processors:
                if article.get('source_language', 'en') != 'en':
                    try:
                        translated = self.processors['translator'].translate_text(text, target_lang='en')
                        article['translated_content'] = translated
                    except Exception as e:
                        logger.warning(f"Translation failed: {e}")
            
            # Tagging
            if self.config.enable_tagging and 'tagger' in self.processors:
                try:
                    tags = self.processors['tagger'].tag_article(
                        article.get('title', ''), 
                        article.get('content', '')
                    )
                    article.update({
                        'keywords': tags.get('keywords', []),
                        'entities': tags.get('entities', {}),
                        'escalation_score': tags.get('escalation_score', 0.0),
                        'sentiment_polarity': tags.get('sentiment', {}).get('polarity', 0.0)
                    })
                except Exception as e:
                    logger.warning(f"Tagging failed: {e}")
            
            # Geographic tagging
            if self.config.enable_geotagging and 'geo_tagger' in self.processors:
                try:
                    geo = self.processors['geo_tagger'].extract_geographic_info(text)
                    article.update({
                        'primary_country': geo.get('primary_country'),
                        'primary_region': geo.get('primary_region'),
                        'primary_city': geo.get('primary_city'),
                        'geographic_confidence': geo.get('confidence', 0.0),
                        'all_locations': geo.get('all_locations', []),
                        'geographic_tags': geo.get('geographic_tags', [])
                    })
                except Exception as e:
                    logger.warning(f"Geographic tagging failed: {e}")
            
            # Embeddings
            if self.config.enable_embedding and 'embedder' in self.processors:
                try:
                    title_embedding = self.processors['embedder'].generate_embedding(article.get('title', ''))
                    content_embedding = self.processors['embedder'].generate_embedding(article.get('content', ''))
                    
                    article.update({
                        'title_embedding': title_embedding,
                        'content_embedding': content_embedding,
                        'title_embedding_success': title_embedding is not None,
                        'content_embedding_success': content_embedding is not None
                    })
                except Exception as e:
                    logger.warning(f"Embedding failed: {e}")
            
            # Add processing metadata
            article.update({
                'word_count': len(text.split()),
                'processed_at': datetime.now().isoformat(),
                'processing_version': '2.0'
            })
            
            return article
            
        except Exception as e:
            logger.error(f"Article processing failed: {e}")
            return article

class SupabaseUploader:
    """Fast batch uploader for Supabase."""
    
    def __init__(self):
        self.supabase = self._init_supabase()
    
    def _init_supabase(self) -> Optional[Client]:
        """Initialize Supabase client."""
        if not HAS_SUPABASE:
            return None
        
        try:
            # Use local StraitWatch Supabase instance
            url = "http://127.0.0.1:54321"
            key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
            
            return create_client(url, key)
        except Exception as e:
            logger.error(f"Supabase initialization failed: {e}")
            return None
    
    def upload_batch(self, articles: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """Upload articles in batch."""
        if not self.supabase:
            return 0, 0, len(articles)
        
        uploaded = 0
        updated = 0
        errors = 0
        
        try:
            for i, article in enumerate(articles):
                try:
                    supabase_article = self._prepare_article_for_supabase(article)
                    
                    # Better validation - require URL and either title or content
                    url = supabase_article.get('url', '')
                    title = supabase_article.get('title', '')
                    content = supabase_article.get('content', '')
                    
                    if not url:
                        logger.warning(f"Skipping article without URL")
                        errors += 1
                        continue
                    
                    if not title and not content:
                        logger.warning(f"Skipping article without title or content: {url}")
                        errors += 1
                        continue
                    
                    # Check if exists
                    existing = self.supabase.table('articles').select('id').eq('url', url).execute()
                    
                    if existing.data:
                        # Only update if we have better content
                        if content and len(content) > 50:  # Only update if we have substantial content
                            self.supabase.table('articles').update(supabase_article).eq('url', url).execute()
                            updated += 1
                        else:
                            # Skip update if content is poor
                            continue
                    else:
                        # Insert new article
                        result = self.supabase.table('articles').insert(supabase_article).execute()
                        if result.data:
                            uploaded += 1
                        else:
                            errors += 1
                            logger.warning(f"Insert failed for {url}: No data returned")
                    
                    # Progress logging
                    if (i + 1) % 10 == 0:
                        logger.info(f"Processed {i + 1}/{len(articles)} articles (Uploaded: {uploaded}, Updated: {updated}, Errors: {errors})")
                        
                except Exception as e:
                    logger.warning(f"Failed to upload article {article.get('url', 'unknown')}: {e}")
                    errors += 1
        except Exception as e:
            logger.error(f"Batch upload failed: {e}")
            errors = len(articles)
        
        return uploaded, updated, errors
    
    def _prepare_article_for_supabase(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare article data for articles table schema."""
        # Clean data and ensure proper types
        content = article.get('content', '') or ''
        title = article.get('title', '') or ''
        
        return {
            'url': article.get('url', ''),
            'title': title,
            'content': content if content else None,
            'source': article.get('source', ''),
            'published_at': article.get('published_at') if article.get('published_at') else None,
            'language': article.get('source_language', 'en'),
            'relevant': None  # Will be determined by tagging pipeline
        }

class ComprehensiveNewsIngest:
    """Main news ingestion pipeline orchestrator."""
    
    def __init__(self, config: IngestConfig = None):
        self.config = config or IngestConfig()
        self.source_manager = TaiwanStraitSourceManager()
        self.processor = ParallelArticleProcessor(self.config)
        self.uploader = SupabaseUploader()
        self.stats = {
            'sources_processed': 0,
            'articles_scraped': 0,
            'articles_uploaded': 0,
            'errors': 0,
            'start_time': None,
            'end_time': None
        }
    
    async def run_comprehensive_ingest(self, source_filter: str = 'all', max_sources: int = None) -> Dict[str, Any]:
        """Run comprehensive news ingestion pipeline."""
        logger.info("Starting comprehensive news ingestion pipeline")
        self.stats['start_time'] = datetime.now()
        
        try:
            # Get sources
            if source_filter == 'high_priority':
                sources = self.source_manager.get_high_priority_sources()
            else:
                sources = self.source_manager.get_all_sources()
            
            if max_sources:
                sources = dict(list(sources.items())[:max_sources])
            
            logger.info(f"Processing {len(sources)} sources")
            
            # Scrape articles
            all_articles = await self.scrape_all_sources(sources)
            self.stats['articles_scraped'] = len(all_articles)
            
            if not all_articles:
                logger.warning("No articles scraped")
                return self._generate_report()
            
            # Process articles with all pipeline components
            if self.config.enable_translation or self.config.enable_tagging or self.config.enable_geotagging or self.config.enable_embedding:
                logger.info(f"Processing {len(all_articles)} articles...")
                processed_articles = await self.process_articles_parallel(all_articles)
            else:
                processed_articles = all_articles
            
            # Upload to Supabase
            upload_stats = self.uploader.upload_batch(processed_articles)
            self.stats['articles_uploaded'] = upload_stats[0] + upload_stats[1]
            
            self.stats['end_time'] = datetime.now()
            
            logger.info(f"Pipeline completed: {self.stats['articles_scraped']} scraped, {self.stats['articles_uploaded']} uploaded")
            
            return self._generate_report()
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.stats['errors'] += 1
            self.stats['end_time'] = datetime.now()
            return self._generate_report()
    
    async def scrape_all_sources(self, sources: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Scrape all sources in parallel."""
        all_articles = []
        
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def scrape_source_with_semaphore(source_key: str, source_config: Dict[str, Any]):
            async with semaphore:
                return await self.scrape_single_source(source_key, source_config)
        
        connector = aiohttp.TCPConnector(limit=100)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            tasks = [
                scrape_source_with_semaphore(key, config)
                for key, config in sources.items()
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    all_articles.extend(result)
                elif isinstance(result, Exception):
                    self.stats['errors'] += 1
        
        logger.info(f"Scraped {len(all_articles)} total articles from {len(sources)} sources")
        return all_articles
    
    async def scrape_single_source(self, source_key: str, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape a single source."""
        try:
            articles = []
            urls = source_config.get('urls', [])
            
            async with aiohttp.ClientSession() as session:
                for url in urls:
                    try:
                        source_articles = await self.processor.scrape_url(session, url, source_config)
                        
                        # Get full content for each article
                        for article in source_articles:
                            content_data = await self.get_article_content(session, article['url'])
                            article.update(content_data)
                        
                        articles.extend(source_articles)
                        await asyncio.sleep(random.uniform(0.5, 1.5))
                        
                    except Exception as e:
                        continue
            
            self.stats['sources_processed'] += 1
            logger.info(f"Scraped {len(articles)} articles from {source_key}")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping source {source_key}: {e}")
            return []
    
    async def get_article_content(self, session: aiohttp.ClientSession, url: str) -> Dict[str, str]:
        """Get full content for an article with better extraction."""
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15), headers=headers) as response:
                if response.status == 200:
                    html = await response.text()
                    content_data = self.processor.extractor.extract_fast(html, url)
                    
                    # Validate extracted content
                    if not content_data.get('title') or len(content_data.get('title', '')) < 5:
                        # Try alternative extraction
                        content_data = self._extract_content_fallback(html, url)
                    
                    return content_data
        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {e}")
        
        return {'title': '', 'content': '', 'date': ''}
    
    def _extract_content_fallback(self, html: str, url: str) -> Dict[str, str]:
        """Fallback content extraction method."""
        if not HAS_BS4:
            return {'title': '', 'content': '', 'date': ''}
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # Try to extract title
            title = ''
            title_selectors = [
                'h1', 'title', '.headline', '.article-title', 
                '[data-testid="Heading"]', '.entry-title'
            ]
            
            for selector in title_selectors:
                elem = soup.select_one(selector)
                if elem and elem.get_text(strip=True):
                    title = elem.get_text(strip=True)[:300]
                    break
            
            # Try to extract content 
            content = ''
            content_selectors = [
                '.article-body', '.entry-content', '.post-content',
                '[data-testid="ArticleBody"]', '.story-body',
                'article', '.content', '.main-content'
            ]
            
            for selector in content_selectors:
                elem = soup.select_one(selector)
                if elem:
                    # Get all paragraph text
                    paragraphs = elem.find_all(['p', 'div'], recursive=True)
                    content_parts = []
                    for p in paragraphs:
                        text = p.get_text(strip=True)
                        if text and len(text) > 20:  # Filter out short text
                            content_parts.append(text)
                    
                    if content_parts:
                        content = ' '.join(content_parts[:10])  # Limit to first 10 paragraphs
                        break
            
            # If still no content, try getting all paragraph text
            if not content:
                paragraphs = soup.find_all('p')
                content_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and len(text) > 30:
                        content_parts.append(text)
                
                if content_parts:
                    content = ' '.join(content_parts[:8])
            
            return {
                'title': title,
                'content': content[:3000] if content else '',  # Limit content length
                'date': ''
            }
            
        except Exception as e:
            logger.error(f"Fallback extraction failed for {url}: {e}")
            return {'title': '', 'content': '', 'date': ''}
    
    async def process_articles_parallel(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process articles in parallel using thread pool."""
        if not articles:
            return []
        
        logger.info(f"Processing {len(articles)} articles in parallel")
        
        # Use thread pool for CPU-bound processing
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all processing tasks
            future_to_article = {
                executor.submit(self.processor.process_article_content, article): article
                for article in articles
            }
            
            processed_articles = []
            for future in concurrent.futures.as_completed(future_to_article):
                try:
                    processed_article = future.result(timeout=30)
                    processed_articles.append(processed_article)
                except Exception as e:
                    original_article = future_to_article[future]
                    logger.error(f"Processing failed for {original_article.get('url', 'unknown')}: {e}")
                    processed_articles.append(original_article)  # Keep original
        
        logger.info(f"Processed {len(processed_articles)} articles")
        return processed_articles

    def _generate_report(self) -> Dict[str, Any]:
        """Generate pipeline execution report."""
        duration = None
        if self.stats['start_time'] and self.stats['end_time']:
            duration = (self.stats['end_time'] - self.stats['start_time']).total_seconds()
        
        return {
            'pipeline_stats': self.stats,
            'duration_seconds': duration,
            'success_rate': (self.stats['articles_uploaded'] / max(self.stats['articles_scraped'], 1)) * 100,
            'timestamp': datetime.now().isoformat()
        }

# Test function for 1000 sources
async def run_test_1000_sources():
    """Test with 1000 sources by duplicating and cycling through regions."""
    logger.info("Starting 1000-source test")
    
    config = IngestConfig(
        max_workers=25,
        articles_per_source=2,  # Reduced for speed
        timeout=8
    )
    
    # Generate 1000 test sources by cycling through existing ones
    source_manager = TaiwanStraitSourceManager()
    base_sources = source_manager.get_all_sources()
    
    test_sources = {}
    regions = ['Taiwan', 'China', 'US', 'Japan', 'Singapore']
    
    source_items = list(base_sources.items())
    for i in range(1000):
        base_key, base_config = source_items[i % len(source_items)]
        region = regions[i % len(regions)]
        
        test_key = f"{base_key}_{region.lower()}_{i}"
        test_config = base_config.copy()
        test_config['name'] = f"{test_config['name']} {region} {i}"
        test_config['region'] = region
        
        test_sources[test_key] = test_config
    
    logger.info(f"Generated {len(test_sources)} test sources")
    
    pipeline = ComprehensiveNewsIngest(config)
    
    # Override source manager for test
    class TestSourceManager:
        def get_all_sources(self):
            return test_sources
    
    pipeline.source_manager = TestSourceManager()
    
    start_time = time.time()
    result = await pipeline.run_comprehensive_ingest('all')
    duration = time.time() - start_time
    
    logger.info(f"1000-source test completed in {duration:.1f} seconds")
    logger.info(f"Sources processed: {result['pipeline_stats']['sources_processed']}")
    logger.info(f"Articles scraped: {result['pipeline_stats']['articles_scraped']}")
    logger.info(f"Articles uploaded: {result['pipeline_stats']['articles_uploaded']}")
    
    return result

# Entry point functions
async def run_fast_ingest(max_sources: int = 50) -> Dict[str, Any]:
    """Run fast ingest with high-priority sources."""
    config = IngestConfig(
        max_workers=20,
        articles_per_source=5,
        timeout=10
    )
    
    pipeline = ComprehensiveNewsIngest(config)
    return await pipeline.run_comprehensive_ingest('high_priority', max_sources)

async def run_comprehensive_ingest(max_sources: int = None) -> Dict[str, Any]:
    """Run comprehensive ingest with all sources."""
    config = IngestConfig(
        max_workers=16,
        articles_per_source=10,
        timeout=15
    )
    
    pipeline = ComprehensiveNewsIngest(config)
    return await pipeline.run_comprehensive_ingest('all', max_sources)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive News Ingestion Pipeline')
    parser.add_argument('--mode', choices=['fast', 'comprehensive', 'test1000'], default='fast',
                       help='Ingestion mode')
    parser.add_argument('--max-sources', type=int, help='Maximum sources to process')
    
    args = parser.parse_args()
    
    if args.mode == 'fast':
        result = asyncio.run(run_fast_ingest(args.max_sources or 30))
    elif args.mode == 'comprehensive':
        result = asyncio.run(run_comprehensive_ingest(args.max_sources))
    elif args.mode == 'test1000':
        result = asyncio.run(run_test_1000_sources())
    
    print(f"\nPipeline Results:")
    print(f"Duration: {result.get('duration_seconds', 0):.1f} seconds")
    print(f"Success Rate: {result.get('success_rate', 0):.1f}%") 