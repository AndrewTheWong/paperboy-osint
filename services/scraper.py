#!/usr/bin/env python3
"""
Unified High-Speed Scraper Service for Paperboy Backend
Combines robust async scraping with pipeline integration
Target: 2+ articles/second with multi-source support
"""

import asyncio
import aiohttp
import logging
import json
import random
import time
import uuid
import os
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
from readability import Document
from langdetect import detect, LangDetectException
import trafilatura
from urllib.parse import urljoin, urlparse
import re
from datetime import datetime
import redis
import aiofiles
from concurrent.futures import ThreadPoolExecutor
import threading


from db.redis_queue import get_redis_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15A372 Safari/604.1",
    "Mozilla/5.0 (Linux; Android 10; SM-G975F) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.120 Mobile Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
]

class UnifiedHighSpeedScraper:
    """Unified high-speed scraper with robust features and pipeline integration"""
    
    def __init__(self, max_concurrent: int = 50, rate_limit: float = 0.1):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit  # seconds between requests

        self.redis_client = get_redis_client()
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_count = 0
        self.start_time = None
        
        # Load sources from config
        self.sources = self._load_sources()
        
    def _load_sources(self) -> List[Dict[str, Any]]:
        """Load sources from JSON config file"""
        try:
            config_path = "sources/taiwan_strait_sources.json"
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                # Fallback to default sources
                return [
                    {"name": "Focus Taiwan", "url": "https://focustaiwan.tw/politics", "type": "news"},
                    {"name": "Taipei Times", "url": "http://www.taipeitimes.com/News/front", "type": "news"},
                    {"name": "China Daily", "url": "https://www.chinadaily.com.cn/world", "type": "news"},
                    {"name": "South China Morning Post", "url": "https://www.scmp.com/news/asia", "type": "news"},
                    {"name": "Reuters Asia", "url": "https://www.reuters.com/world/asia-pacific/", "type": "news"},
                    {"name": "Bloomberg Asia", "url": "https://www.bloomberg.com/asia", "type": "news"},
                    {"name": "CNN Asia", "url": "https://www.cnn.com/world/asia", "type": "news"},
                    {"name": "NYT Asia", "url": "https://www.nytimes.com/section/world/asia", "type": "news"}
                ]
        except Exception as e:
            logger.error(f"Failed to load sources: {e}")
            return []
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session with proper configuration"""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=30, connect=10)
            connector = aiohttp.TCPConnector(limit=self.max_concurrent, limit_per_host=10)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                connector=connector,
                headers={'User-Agent': random.choice(USER_AGENTS)}
            )
        return self.session
    
    async def _fetch_url(self, url: str, source_name: str) -> Optional[str]:
        """Fetch URL with rate limiting and error handling"""
        async with self.semaphore:
            try:
                session = await self._get_session()
                
                # Rate limiting
                if self.rate_limit > 0:
                    await asyncio.sleep(self.rate_limit)
                
                # Rotate user agent
                headers = {'User-Agent': random.choice(USER_AGENTS)}
                
                async with session.get(url, headers=headers, allow_redirects=True) as response:
                    if response.status == 200:
                        content = await response.text()
                        self.request_count += 1
                        logger.debug(f"✅ Fetched {url} ({len(content)} bytes)")
                        return content
                    else:
                        logger.warning(f"Failed to fetch {url}: {response.status}")
                        return None
                        
            except Exception as e:
                logger.warning(f"Error fetching {url}: {e}")
                return None
    
    def _extract_metadata_robust(self, html: str, url: str) -> Dict[str, Any]:
        """Robust metadata extraction using multiple methods"""
        metadata = {
            'title': None,
            'description': None,
            'author': None,
            'published_date': None,
            'language': 'en'
        }
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Method 1: Trafilatura (if available)
            try:
                # Use the correct trafilatura API
                meta = trafilatura.extract_metadata(html, url=url)
                if meta and isinstance(meta, dict):
                    metadata['title'] = meta.get('title')
                    metadata['description'] = meta.get('description')
                    metadata['author'] = meta.get('author')
                    metadata['published_date'] = meta.get('date')
            except Exception as e:
                logger.debug(f"trafilatura metadata failed: {e}")
            
            # Method 2: BeautifulSoup fallback
            if not metadata['title']:
                # Try multiple title selectors
                title_selectors = [
                    'meta[property="og:title"]',
                    'meta[name="twitter:title"]',
                    'meta[name="title"]',
                    'h1',
                    'title'
                ]
                
                for selector in title_selectors:
                    element = soup.select_one(selector)
                    if element:
                        metadata['title'] = element.get('content') or element.get_text().strip()
                        if metadata['title']:
                            break
            
            # Extract description
            if not metadata['description']:
                desc_selectors = [
                    'meta[name="description"]',
                    'meta[property="og:description"]',
                    'meta[name="twitter:description"]'
                ]
                
                for selector in desc_selectors:
                    element = soup.select_one(selector)
                    if element:
                        metadata['description'] = element.get('content', '').strip()
                        if metadata['description']:
                            break
            
            # Language detection
            try:
                text_sample = soup.get_text()[:1000]
                if text_sample:
                    detected_lang = detect(text_sample)
                    metadata['language'] = detected_lang
            except LangDetectException:
                pass
                
        except Exception as e:
            logger.warning(f"Metadata extraction failed: {e}")
        
        return metadata
    
    def _extract_content_robust(self, html: str, url: str) -> Optional[str]:
        """Robust content extraction using multiple methods"""
        content = None
        
        # Method 1: Trafilatura
        try:
            extracted = trafilatura.extract(html, include_formatting=True, include_links=True)
            if extracted and len(extracted.strip()) > 100:
                content = extracted
        except Exception as e:
            logger.debug(f"trafilatura content extraction failed: {e}")
        
        # Method 2: Readability
        if not content:
            try:
                doc = Document(html)
                content = doc.summary()
                if content and len(content.strip()) > 100:
                    content = content
            except Exception as e:
                logger.debug(f"readability extraction failed: {e}")
        
        # Method 3: BeautifulSoup fallback
        if not content:
            try:
                soup = BeautifulSoup(html, 'html.parser')
                
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                
                # Try to find main content areas
                content_selectors = [
                    'article',
                    '[role="main"]',
                    '.content',
                    '.article-content',
                    '.post-content',
                    'main'
                ]
                
                for selector in content_selectors:
                    element = soup.select_one(selector)
                    if element:
                        content = element.get_text(separator='\n', strip=True)
                        if len(content) > 100:
                            break
                
                # Fallback to body text
                if not content or len(content) < 100:
                    content = soup.get_text(separator='\n', strip=True)
                    
            except Exception as e:
                logger.debug(f"BeautifulSoup extraction failed: {e}")
        
        return content
    
    async def _scrape_article(self, url: str, source_name: str) -> Optional[Dict[str, Any]]:
        """Scrape a single article with robust extraction"""
        try:
            # Fetch HTML
            html = await self._fetch_url(url, source_name)
            if not html:
                return None
            
            # Extract metadata
            metadata = self._extract_metadata_robust(html, url)
            
            # Extract content
            content = self._extract_content_robust(html, url)
            if not content or len(content.strip()) < 50:
                logger.debug(f"Insufficient content from {url}")
                return None
            
            # Language detection (no translation for now)
            language = metadata.get('language', 'en')
            
            article = {
                'title': metadata.get('title', 'Untitled'),
                'content': content,
                'url': url,
                'source': source_name,
                'author': metadata.get('author'),
                'published_date': metadata.get('published_date'),
                'language': language,
                'metadata': metadata
            }
            
            return article
            
        except Exception as e:
            logger.warning(f"Failed to scrape article {url}: {e}")
            return None
    
    async def _scrape_source(self, source: Dict[str, Any], max_articles: int = 10) -> List[Dict[str, Any]]:
        """Scrape articles from a single source"""
        source_name = source['name']
        url = source['url']
        source_type = source.get('type', 'news')
        
        logger.info(f"📡 Scraping {source_name}: {url}")
        
        try:
            # Fetch the main page
            html = await self._fetch_url(url, source_name)
            if not html:
                logger.warning(f"⚠️ {source_name}: Failed to fetch main page")
                return []
            
            # Extract article links
            soup = BeautifulSoup(html, 'html.parser')
            article_links = []
            
            # Find article links based on source type
            if source_type == 'news':
                # Look for common article link patterns
                link_patterns = [
                    'a[href*="/article/"]',
                    'a[href*="/news/"]',
                    'a[href*="/story/"]',
                    'a[href*="/202"]',  # Year-based URLs
                    'a[href*="/2025"]',
                    'a[href*="/2024"]',
                    'h1 a',
                    'h2 a',
                    'h3 a'
                ]
                
                for pattern in link_patterns:
                    links = soup.select(pattern)
                    for link in links:
                        href = link.get('href')
                        if href:
                            full_url = urljoin(url, href)
                            if full_url not in [l['url'] for l in article_links]:
                                article_links.append({'url': full_url, 'title': link.get_text().strip()})
            
            # Limit to max_articles
            article_links = article_links[:max_articles]
            
            if not article_links:
                logger.warning(f"⚠️ {source_name}: No articles found")
                return []
            
            # Scrape articles concurrently
            tasks = []
            for link in article_links:
                task = self._scrape_article(link['url'], source_name)
                tasks.append(task)
            
            # Execute with concurrency limit
            articles = []
            for completed in asyncio.as_completed(tasks):
                article = await completed
                if article:
                    articles.append(article)
                    logger.debug(f"✅ Scraped: {article['title'][:50]}...")
            
            logger.info(f"✅ {source_name}: {len(articles)} articles queued for preprocessing")
            return articles
            
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {e}")
            return []
    
    async def _queue_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Queue articles for preprocessing in Redis"""
        queued_count = 0
        
        for article in articles:
            try:
                # Generate unique article ID
                article_id = str(uuid.uuid4())
                
                # Create article data package for preprocessing
                article_data = {
                    'article_id': article_id,
                    'title': article['title'],
                    'body': article['content'],
                    'source_url': article['url'],
                    'region': None,  # Will be determined during preprocessing
                    'topic': None,   # Will be determined during preprocessing
                    'source': article['source'],
                    'language': article.get('language', 'en'),
                    'author': article.get('author'),
                    'published_date': article.get('published_date'),
                    'metadata': article.get('metadata', {}),
                    'created_at': datetime.utcnow().isoformat()
                }
                
                # Queue for preprocessing
                self.redis_client.lpush('preprocess', json.dumps(article_data))
                queued_count += 1
                
            except Exception as e:
                logger.error(f"Failed to queue article: {e}")
        
        return queued_count
    
    async def scrape_and_queue(self, sources: List[Dict[str, Any]] = None, max_articles_per_source: int = 10) -> Dict[str, Any]:
        """Scrape all sources and queue articles to Redis (do not store directly)"""
        if sources is None:
            sources = self.sources
        total_scraped = 0
        total_queued = 0
        errors = []
        start = time.time()
        for source in sources:
            try:
                articles = await self._scrape_source(source, max_articles=max_articles_per_source)
                total_scraped += len(articles)
                queued = await self._queue_articles(articles)
                total_queued += queued
            except Exception as e:
                logger.warning(f"Error scraping {source['name']}: {e}")
                errors.append(str(e))
        duration = time.time() - start
        logger.info(f"✅ Scraping completed: {total_queued} articles passed to preprocessing in {duration:.1f}s")
        return {
            'status': 'completed',
            'duration_seconds': duration,
            'sources_processed': len(sources),
            'total_scraped': total_scraped,
            'total_queued': total_queued,
            'errors': errors,
            'timestamp': datetime.utcnow().isoformat()
        }
    
    async def close(self):
        """Close the scraper session"""
        if self.session and not self.session.closed:
            await self.session.close()

# Global scraper instance
_scraper_instance = None

async def get_scraper() -> UnifiedHighSpeedScraper:
    """Get or create scraper instance"""
    global _scraper_instance
    if _scraper_instance is None:
        _scraper_instance = UnifiedHighSpeedScraper()
    return _scraper_instance

async def run_scraper(sources: List[Dict[str, Any]] = None, max_articles_per_source: int = 10) -> Dict[str, Any]:
    """Run the unified scraper"""
    scraper = await get_scraper()
    return await scraper.scrape_and_queue(sources, max_articles_per_source)

# Legacy compatibility functions
async def scrape_and_store(sources: List[Dict[str, Any]], max_articles_per_source: int = 10) -> Dict[str, Any]:
    """Legacy function for compatibility"""
    return await run_scraper(sources, max_articles_per_source)

# Export for use in tasks
__all__ = ['UnifiedHighSpeedScraper', 'run_scraper', 'scrape_and_store'] 