#!/usr/bin/env python3
"""
Scraper Service for Paperboy Backend
Feeds directly into Supabase for API consumption
"""

import asyncio
import aiohttp
import logging
import time
from datetime import datetime
from typing import List, Dict, Any, Optional
from bs4 import BeautifulSoup
import trafilatura
from urllib.parse import urljoin, urlparse
import re
try:
    from newspaper import Article as NewspaperArticle
except ImportError:
    NewspaperArticle = None

from app.utils.supabase_client import get_supabase_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArticleScraper:
    """Scraper that feeds directly into Supabase"""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        self.session = None
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(headers=self.headers)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def scrape_and_store(self, sources: List[Dict[str, Any]], max_articles_per_source: int = 10) -> Dict[str, Any]:
        """Scrape articles from sources and store directly in Supabase"""
        
        logger.info(f"🚀 Starting scraper for {len(sources)} sources")
        
        start_time = time.time()
        total_scraped = 0
        total_stored = 0
        errors = []
        
        for source in sources:
            try:
                source_name = source.get('name', 'Unknown')
                source_url = source.get('url')
                source_type = source.get('type', 'news')
                
                logger.info(f"📡 Scraping {source_name}: {source_url}")
                
                # Scrape articles from this source
                articles = await self._scrape_source(source_url, source_name, source_type, max_articles_per_source)
                
                if articles:
                    # Store articles directly in Supabase
                    stored_count = await self._store_articles(articles)
                    total_scraped += len(articles)
                    total_stored += stored_count
                    
                    logger.info(f"✅ {source_name}: {len(articles)} scraped, {stored_count} stored")
                else:
                    logger.warning(f"⚠️ {source_name}: No articles found")
                    
            except Exception as e:
                error_msg = f"Error scraping {source.get('name', 'Unknown')}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
        
        duration = time.time() - start_time
        
        result = {
            'status': 'completed',
            'duration_seconds': round(duration, 2),
            'sources_processed': len(sources),
            'total_scraped': total_scraped,
            'total_stored': total_stored,
            'errors': errors,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"✅ Scraping completed: {total_stored} articles stored in {duration:.1f}s")
        return result
    
    async def _scrape_source(self, url: str, source_name: str, source_type: str, max_articles: int) -> List[Dict[str, Any]]:
        """Scrape articles from a single source"""
        
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    logger.warning(f"Failed to fetch {url}: {response.status}")
                    return []
                
                html = await response.text()
                
                # Extract article links
                article_links = self._extract_article_links(html, url, source_type)
                
                if not article_links:
                    return []
                
                # Limit to max articles
                article_links = article_links[:max_articles]
                
                # Scrape each article
                articles = []
                for link in article_links:
                    try:
                        article = await self._scrape_article(link, source_name)
                        if article:
                            articles.append(article)
                    except Exception as e:
                        logger.warning(f"Failed to scrape article {link}: {e}")
                
                return articles
                
        except Exception as e:
            logger.error(f"Error scraping source {url}: {e}")
            return []
    
    def _extract_article_links(self, html: str, base_url: str, source_type: str) -> List[str]:
        """Extract article links from HTML"""
        
        soup = BeautifulSoup(html, 'html.parser')
        links = []
        
        # Common article link patterns
        article_selectors = [
            'a[href*="/article/"]',
            'a[href*="/news/"]',
            'a[href*="/story/"]',
            'a[href*="/202"]',  # Year-based URLs
            'a[href*="/2025"]',
            'a[href*="/2024"]',
            '.article-link a',
            '.news-link a',
            'article a',
            '.entry-title a',
            '.headline a'
        ]
        
        for selector in article_selectors:
            elements = soup.select(selector)
            for element in elements:
                href = element.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_article_url(full_url):
                        links.append(full_url)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_links = []
        for link in links:
            if link not in seen:
                seen.add(link)
                unique_links.append(link)
        
        return unique_links
    
    def _is_valid_article_url(self, url: str) -> bool:
        """Check if URL is a valid article URL"""
        
        # Skip non-http URLs
        if not url.startswith(('http://', 'https://')):
            return False
        
        # Skip common non-article URLs
        skip_patterns = [
            '/tag/',
            '/category/',
            '/author/',
            '/about/',
            '/contact/',
            '/privacy/',
            '/terms/',
            '/advertise/',
            '/subscribe/',
            '/login/',
            '/register/',
            '/search',
            '/sitemap',
            '.pdf',
            '.jpg',
            '.png',
            '.gif',
            '.mp4',
            '.mp3'
        ]
        
        for pattern in skip_patterns:
            if pattern in url.lower():
                return False
        
        return True
    
    async def _scrape_article(self, url: str, source_name: str) -> Optional[Dict[str, Any]]:
        """Scrape a single article"""
        
        try:
            async with self.session.get(url, timeout=30) as response:
                if response.status != 200:
                    return None
                
                html = await response.text()
                
                # Use trafilatura for content extraction
                extracted = trafilatura.extract(html, include_formatting=True, include_links=True)
                
                if not extracted and NewspaperArticle is not None:
                    # Fallback to newspaper3k if available
                    try:
                        n3k = NewspaperArticle(url)
                        n3k.download()
                        n3k.parse()
                        content = n3k.text
                    except Exception as e:
                        logger.warning(f"newspaper3k extraction failed for {url}: {e}")
                        content = None
                elif not extracted:
                    # Fallback to BeautifulSoup
                    soup = BeautifulSoup(html, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style"]):
                        script.decompose()
                    
                    # Try to find main content
                    content_selectors = [
                        'article',
                        '.article-content',
                        '.post-content',
                        '.entry-content',
                        '.story-content',
                        '.content',
                        'main',
                        '#content'
                    ]
                    
                    content = None
                    for selector in content_selectors:
                        element = soup.select_one(selector)
                        if element:
                            content = element.get_text(separator=' ', strip=True)
                            break
                    
                    if not content:
                        # Last resort: get body text
                        content = soup.get_text(separator=' ', strip=True)
                else:
                    content = extracted
                
                # Extract title
                try:
                    meta = trafilatura.extract_metadata(html)
                    title = meta["title"] if meta and "title" in meta else None
                except Exception as e:
                    logger.warning(f"trafilatura.extract_metadata failed: {e}")
                    title = None
                if not title:
                    soup = BeautifulSoup(html, 'html.parser')
                    title_elem = soup.find('title')
                    title = title_elem.get_text(strip=True) if title_elem else 'Untitled'
                
                # Clean and validate content
                if not content or len(content.strip()) < 100:
                    return None
                
                # Truncate content if too long
                if len(content) > 15000:
                    content = content[:15000] + "..."
                
                article = {
                    'title': title.strip(),
                    'content': content.strip(),
                    'url': url,
                    'source': source_name,
                    'published_at': datetime.now().isoformat(),
                    'scraped_at': datetime.now().isoformat(),
                    'language': 'en'
                }
                
                return article
                
        except Exception as e:
            logger.warning(f"Error scraping article {url}: {e}")
            return None
    
    async def _store_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Store articles in Supabase"""
        
        stored_count = 0
        
        for article in articles:
            try:
                # Check if article already exists
                existing = self.supabase.table('articles').select('id').eq('url', article['url']).execute()
                
                if existing.data:
                    # Article already exists, skip
                    continue
                
                # Insert new article
                result = self.supabase.table('articles').insert({
                    'title': article['title'],
                    'content': article['content'],
                    'url': article['url'],
                    'source': article['source'],
                    'published_at': article['published_at'],
                    'relevant': None  # Will be set by processing pipeline
                }).execute()
                
                if result.data:
                    stored_count += 1
                    
            except Exception as e:
                logger.error(f"Error storing article {article.get('url', 'Unknown')}: {e}")
        
        return stored_count

# Predefined sources for Taiwan Strait monitoring
TAIWAN_STRAIT_SOURCES = [
    {
        'name': 'Focus Taiwan',
        'url': 'https://focustaiwan.tw/politics',
        'type': 'news'
    },
    {
        'name': 'Taipei Times',
        'url': 'http://www.taipeitimes.com/News/front',
        'type': 'news'
    },
    {
        'name': 'China Daily',
        'url': 'https://www.chinadaily.com.cn/world',
        'type': 'news'
    },
    {
        'name': 'South China Morning Post',
        'url': 'https://www.scmp.com/news/asia',
        'type': 'news'
    },
    {
        'name': 'Reuters Asia',
        'url': 'https://www.reuters.com/world/asia-pacific/',
        'type': 'news'
    },
    {
        'name': 'Bloomberg Asia',
        'url': 'https://www.bloomberg.com/asia',
        'type': 'news'
    },
    {
        'name': 'CNN Asia',
        'url': 'https://www.cnn.com/world/asia',
        'type': 'news'
    },
    {
        'name': 'NYT Asia',
        'url': 'https://www.nytimes.com/section/world/asia',
        'type': 'news'
    }
]

async def run_scraper(sources: List[Dict[str, Any]] = None, max_articles_per_source: int = 10) -> Dict[str, Any]:
    """Run the scraper with specified sources"""
    
    if sources is None:
        sources = TAIWAN_STRAIT_SOURCES
    
    async with ArticleScraper() as scraper:
        result = await scraper.scrape_and_store(sources, max_articles_per_source)
        return result 