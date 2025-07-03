#!/usr/bin/env python3
"""
Async web crawler for StraitWatch
Crawls news sites and sends articles to the backend
"""

import asyncio
import aiohttp
import json
import logging
import time
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
from typing import List, Set, Dict, Optional
import re
import uuid
import requests

from ingest_client import IngestClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class AsyncCrawler:
    """Asynchronous web crawler for news articles"""
    
    def __init__(self, max_concurrent: int = 10, timeout: int = 30, collect_batch: bool = False):
        """
        Initialize crawler
        
        Args:
            max_concurrent: Maximum concurrent requests
            timeout: Request timeout in seconds
            collect_batch: Whether to collect articles instead of sending them
        """
        self.max_concurrent = max_concurrent
        self.timeout = timeout
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.visited_urls: Set[str] = set()
        self.to_visit: asyncio.Queue = asyncio.Queue()
        self.stats = {
            'visited': 0,
            'sent': 0,
            'failed': 0,
            'start_time': None
        }
        self.ingest_client = IngestClient()
        self.collect_batch = collect_batch
        self.collected_articles = []
        
    def is_valid_url(self, url: str) -> bool:
        """
        Check if URL is valid for crawling
        
        Args:
            url: URL to check
            
        Returns:
            bool: True if valid
        """
        try:
            parsed = urlparse(url)
            # Skip non-HTTP URLs
            if parsed.scheme not in ['http', 'https']:
                return False
            # Skip common non-article URLs
            skip_patterns = [
                r'\.pdf$', r'\.doc$', r'\.docx$', r'\.xls$', r'\.xlsx$',
                r'\.zip$', r'\.rar$', r'\.mp3$', r'\.mp4$', r'\.avi$',
                r'/login', r'/signup', r'/register', r'/admin',
                r'\.jpg$', r'\.jpeg$', r'\.png$', r'\.gif$', r'\.svg$'
            ]
            for pattern in skip_patterns:
                if re.search(pattern, url, re.IGNORECASE):
                    return False
            return True
        except Exception:
            return False
    
    def is_same_domain(self, base_url: str, url: str) -> bool:
        """
        Check if URL is from the same domain
        
        Args:
            base_url: Base URL
            url: URL to check
            
        Returns:
            bool: True if same domain
        """
        try:
            base_domain = urlparse(base_url).netloc
            url_domain = urlparse(url).netloc
            return base_domain == url_domain
        except Exception:
            return False
    
    def extract_article_content(self, soup: BeautifulSoup, url: str) -> Optional[Dict]:
        """
        Extract article content from BeautifulSoup object
        
        Args:
            soup: BeautifulSoup object
            url: Source URL
            
        Returns:
            Optional[Dict]: Article data or None
        """
        try:
            # Extract title
            title = ""
            title_selectors = [
                'h1', 'h1.title', 'h1.headline', '.title', '.headline',
                'title', '[property="og:title"]', '[name="twitter:title"]'
            ]
            
            for selector in title_selectors:
                title_elem = soup.select_one(selector)
                if title_elem:
                    title = title_elem.get_text().strip()
                    break
            
            if not title:
                title = soup.find('title')
                if title:
                    title = title.get_text().strip()
            
            # Extract body text
            body_text = ""
            body_selectors = [
                'article', '.article-content', '.story-content', '.post-content',
                '.entry-content', '.content', 'main', '.main-content'
            ]
            
            # Try to find main content area
            content_area = None
            for selector in body_selectors:
                content_area = soup.select_one(selector)
                if content_area:
                    break
            
            if not content_area:
                # Fallback to body
                content_area = soup.find('body')
            
            if content_area:
                # Extract text from paragraphs
                paragraphs = content_area.find_all('p')
                body_text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
            
            # Check if we have enough content
            if len(body_text) < 200:  # Minimum article length
                return None
            
            # Infer region based on domain
            region = self.infer_region(url)
            
            # Infer topic from content
            topic = self.infer_topic(title, body_text)
            
            return {
                'title': title,
                'body': body_text,
                'source_url': url,
                'region': region,
                'topic': topic
            }
            
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return None
    
    def infer_region(self, url: str) -> str:
        """
        Infer region from URL
        
        Args:
            url: Source URL
            
        Returns:
            str: Inferred region
        """
        domain = urlparse(url).netloc.lower()
        
        if any(x in domain for x in ['cn', 'china', 'globaltimes']):
            return "China"
        elif any(x in domain for x in ['sg', 'singapore', 'straitstimes', 'todayonline']):
            return "Singapore"
        elif any(x in domain for x in ['hk', 'hongkong', 'scmp']):
            return "Hong Kong"
        elif any(x in domain for x in ['asia', 'channelnewsasia']):
            return "Southeast Asia"
        elif any(x in domain for x in ['reuters', 'bbc', 'cnn', 'yahoo']):
            return "International"
        else:
            return "East Asia"
    
    def infer_topic(self, title: str, body: str) -> str:
        """
        Infer topic from title and body
        
        Args:
            title: Article title
            body: Article body
            
        Returns:
            str: Inferred topic
        """
        text = (title + " " + body).lower()
        
        # Define topic keywords
        topics = {
            'Politics': ['politics', 'government', 'election', 'president', 'minister', 'parliament'],
            'Economy': ['economy', 'economic', 'finance', 'business', 'trade', 'market', 'stock'],
            'Technology': ['technology', 'tech', 'digital', 'ai', 'artificial intelligence', 'software'],
            'Security': ['security', 'defense', 'military', 'navy', 'army', 'weapon', 'terrorism'],
            'Diplomacy': ['diplomacy', 'foreign', 'international', 'treaty', 'alliance', 'summit'],
            'Society': ['society', 'social', 'culture', 'education', 'health', 'environment']
        }
        
        # Count keyword matches
        topic_scores = {}
        for topic, keywords in topics.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                topic_scores[topic] = score
        
        if topic_scores:
            # Return topic with highest score
            return max(topic_scores, key=topic_scores.get)
        else:
            return "General"
    
    def extract_links(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """
        Extract internal links from page
        
        Args:
            soup: BeautifulSoup object
            base_url: Base URL for resolving relative links
            
        Returns:
            List[str]: List of internal URLs
        """
        links = []
        try:
            for link in soup.find_all('a', href=True):
                href = link['href']
                # Resolve relative URLs
                full_url = urljoin(base_url, href)
                # Check if it's a valid internal link
                if self.is_valid_url(full_url) and self.is_same_domain(base_url, full_url):
                    links.append(full_url)
        except Exception as e:
            logger.error(f"Error extracting links from {base_url}: {e}")
        
        return list(set(links))  # Remove duplicates
    
    async def fetch_page(self, session: aiohttp.ClientSession, url: str) -> Optional[str]:
        """
        Fetch page content
        
        Args:
            session: aiohttp session
            url: URL to fetch
            
        Returns:
            Optional[str]: Page content or None
        """
        async with self.semaphore:
            try:
                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                }
                
                async with session.get(url, headers=headers, timeout=self.timeout) as response:
                    if response.status == 200:
                        content_type = response.headers.get('content-type', '').lower()
                        if 'text/html' in content_type:
                            return await response.text()
                        else:
                            logger.debug(f"Skipping non-HTML content: {url}")
                            return None
                    else:
                        logger.debug(f"HTTP {response.status} for {url}")
                        return None
                        
            except asyncio.TimeoutError:
                logger.debug(f"Timeout fetching {url}")
                return None
            except Exception as e:
                logger.debug(f"Error fetching {url}: {e}")
                return None
    
    async def process_page(self, session: aiohttp.ClientSession, url: str, depth: int) -> None:
        """
        Process a single page
        
        Args:
            session: aiohttp session
            url: URL to process
            depth: Current crawl depth
        """
        if url in self.visited_urls:
            return
        
        self.visited_urls.add(url)
        self.stats['visited'] += 1
        
        logger.info(f"Processing {url} (depth {depth})")
        
        # Fetch page content
        content = await self.fetch_page(session, url)
        if not content:
            self.stats['failed'] += 1
            return
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(content, 'html.parser')
        
        # Extract article content
        article = self.extract_article_content(soup, url)
        if article:
            article['article_id'] = str(uuid.uuid4())
            article['source_url'] = article['source_url']
            if self.collect_batch:
                self.collected_articles.append(article)
                # Send batch and exit if 20 collected
                if len(self.collected_articles) >= 20:
                    await self.send_batch_and_exit()
                    return
            else:
                # Send to backend
                try:
                    result = self.ingest_client.send_article(
                        title=article['title'],
                        body=article['body'],
                        source_url=article['source_url'],
                        region=article.get('region'),
                        topic=article.get('topic'),
                        article_id=article.get('article_id')
                    )
                    if result.get('status') == 'queued':
                        self.stats['sent'] += 1
                        logger.info(f"✅ Sent article: {article['title'][:50]}...")
                    else:
                        self.stats['failed'] += 1
                        logger.error(f"❌ Failed to send article: {result.get('error', 'Unknown error')}")
                except Exception as e:
                    self.stats['failed'] += 1
                    logger.error(f"❌ Error sending article: {e}")
        
        # Extract links for next level
        if depth < 2:  # Max depth of 2
            links = self.extract_links(soup, url)
            for link in links:
                if link not in self.visited_urls:
                    await self.to_visit.put((link, depth + 1))
    
    async def send_batch_and_exit(self):
        logger.info(f"📤 Sending batch of {len(self.collected_articles)} articles to /ingest/v2/batch-optimized/")
        try:
            response = requests.post(
                f"http://localhost:8000/ingest/v2/batch-optimized/?batch_size={len(self.collected_articles)}",
                json=self.collected_articles,
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            if response.status_code == 200:
                logger.info(f"✅ Batch sent successfully: {response.json()}")
            else:
                logger.error(f"❌ Batch send failed: {response.status_code} {response.text}")
        except Exception as e:
            logger.error(f"❌ Error sending batch: {e}")
        import sys
        sys.exit(0)
    
    async def crawl_and_ingest(self, start_urls: List[str], max_depth: int = 2) -> Dict:
        """
        Main crawling function
        
        Args:
            start_urls: List of starting URLs
            max_depth: Maximum crawl depth
            
        Returns:
            Dict: Crawling statistics
        """
        self.stats['start_time'] = time.time()
        
        # Add starting URLs to queue
        for url in start_urls:
            await self.to_visit.put((url, 0))
        
        # Create aiohttp session
        connector = aiohttp.TCPConnector(limit=100, limit_per_host=30)
        timeout = aiohttp.ClientTimeout(total=self.timeout)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Process URLs until queue is empty
            while not self.to_visit.empty():
                url, depth = await self.to_visit.get()
                
                if depth <= max_depth:
                    await self.process_page(session, url, depth)
                
                # Small delay to be respectful
                await asyncio.sleep(0.1)
        
        # Calculate final statistics
        elapsed_time = time.time() - self.stats['start_time']
        self.stats['elapsed_time'] = elapsed_time
        self.stats['avg_speed'] = self.stats['visited'] / elapsed_time if elapsed_time > 0 else 0
        
        logger.info(f"🎉 Crawling complete!")
        logger.info(f"📊 Statistics:")
        logger.info(f"   Visited: {self.stats['visited']}")
        logger.info(f"   Articles sent: {self.stats['sent']}")
        logger.info(f"   Failed: {self.stats['failed']}")
        logger.info(f"   Time: {elapsed_time:.2f}s")
        logger.info(f"   Speed: {self.stats['avg_speed']:.2f} pages/sec")
        
        return self.stats

async def main():
    """Main function"""
    import argparse
    parser = argparse.ArgumentParser(description="Async Crawler for StraitWatch")
    parser.add_argument('--batch', action='store_true', help='Collect and send articles as a batch')
    args = parser.parse_args()
    try:
        # Load URLs from JSON file
        with open('scraper/input_urls.json', 'r') as f:
            start_urls = json.load(f)
        
        logger.info(f"🚀 Starting crawler with {len(start_urls)} seed URLs")
        
        # Create and run crawler
        crawler = AsyncCrawler(max_concurrent=10, timeout=30, collect_batch=args.batch)
        stats = await crawler.crawl_and_ingest(start_urls, max_depth=2)
        
        logger.info("✅ Crawling completed successfully")
        
    except FileNotFoundError:
        logger.error("❌ input_urls.json not found in scraper/ directory")
    except json.JSONDecodeError:
        logger.error("❌ Invalid JSON in input_urls.json")
    except Exception as e:
        logger.error(f"❌ Error in main: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 