#!/usr/bin/env python3
"""
Enhanced News Scraper - Streamlined Version
Use NewsIngest.py for full parallel processing pipeline.
"""

import asyncio
import aiohttp
import json
import logging
import random
import time
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse
import os

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

# Configure enhanced logging with real-time output
class RealTimeFormatter(logging.Formatter):
    def format(self, record):
        formatted = super().format(record)
        print(f"🔄 {formatted}", flush=True)  # Real-time output
        return formatted

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('enhanced_scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('EnhancedScraper')

@dataclass
class EnhancedScrapingConfig:
    """Enhanced configuration for comprehensive news scraping."""
    max_workers: int = 8
    max_retries: int = 2
    base_delay: float = 1.0
    max_delay: float = 15.0
    selenium_timeout: int = 20
    articles_per_source: int = 10
    content_timeout: int = 10
    max_text_length: int = 20000
    use_headless_chrome: bool = True  # Prevent Chrome windows
    enable_translation: bool = True
    enable_tagging: bool = True
    enable_content_extraction: bool = True

class EnhancedContentExtractor:
    """Advanced content extraction with site-specific selectors."""
    
    def __init__(self):
        self.site_selectors = {
            'nytimes.com': {
                'title': 'h1[data-testid="headline"]',
                'content': 'section[name="articleBody"] p',
                'date': 'time[datetime]'
            },
            'washingtonpost.com': {
                'title': 'h1[data-testid="headline"]',
                'content': '.article-body p',
                'date': 'time'
            },
            'reuters.com': {
                'title': 'h1[data-testid="Heading"]',
                'content': '[data-testid="ArticleBody"] p',
                'date': 'time[datetime]'
            },
            'bbc.com': {
                'title': 'h1[data-testid="headline"]',
                'content': '[data-component="text-block"] p',
                'date': 'time[datetime]'
            },
            'aljazeera.com': {
                'title': 'h1',
                'content': '.wysiwyg p',
                'date': 'time[datetime]'
            },
            'taipeitimes.com': {
                'title': 'h1.article-title',
                'content': '.article-content p',
                'date': '.article-date'
            },
            'default': {
                'title': 'h1, .title, .headline, [data-testid="headline"]',
                'content': 'article p, .content p, .article-body p, .story-body p',
                'date': 'time, .date, .published, [datetime]'
            }
        }
    
    def extract_content(self, html: str, url: str) -> Dict[str, str]:
        """Extract title, content, and date from HTML."""
        if not HAS_BS4:
            return {'title': '', 'content': '', 'date': ''}
        
        soup = BeautifulSoup(html, 'html.parser')
        domain = urlparse(url).netloc.lower()
        
        # Find matching selectors
        selectors = self.site_selectors.get('default')
        for site, site_selectors in self.site_selectors.items():
            if site in domain:
                selectors = site_selectors
                break
        
        # Extract title
        title = ''
        for selector in selectors['title'].split(','):
            elem = soup.select_one(selector.strip())
            if elem:
                title = elem.get_text(strip=True)
                break
        
        # Extract content
        content_parts = []
        for selector in selectors['content'].split(','):
            elements = soup.select(selector.strip())
            if elements:
                content_parts = [elem.get_text(strip=True) for elem in elements if elem.get_text(strip=True)]
                break
        
        content = ' '.join(content_parts[:10])  # First 10 paragraphs
        
        # Extract date
        date = ''
        for selector in selectors['date'].split(','):
            elem = soup.select_one(selector.strip())
            if elem:
                date = elem.get('datetime') or elem.get_text(strip=True)
                break
        
        return {
            'title': title[:500] if title else '',
            'content': content[:5000] if content else '',
            'date': date
        }

class ArticleTranslator:
    """Translation service for non-English content."""
    
    def __init__(self):
        self.translator = Translator() if HAS_TRANSLATE else None
        self.cache = {}
    
    def detect_language(self, text: str) -> str:
        """Detect language of text."""
        if not HAS_LANGDETECT or not text:
            return 'en'
        
        try:
            return detect(text)
        except:
            return 'en'
    
    def translate_to_english(self, text: str, source_lang: str = None) -> str:
        """Translate text to English."""
        if not self.translator or not text:
            return text
        
        if text in self.cache:
            return self.cache[text]
        
        try:
            if not source_lang:
                source_lang = self.detect_language(text)
            
            if source_lang == 'en':
                return text
            
            result = self.translator.translate(text, dest='en', src=source_lang)
            translated = result.text
            self.cache[text] = translated
            return translated
        except Exception as e:
            logger.warning(f"Translation failed: {e}")
            return text

class ArticleTagger:
    """Content tagging and analysis."""
    
    def __init__(self):
        self.escalation_keywords = [
            'military', 'war', 'conflict', 'attack', 'strike', 'bomb', 'missile',
            'tension', 'threat', 'sanctions', 'embargo', 'crisis', 'emergency'
        ]
        
        self.geographic_keywords = {
            'China': ['china', 'chinese', 'beijing', 'prc', 'mainland'],
            'Taiwan': ['taiwan', 'taiwanese', 'taipei', 'roc', 'formosa'],
            'Japan': ['japan', 'japanese', 'tokyo', 'nippon'],
            'Korea': ['korea', 'korean', 'seoul', 'pyongyang'],
            'US': ['america', 'american', 'usa', 'united states', 'washington'],
            'Russia': ['russia', 'russian', 'moscow', 'kremlin'],
            'Iran': ['iran', 'iranian', 'tehran', 'persian'],
            'Israel': ['israel', 'israeli', 'jerusalem', 'tel aviv']
        }
    
    def tag_article(self, title: str, content: str) -> Dict[str, Any]:
        """Generate tags and scores for article."""
        text = f"{title} {content}".lower()
        
        # Keyword tags
        keyword_tags = []
        escalation_score = 0
        
        for keyword in self.escalation_keywords:
            if keyword in text:
                keyword_tags.append(keyword)
                escalation_score += text.count(keyword)
        
        # Geographic tags
        geographic_tags = []
        for country, keywords in self.geographic_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    geographic_tags.append(country)
                    break
        
        # Sentiment analysis (simple)
        positive_words = ['peace', 'agreement', 'cooperation', 'dialogue', 'success']
        negative_words = ['war', 'conflict', 'crisis', 'attack', 'threat']
        
        sentiment_score = 0
        for word in positive_words:
            sentiment_score += text.count(word)
        for word in negative_words:
            sentiment_score -= text.count(word)
        
        return {
            'keyword_tags': keyword_tags,
            'geographic_tags': geographic_tags,
            'sentiment_score': max(-10, min(10, sentiment_score)),
            'escalation_score': min(100, escalation_score * 10),
            'significance_score': len(keyword_tags) * 5 + len(geographic_tags) * 3
        }

class EnhancedNewsScraper:
    """Enhanced news scraper with comprehensive features."""
    
    def __init__(self, config: EnhancedScrapingConfig = None):
        self.config = config or EnhancedScrapingConfig()
        self.sources = self._load_comprehensive_sources()
        self.content_extractor = EnhancedContentExtractor()
        self.translator = ArticleTranslator() if self.config.enable_translation else None
        self.tagger = ArticleTagger() if self.config.enable_tagging else None
        self.supabase = self._init_supabase() if HAS_SUPABASE else None
        
        logger.info(f"🚀 Enhanced scraper initialized with {len(self._count_sources())} sources")
    
    def _load_comprehensive_sources(self) -> Dict[str, Any]:
        """Load comprehensive 100+ source configuration."""
        sources_file = Path(__file__).parent / "comprehensive_sources.json"
        try:
            with open(sources_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Could not load comprehensive sources: {e}")
            # Fallback to basic sources
            return {
                "basic_sources": {
                    "bbc": {
                        "name": "BBC",
                        "urls": ["https://www.bbc.com/news/world"],
                        "region": "UK",
                        "language": "en",
                        "paywall": "none",
                        "category": "international"
                    }
                }
            }
    
    def _count_sources(self) -> int:
        """Count total number of sources."""
        count = 0
        for category in self.sources.values():
            count += len(category)
        return count
    
    def _init_supabase(self) -> Optional[Client]:
        """Initialize Supabase client."""
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            return create_client(supabase_url, supabase_key)
        else:
            logger.warning("Supabase credentials not found")
            return None
    
    def create_headless_driver(self) -> Optional[Any]:
        """Create headless Chrome driver to prevent windows opening."""
        if not HAS_SELENIUM:
            return None
        
        try:
            options = ChromeOptions()
            
            if self.config.use_headless_chrome:
                options.add_argument('--headless=new')  # New headless mode
                options.add_argument('--no-sandbox')
                options.add_argument('--disable-dev-shm-usage')
                options.add_argument('--disable-gpu')
                options.add_argument('--window-size=1920,1080')
                
            # Stealth options
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option('useAutomationExtension', False)
            options.add_argument('--disable-blink-features=AutomationControlled')
            
            driver = uc.Chrome(options=options, version_main=None)
            driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
            
            return driver
        except Exception as e:
            logger.warning(f"Could not create Chrome driver: {e}")
            return None
    
    async def scrape_with_content_extraction(self, url: str, source_config: Dict[str, Any]) -> Optional[str]:
        """Scrape with enhanced content extraction."""
        try:
            # Try requests first
            timeout = aiohttp.ClientTimeout(total=self.config.content_timeout)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            }
            
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(url) as response:
                    if response.status == 200:
                        return await response.text()
                    
        except Exception as e:
            logger.debug(f"HTTP request failed for {url}: {e}")
        
        # Fallback to Selenium if needed
        if source_config.get('requires_javascript') or source_config.get('paywall') == 'hard':
            return await self._scrape_with_selenium(url)
        
        return None
    
    async def _scrape_with_selenium(self, url: str) -> Optional[str]:
        """Scrape using headless Selenium."""
        driver = None
        try:
            driver = self.create_headless_driver()
            if not driver:
                return None
            
            logger.info(f"🌐 Using headless browser for {url}")
            driver.get(url)
            
            # Wait for content
            WebDriverWait(driver, self.config.selenium_timeout).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            return driver.page_source
            
        except Exception as e:
            logger.warning(f"Selenium scraping failed for {url}: {e}")
            return None
        finally:
            if driver:
                try:
                    driver.quit()
                except:
                    pass
    
    async def process_article(self, url: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process single article with full pipeline."""
        logger.info(f"📄 Processing: {url}")
        
        try:
            # Scrape content
            html = await self.scrape_with_content_extraction(url, source_config)
            if not html:
                logger.warning(f"❌ No content retrieved for {url}")
                return None
            
            # Extract structured content
            extracted = self.content_extractor.extract_content(html, url)
            if not extracted['title'] and not extracted['content']:
                logger.warning(f"⚠️ No structured content extracted from {url}")
                return None
            
            # Detect language
            language = 'en'
            if HAS_LANGDETECT and extracted['content']:
                try:
                    language = detect(extracted['content'])
                except:
                    language = 'en'
            
            # Translate if needed
            title_en = extracted['title']
            content_en = extracted['content']
            
            if self.translator and language != 'en':
                logger.info(f"🌍 Translating from {language} to English")
                title_en = self.translator.translate_to_english(extracted['title'], language)
                content_en = self.translator.translate_to_english(extracted['content'], language)
            
            # Tag and analyze
            tags = {}
            if self.tagger:
                tags = self.tagger.tag_article(title_en, content_en)
                logger.info(f"🏷️ Tagged with {len(tags.get('keyword_tags', []))} keywords")
            
            # Build article object
            article = {
                'url': url,
                'title': title_en,
                'content': content_en,
                'summary': content_en[:500] + '...' if len(content_en) > 500 else content_en,
                'source': source_config.get('name', 'Unknown'),
                'language': language,
                'primary_country': self._extract_primary_country(tags.get('geographic_tags', [])),
                'primary_region': source_config.get('region', 'Unknown'),
                'published_at': self._parse_date(extracted.get('date')),
                'scraped_at': datetime.now().isoformat(),
                'word_count': len(content_en.split()),
                'article_type': 'news',
                'keyword_tags': tags.get('keyword_tags', []),
                'geographic_tags': tags.get('geographic_tags', []),
                'sentiment_score': tags.get('sentiment_score', 0),
                'escalation_score': tags.get('escalation_score', 0),
                'significance_score': tags.get('significance_score', 0)
            }
            
            logger.info(f"✅ Successfully processed: {article['title'][:100]}...")
            return article
            
        except Exception as e:
            logger.error(f"❌ Error processing {url}: {e}")
            return None
    
    def _extract_primary_country(self, geographic_tags: List[str]) -> Optional[str]:
        """Extract primary country from geographic tags."""
        if not geographic_tags:
            return None
        return geographic_tags[0]  # First mentioned country
    
    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse date string to ISO format."""
        if not date_str:
            return datetime.now().isoformat()
        
        # Add more sophisticated date parsing here
        return datetime.now().isoformat()
    
    async def scrape_source(self, source_key: str, source_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scrape a single source with enhanced processing."""
        logger.info(f"🔍 Scraping {source_config.get('name', source_key)} ({source_config.get('region', 'Unknown')})")
        
        articles = []
        urls = source_config.get('urls', [])
        
        for base_url in urls[:2]:  # Limit URLs per source
            try:
                # Get article URLs from index page
                html = await self.scrape_with_content_extraction(base_url, source_config)
                if not html:
                    continue
                
                article_urls = self._extract_article_urls(html, base_url, source_config)
                logger.info(f"📋 Found {len(article_urls)} article URLs from {base_url}")
                
                # Process individual articles
                for article_url in article_urls[:self.config.articles_per_source]:
                    article = await self.process_article(article_url, source_config)
                    if article:
                        articles.append(article)
                    
                    # Rate limiting
                    await asyncio.sleep(random.uniform(0.5, 1.5))
                
            except Exception as e:
                logger.error(f"❌ Error scraping source {source_key}: {e}")
        
        logger.info(f"📊 {source_key}: {len(articles)} articles processed")
        return articles
    
    def _extract_article_urls(self, html: str, base_url: str, source_config: Dict[str, Any]) -> List[str]:
        """Extract article URLs from index page."""
        if not HAS_BS4:
            return []
        
        soup = BeautifulSoup(html, 'html.parser')
        urls = set()
        
        # Find article links
        selectors = [
            'a[href*="/news/"]',
            'a[href*="/article/"]',
            'a[href*="/story/"]',
            'a[href*="/world/"]',
            'a[href*="/politics/"]',
            'article a',
            '.article-link',
            '.story-link'
        ]
        
        for selector in selectors:
            links = soup.select(selector)
            for link in links:
                href = link.get('href')
                if href:
                    full_url = urljoin(base_url, href)
                    if self._is_valid_article_url(full_url):
                        urls.add(full_url)
        
        return list(urls)
    
    def _is_valid_article_url(self, url: str) -> bool:
        """Check if URL is a valid article URL."""
        if not url or len(url) < 10:
            return False
        
        # Skip non-article patterns
        skip_patterns = [
            'javascript:', 'mailto:', '#', 'tel:', 'ftp:',
            '.pdf', '.jpg', '.png', '.gif', '.mp4', '.mp3',
            '/tag/', '/category/', '/search/', '/author/',
            'facebook.com', 'twitter.com', 'instagram.com'
        ]
        
        url_lower = url.lower()
        return not any(pattern in url_lower for pattern in skip_patterns)
    
    def _prepare_article_for_supabase(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare article data for osint_articles schema."""
        # Clean data and ensure proper types
        scraped_at = article.get('scraped_at') or datetime.now().isoformat()
        content = article.get('content', '') or ''
        title = article.get('title', '') or ''
        
        return {
            'url': article.get('url', ''),
            'title': title,
            'content': content if content else None,
            'source': article.get('source', ''),
            'published_at': article.get('published_at') if article.get('published_at') else None,
            'scraped_at': scraped_at,
            'language': article.get('language', 'en'),
            'primary_region': article.get('primary_region') if article.get('primary_region') else None,
            'primary_country': article.get('primary_country') if article.get('primary_country') else None,
            'primary_city': article.get('primary_city') if article.get('primary_city') else None,
            'word_count': len(content.split()) if content else 0,
            'article_type': 'news',
            'keyword_tags': article.get('keywords', []),
            'escalation_score': float(article.get('escalation_score', 0.0)),
            'sentiment_score': float(article.get('sentiment_polarity', 0.0)),
            'geographic_confidence': float(article.get('geographic_confidence', 0.0)),
            'all_locations': article.get('all_locations', []),
            'geographic_tags': article.get('geographic_tags', []),
            'semantic_tags': article.get('semantic_tags', []),
            'title_embedding': article.get('title_embedding') if article.get('title_embedding') else None,
            'content_embedding': article.get('content_embedding') if article.get('content_embedding') else None,
            'significance_score': float(article.get('significance_score', 0.0))
        }

    async def upload_to_supabase(self, articles: List[Dict[str, Any]]) -> Tuple[int, int, int]:
        """Upload articles to Supabase with comprehensive data."""
        if not self.supabase:
            logger.warning("⚠️ Supabase not available")
            return 0, 0, len(articles)
        
        uploaded = 0
        updated = 0
        errors = 0
        
        logger.info(f"📤 Uploading {len(articles)} articles to Supabase...")
        
        for article in articles:
            try:
                # Prepare article for Supabase schema
                supabase_article = self._prepare_article_for_supabase(article)
                
                # Check if exists
                existing = self.supabase.table('osint_articles').select('id').eq('url', article['url']).execute()
                
                if existing.data:
                    # Update existing
                    self.supabase.table('osint_articles').update(supabase_article).eq('url', article['url']).execute()
                    updated += 1
                    logger.debug(f"🔄 Updated: {article['title'][:50]}...")
                else:
                    # Insert new
                    self.supabase.table('osint_articles').insert(supabase_article).execute()
                    uploaded += 1
                    logger.debug(f"➕ Inserted: {article['title'][:50]}...")
                    
            except Exception as e:
                logger.warning(f"❌ Upload failed for {article.get('url', 'unknown')}: {e}")
                errors += 1
        
        logger.info(f"📊 Upload complete: {uploaded} new, {updated} updated, {errors} errors")
        return uploaded, updated, errors
    
    async def run_comprehensive_scraping(self, categories: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive scraping across all sources."""
        start_time = datetime.now()
        logger.info(f"🚀 Starting comprehensive news scraping at {start_time}")
        
        all_articles = []
        stats = {
            'start_time': start_time,
            'sources_attempted': 0,
            'sources_successful': 0,
            'total_articles': 0,
            'by_category': {},
            'by_region': {},
            'by_language': {}
        }
        
        # Filter sources by categories
        sources_to_scrape = {}
        for category, sources in self.sources.items():
            if categories is None or category in categories:
                sources_to_scrape.update(sources)
                stats['by_category'][category] = 0
        
        stats['sources_attempted'] = len(sources_to_scrape)
        logger.info(f"📊 Will scrape {stats['sources_attempted']} sources across {len(self.sources)} categories")
        
        # Create semaphore for concurrency control
        semaphore = asyncio.Semaphore(self.config.max_workers)
        
        async def scrape_with_semaphore(source_key, source_config):
            async with semaphore:
                try:
                    articles = await self.scrape_source(source_key, source_config)
                    if articles:
                        stats['sources_successful'] += 1
                        stats['total_articles'] += len(articles)
                        
                        # Update category stats
                        for category in self.sources:
                            if source_key in self.sources[category]:
                                stats['by_category'][category] += len(articles)
                        
                        # Update region/language stats
                        region = source_config.get('region', 'Unknown')
                        language = source_config.get('language', 'en')
                        stats['by_region'][region] = stats['by_region'].get(region, 0) + len(articles)
                        stats['by_language'][language] = stats['by_language'].get(language, 0) + len(articles)
                    
                    return articles
                except Exception as e:
                    logger.error(f"❌ Source {source_key} failed: {e}")
                    return []
        
        # Execute scraping tasks
        tasks = [scrape_with_semaphore(key, config) for key, config in sources_to_scrape.items()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Collect results
        for result in results:
            if isinstance(result, list):
                all_articles.extend(result)
        
        # Upload to Supabase
        upload_stats = (0, 0, 0)
        if all_articles and self.supabase:
            upload_stats = await self.upload_to_supabase(all_articles)
        
        # Generate final report
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        stats.update({
            'end_time': end_time,
            'duration_seconds': duration,
            'articles_scraped': len(all_articles),
            'upload_new': upload_stats[0],
            'upload_updated': upload_stats[1],
            'upload_errors': upload_stats[2]
        })
        
        logger.info(f"🎉 Scraping complete! {stats['total_articles']} articles from {stats['sources_successful']}/{stats['sources_attempted']} sources in {duration:.1f}s")
        
        return {
            'articles': all_articles,
            'statistics': stats
        }

# Main execution function
async def main():
    """Run enhanced scraping pipeline."""
    config = EnhancedScrapingConfig(
        max_workers=6,
        articles_per_source=5,
        use_headless_chrome=True,  # Prevent Chrome windows
        enable_translation=True,
        enable_tagging=True,
        enable_content_extraction=True
    )
    
    scraper = EnhancedNewsScraper(config)
    results = await scraper.run_comprehensive_scraping()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"data/enhanced_scraping_results_{timestamp}.json"
    
    Path("data").mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"💾 Results saved to {output_file}")
    return results

if __name__ == "__main__":
    asyncio.run(main()) 