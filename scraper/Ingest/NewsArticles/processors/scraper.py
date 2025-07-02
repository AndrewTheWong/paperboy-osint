#!/usr/bin/env python3
"""
Professional News Article Scraper
High-performance web scraping with parallel Selenium instances and robust error handling
"""

import asyncio
import logging
import time
import random
from typing import Dict, List, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException, WebDriverException
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

class ArticleScraper:
    """High-performance article scraper with parallel processing"""
    
    def __init__(self, max_workers: int = 8, timeout: int = 30):
        self.max_workers = max_workers
        self.timeout = timeout
        self.drivers = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def _create_driver(self) -> webdriver.Chrome:
        """Create a Chrome WebDriver instance with optimized settings"""
        options = Options()
        options.add_argument('--headless')
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--disable-extensions')
        options.add_argument('--disable-logging')
        options.add_argument('--disable-web-security')
        options.add_argument('--allow-running-insecure-content')
        options.add_argument('--ignore-certificate-errors')
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        driver = webdriver.Chrome(options=options)
        driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        driver.set_page_load_timeout(self.timeout)
        return driver
    
    def initialize_drivers(self):
        """Initialize the driver pool"""
        logger.info(f"Initializing {self.max_workers} Chrome drivers...")
        for i in range(self.max_workers):
            try:
                driver = self._create_driver()
                self.drivers.append(driver)
                logger.info(f"Driver {i+1}/{self.max_workers} initialized")
            except Exception as e:
                logger.error(f"Failed to initialize driver {i+1}: {e}")
        
        if not self.drivers:
            raise RuntimeError("Failed to initialize any drivers")
        
        logger.info(f"Successfully initialized {len(self.drivers)} drivers")
    
    def close_drivers(self):
        """Close all drivers"""
        for driver in self.drivers:
            try:
                driver.quit()
            except Exception as e:
                logger.warning(f"Error closing driver: {e}")
        self.drivers.clear()
    
    def scrape_article_content(self, url: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Scrape a single article with fallback strategies"""
        try:
            # Try simple requests first for faster sources
            if not source_config.get('requires_javascript', False):
                return self._scrape_with_requests(url, source_config)
            else:
                return self._scrape_with_selenium(url, source_config)
        except Exception as e:
            logger.error(f"Failed to scrape {url}: {e}")
            return None
    
    def _scrape_with_requests(self, url: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Fast scraping using requests for simple sites"""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract content using common selectors
            title = self._extract_title(soup, source_config)
            content = self._extract_content(soup, source_config)
            
            if not title or not content or len(content) < 100:
                return None
            
            return {
                'url': url,
                'title': title.strip(),
                'content': content.strip(),
                'source': source_config['name'],
                'language': source_config.get('language', 'en'),
                'scrape_method': 'requests'
            }
        except Exception as e:
            logger.debug(f"Requests scraping failed for {url}: {e}")
            return None
    
    def _scrape_with_selenium(self, url: str, source_config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Selenium scraping for JavaScript-heavy sites"""
        driver = None
        try:
            # Get available driver
            if not self.drivers:
                return None
            
            driver = self.drivers.pop(0)
            
            driver.get(url)
            
            # Wait for content to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Random delay to appear human
            time.sleep(random.uniform(1, 3))
            
            # Extract content
            title = self._extract_title_selenium(driver, source_config)
            content = self._extract_content_selenium(driver, source_config)
            
            if not title or not content or len(content) < 100:
                return None
            
            return {
                'url': url,
                'title': title.strip(),
                'content': content.strip(),
                'source': source_config['name'],
                'language': source_config.get('language', 'en'),
                'scrape_method': 'selenium'
            }
        except Exception as e:
            logger.debug(f"Selenium scraping failed for {url}: {e}")
            return None
        finally:
            if driver:
                self.drivers.append(driver)  # Return driver to pool
    
    def _extract_title(self, soup: BeautifulSoup, source_config: Dict[str, Any]) -> Optional[str]:
        """Extract title using multiple selectors"""
        selectors = [
            'h1',
            '.headline',
            '.title',
            '[data-testid="headline"]',
            '.entry-title',
            '.article-title'
        ]
        
        for selector in selectors:
            element = soup.select_one(selector)
            if element and element.get_text(strip=True):
                return element.get_text(strip=True)
        
        # Fallback to page title
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True)
        
        return None
    
    def _extract_content(self, soup: BeautifulSoup, source_config: Dict[str, Any]) -> Optional[str]:
        """Extract article content using multiple selectors"""
        selectors = [
            '.article-body',
            '.entry-content',
            '.post-content',
            '.story-body',
            '[data-testid="article-body"]',
            '.content',
            'article p'
        ]
        
        for selector in selectors:
            elements = soup.select(selector)
            if elements:
                content = ' '.join([elem.get_text(strip=True) for elem in elements])
                if len(content) > 100:
                    return content
        
        # Fallback: get all paragraphs
        paragraphs = soup.find_all('p')
        if paragraphs:
            content = ' '.join([p.get_text(strip=True) for p in paragraphs])
            if len(content) > 100:
                return content
        
        return None
    
    def _extract_title_selenium(self, driver: webdriver.Chrome, source_config: Dict[str, Any]) -> Optional[str]:
        """Extract title using Selenium"""
        selectors = [
            'h1',
            '.headline',
            '.title',
            '[data-testid="headline"]',
            '.entry-title',
            '.article-title'
        ]
        
        for selector in selectors:
            try:
                element = driver.find_element(By.CSS_SELECTOR, selector)
                text = element.text.strip()
                if text:
                    return text
            except:
                continue
        
        return driver.title
    
    def _extract_content_selenium(self, driver: webdriver.Chrome, source_config: Dict[str, Any]) -> Optional[str]:
        """Extract content using Selenium"""
        selectors = [
            '.article-body',
            '.entry-content',
            '.post-content',
            '.story-body',
            '[data-testid="article-body"]',
            '.content'
        ]
        
        for selector in selectors:
            try:
                elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if elements:
                    content = ' '.join([elem.text.strip() for elem in elements])
                    if len(content) > 100:
                        return content
            except:
                continue
        
        # Fallback: get all paragraphs
        try:
            paragraphs = driver.find_elements(By.TAG_NAME, 'p')
            if paragraphs:
                content = ' '.join([p.text.strip() for p in paragraphs])
                if len(content) > 100:
                    return content
        except:
            pass
        
        return None
    
    def discover_article_urls(self, source_url: str, source_config: Dict[str, Any], max_urls: int = 50) -> List[str]:
        """Discover article URLs from a source page"""
        try:
            if source_config.get('requires_javascript', False):
                return self._discover_urls_selenium(source_url, source_config, max_urls)
            else:
                return self._discover_urls_requests(source_url, source_config, max_urls)
        except Exception as e:
            logger.error(f"Failed to discover URLs from {source_url}: {e}")
            return []
    
    def _discover_urls_requests(self, source_url: str, source_config: Dict[str, Any], max_urls: int) -> List[str]:
        """Discover URLs using requests"""
        response = self.session.get(source_url, timeout=self.timeout)
        response.raise_for_status()
        
        soup = BeautifulSoup(response.content, 'html.parser')
        return self._extract_article_urls(soup, source_url, max_urls)
    
    def _discover_urls_selenium(self, source_url: str, source_config: Dict[str, Any], max_urls: int) -> List[str]:
        """Discover URLs using Selenium"""
        driver = None
        try:
            if not self.drivers:
                return []
            
            driver = self.drivers.pop(0)
            driver.get(source_url)
            
            # Wait for content
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "a"))
            )
            
            # Get page source and parse
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            return self._extract_article_urls(soup, source_url, max_urls)
        finally:
            if driver:
                self.drivers.append(driver)
    
    def _extract_article_urls(self, soup: BeautifulSoup, base_url: str, max_urls: int) -> List[str]:
        """Extract article URLs from page"""
        urls = set()
        
        # Common article link patterns
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Convert relative URLs to absolute
            full_url = urljoin(base_url, href)
            
            # Filter for likely article URLs
            if self._is_article_url(full_url):
                urls.add(full_url)
                if len(urls) >= max_urls:
                    break
        
        return list(urls)[:max_urls]
    
    def _is_article_url(self, url: str) -> bool:
        """Check if URL is likely an article"""
        # Basic filtering for article URLs
        article_patterns = [
            '/article/', '/news/', '/story/', '/post/', '/blog/',
            '/world/', '/politics/', '/international/'
        ]
        
        exclude_patterns = [
            '/tag/', '/category/', '/author/', '/search/',
            '.pdf', '.jpg', '.png', '.gif', '.css', '.js'
        ]
        
        url_lower = url.lower()
        
        # Must contain article-like patterns
        has_article_pattern = any(pattern in url_lower for pattern in article_patterns)
        
        # Must not contain exclude patterns
        has_exclude_pattern = any(pattern in url_lower for pattern in exclude_patterns)
        
        return has_article_pattern and not has_exclude_pattern
    
    def scrape_articles_parallel(self, urls_and_configs: List[tuple], max_articles: int = 1000) -> List[Dict[str, Any]]:
        """Scrape multiple articles in parallel"""
        logger.info(f"Starting parallel scraping of up to {max_articles} articles using {self.max_workers} workers")
        
        articles = []
        processed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit initial batch
            futures = []
            for url, config in urls_and_configs[:max_articles]:
                future = executor.submit(self.scrape_article_content, url, config)
                futures.append(future)
                processed += 1
            
            # Collect results
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=self.timeout + 10)
                    if result:
                        articles.append(result)
                        logger.info(f"Scraped article {len(articles)}: {result['title'][:50]}...")
                    
                    if len(articles) >= max_articles:
                        break
                        
                except Exception as e:
                    logger.debug(f"Failed to scrape article {i+1}: {e}")
        
        logger.info(f"Successfully scraped {len(articles)} articles from {processed} attempts")
        return articles[:max_articles] 