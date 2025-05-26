#!/usr/bin/env python3
"""
Universal fallback scraper that can be used for any site.
"""
import logging
from datetime import datetime
import random
import time
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

# Try importing BeautifulSoup, but continue with a stub if it's not available
try:
    import requests
    from bs4 import BeautifulSoup
    HAS_DEPS = True
except ImportError:
    logger.warning("Missing dependencies: beautifulsoup4 or requests. Install with: pip install beautifulsoup4 requests")
    HAS_DEPS = False

# List of user agents to rotate
USER_AGENTS = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:89.0) Gecko/20100101 Firefox/89.0',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
]

def get_random_headers():
    """Generate random headers with a rotating user agent."""
    return {
        'User-Agent': random.choice(USER_AGENTS),
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
        'Accept-Language': 'en-US,en;q=0.5',
        'Connection': 'keep-alive',
        'Upgrade-Insecure-Requests': '1',
        'Cache-Control': 'max-age=0'
    }

def make_request(url, max_retries=3, backoff_factor=1.5):
    """Make a request with retry logic and backoff."""
    if not HAS_DEPS:
        logger.error("Cannot make request: missing requests library")
        return None
        
    for attempt in range(max_retries):
        try:
            headers = get_random_headers()
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            logger.warning(f"Request attempt {attempt + 1}/{max_retries} failed: {str(e)}")
            if attempt < max_retries - 1:
                sleep_time = backoff_factor * (2 ** attempt)
                logger.info(f"Retrying in {sleep_time:.2f} seconds...")
                time.sleep(sleep_time)
            else:
                logger.error(f"Failed to fetch {url} after {max_retries} attempts")
                return None

def extract_article_links(soup, base_url):
    """Extract article links from a page."""
    if not HAS_DEPS:
        logger.error("Cannot extract links: missing BeautifulSoup library")
        return []
        
    links = []
    
    # Look for all anchors in the page
    all_links = soup.find_all('a', href=True)
    
    for link in all_links:
        href = link.get('href', '').strip()
        
        # Skip if empty or just a fragment/anchor
        if not href or href.startswith('#'):
            continue
            
        # Make URL absolute if it's relative
        if not href.startswith(('http://', 'https://')):
            href = base_url.rstrip('/') + ('/' if not href.startswith('/') else '') + href
        
        # Check if URL is likely an article (contains /article/, /news/, etc.)
        article_indicators = ['/article/', '/news/', '/story/', '/politics/', 
                             '/world/', '/asia/', '/content/', '/archive/']
        
        if any(indicator in href.lower() for indicator in article_indicators):
            title = link.text.strip()
            
            # Only include links with reasonable titles (at least 20 chars)
            if title and len(title) > 20:
                links.append({
                    'url': href,
                    'title': title
                })
    
    return links

def scrape_site(url, source_name=None, language='en'):
    """
    Universal scraper that extracts article links from any site.
    
    Args:
        url: The URL to scrape
        source_name: The name of the source (defaults to domain name)
        language: The language code of the articles
        
    Returns:
        List of article dictionaries
    """
    logger.info(f"Scraping site: {url}")
    
    # Check for dependencies
    if not HAS_DEPS:
        logger.error("Cannot scrape site: missing dependencies (beautifulsoup4 or requests)")
        # Return sample data as fallback
        timestamp = datetime.utcnow().isoformat()
        return [
            {
                "title": f"Sample article from {source_name or 'unknown source'}",
                "url": f"{url}/sample/1",
                "source": source_name or "Sample Source",
                "scraped_at": timestamp,
                "language": language
            }
        ]
    
    # Extract domain for source name if not provided
    if not source_name:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc
        source_name = domain.replace('www.', '').split('.')[0].title()
    
    timestamp = datetime.utcnow().isoformat()
    articles = []
    
    # Make request
    response = make_request(url)
    if not response:
        logger.error(f"Failed to fetch {url}")
        return articles
    
    # Parse HTML
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract article links
    article_links = extract_article_links(soup, url)
    
    # Create article dictionaries
    for link in article_links:
        articles.append({
            'title': link['title'],
            'url': link['url'],
            'source': source_name,
            'scraped_at': timestamp,
            'language': language
        })
    
    logger.info(f"Scraped {len(articles)} articles from {source_name}")
    return articles

if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test scraping a site
    import sys
    url = sys.argv[1] if len(sys.argv) > 1 else "https://www.bbc.com/news/world/asia"
    source = sys.argv[2] if len(sys.argv) > 2 else None
    
    articles = scrape_site(url, source)
    
    # Print results
    print(f"\nScraped {len(articles)} articles from {articles[0]['source'] if articles else 'unknown source'}")
    for article in articles[:5]:  # Show first 5 articles
        print(f"\n{article['title'][:100]}...")
        print(f"URL: {article['url']}") 