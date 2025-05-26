import requests
from bs4 import BeautifulSoup
from datetime import datetime
import random
import time
import logging

logger = logging.getLogger(__name__)

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

def scrape():
    """Scrape articles from Xinhua World News."""
    url = "https://english.news.cn/world/"
    articles = []
    timestamp = datetime.utcnow().isoformat()
    
    logger.info(f"Scraping Xinhua from {url}")
    response = make_request(url)
    if not response:
        return articles
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find all links within content sections
    article_links = soup.select('a[href*="/202"]')
    
    for link in article_links:
        title = link.text.strip()
        url = link.get('href', '')
        
        # Skip non-article links or links without proper titles
        if not url or not title or len(title) < 10:
            continue
            
        # Add protocol if missing
        if url.startswith('//'):
            url = f"https:{url}"
        elif not url.startswith(('http://', 'https://')):
            url = f"https://english.news.cn{url}" if url.startswith('/') else f"https://english.news.cn/{url}"
            
        if title and url and not any(a['url'] == url for a in articles):
            articles.append({
                'title': title,
                'url': url,
                'source': 'Xinhua',
                'scraped_at': timestamp
            })
    
    logger.info(f"Scraped {len(articles)} articles from Xinhua")
    return articles

if __name__ == "__main__":
    # Configure logging for standalone testing
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Test the scraper when run directly
    articles = scrape()
    
    # Print results
    print(f"\nScraped {len(articles)} articles from Xinhua")
    for article in articles[:5]:  # Show first 5 articles
        print(f"\n{article['title'][:100]}...")
        print(f"URL: {article['url']}") 