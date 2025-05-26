import requests
from bs4 import BeautifulSoup
from datetime import datetime
import random
import time
import logging
import re

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('web_scraper')

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

def scrape_taipei_times():
    """Scrape articles from Taipei Times front page."""
    url = "https://www.taipeitimes.com/News/front"
    articles = []
    timestamp = datetime.utcnow().isoformat()
    
    response = make_request(url)
    if not response:
        return articles
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Find articles by looking at all anchors with href containing "archives"
    article_links = soup.find_all('a', href=lambda href: href and "/archives/" in href)
    
    for link in article_links:
        title = link.text.strip()
        relative_url = link.get('href', '')
        
        # Skip non-article links or links without proper titles
        if not relative_url or not title or len(title) < 10:
            continue
            
        full_url = f"https://www.taipeitimes.com{relative_url}" if relative_url and not relative_url.startswith('http') else relative_url
        
        if title and full_url and not any(a['url'] == full_url for a in articles):
            articles.append({
                'title': title,
                'url': full_url,
                'source': 'Taipei Times',
                'scraped_at': timestamp
            })
    
    logger.info(f"Scraped {len(articles)} articles from Taipei Times")
    return articles

def scrape_xinhua():
    """Scrape articles from Xinhua World News."""
    url = "https://english.news.cn/world/"
    articles = []
    timestamp = datetime.utcnow().isoformat()
    
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

def scrape_south_china_morning_post():
    """Scrape articles from South China Morning Post."""
    url = "https://www.scmp.com/news/china"
    articles = []
    timestamp = datetime.utcnow().isoformat()
    
    response = make_request(url)
    if not response:
        return articles
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Try multiple selector patterns to find article links
    article_links = soup.select('a[href*="/article/"]')
    
    for link in article_links:
        # Check if link contains a title element
        title_element = link.find('h1') or link.find('h2') or link.find('h3') or link.find('h4') or link
        title = title_element.text.strip()
        url = link.get('href', '')
        
        # Skip non-article links or links without proper titles
        if not url or not title or len(title) < 10:
            continue
            
        # Add domain if URL is relative
        if url.startswith('/'):
            url = f"https://www.scmp.com{url}"
            
        if title and url and not any(a['url'] == url for a in articles):
            articles.append({
                'title': title,
                'url': url,
                'source': 'South China Morning Post',
                'scraped_at': timestamp
            })
    
    logger.info(f"Scraped {len(articles)} articles from South China Morning Post")
    return articles

def scrape_nyt_world():
    """Scrape articles from New York Times World section."""
    url = "https://www.nytimes.com/section/world"
    articles = []
    timestamp = datetime.utcnow().isoformat()
    
    response = make_request(url)
    if not response:
        return articles
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # The New York Times has strict anti-scraping measures
    # Look for all links that might be articles
    article_links = soup.select('a[href*="/world/"]')
    
    for link in article_links:
        # Find title within the link
        title_element = link.find('h2') or link.find('h3') or link
        title = title_element.text.strip()
        url = link.get('href', '')
        
        # Skip non-article links or links without proper titles
        if not url or not title or len(title) < 10:
            continue
            
        # Skip links that aren't to articles
        if '/interactive/' in url or '/video/' in url or '/section/' in url:
            continue
            
        # Add domain if URL is relative
        if url.startswith('/'):
            url = f"https://www.nytimes.com{url}"
            
        if title and url and not any(a['url'] == url for a in articles):
            articles.append({
                'title': title,
                'url': url,
                'source': 'New York Times',
                'scraped_at': timestamp
            })
    
    logger.info(f"Scraped {len(articles)} articles from New York Times")
    return articles

def scrape_china_daily():
    """Scrape articles from China Daily."""
    url = "https://www.chinadaily.com.cn/world"
    articles = []
    timestamp = datetime.utcnow().isoformat()
    
    response = make_request(url)
    if not response:
        return articles
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # China Daily article pattern
    article_links = soup.select('a[href*="/a/"]')
    
    for link in article_links:
        title = link.text.strip()
        url = link.get('href', '')
        
        # Skip non-article links or links without proper titles
        if not url or not title or len(title) < 10:
            continue
            
        # Add domain if URL is relative
        if url.startswith('/'):
            url = f"https://www.chinadaily.com.cn{url}"
        elif not url.startswith(('http://', 'https://')):
            url = f"https://www.chinadaily.com.cn/{url}"
            
        if title and url and not any(a['url'] == url for a in articles):
            articles.append({
                'title': title,
                'url': url,
                'source': 'China Daily',
                'scraped_at': timestamp
            })
    
    logger.info(f"Scraped {len(articles)} articles from China Daily")
    return articles

def scrape_globaltimes():
    """Scrape articles from Global Times."""
    url = "https://www.globaltimes.cn/world/"
    articles = []
    timestamp = datetime.utcnow().isoformat()
    
    response = make_request(url)
    if not response:
        return articles
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Global Times article pattern
    article_links = soup.select('a.list_link, a.article-title')
    
    for link in article_links:
        title = link.text.strip()
        url = link.get('href', '')
        
        # Skip non-article links or links without proper titles
        if not url or not title or len(title) < 10:
            continue
            
        # Add domain if URL is relative
        if url.startswith('/'):
            url = f"https://www.globaltimes.cn{url}"
            
        if title and url and not any(a['url'] == url for a in articles):
            articles.append({
                'title': title,
                'url': url,
                'source': 'Global Times',
                'scraped_at': timestamp
            })
    
    logger.info(f"Scraped {len(articles)} articles from Global Times")
    return articles

def scrape_all_sources():
    """Scrape articles from all configured sources."""
    all_articles = []
    
    # Dict of scraper functions to call
    scrapers = {
        'Taipei Times': scrape_taipei_times,
        'Xinhua': scrape_xinhua,
        'South China Morning Post': scrape_south_china_morning_post,
        'New York Times': scrape_nyt_world,
        'China Daily': scrape_china_daily,
        'Global Times': scrape_globaltimes
    }
    
    for source_name, scraper_func in scrapers.items():
        try:
            logger.info(f"Scraping {source_name}...")
            articles = scraper_func()
            all_articles.extend(articles)
            
            # Add a small delay between requests to different domains
            time.sleep(random.uniform(1.0, 3.0))
            
        except Exception as e:
            logger.error(f"Error scraping {source_name}: {str(e)}")
    
    logger.info(f"Scraped a total of {len(all_articles)} articles from all sources")
    return all_articles

if __name__ == "__main__":
    # Test the scrapers when run directly
    articles = scrape_all_sources()
    
    # Print a summary of results
    for source in set(article['source'] for article in articles):
        source_articles = [a for a in articles if a['source'] == source]
        print(f"{source}: {len(source_articles)} articles")
    
    # Print a sample article from each source
    print("\nSample articles:")
    for source in set(article['source'] for article in articles):
        source_articles = [a for a in articles if a['source'] == source]
        if source_articles:
            sample = source_articles[0]
            print(f"\n{sample['source']} - {sample['title']}")
            print(f"URL: {sample['url']}")
            print(f"Scraped at: {sample['scraped_at']}") 