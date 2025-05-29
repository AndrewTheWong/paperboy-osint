"""
Simple Test Script for Article Scraping
Tests different scraping methods on sample URLs
"""

import requests
from bs4 import BeautifulSoup
import re

def clean_text(text):
    """Clean extracted text"""
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    text = re.sub(r'\n+', '\n', text)
    
    return text

def test_simple_scraper(url):
    """Test simple scraping approach"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        print(f"📥 Requesting: {url}")
        response = requests.get(url, headers=headers, timeout=10, allow_redirects=True)
        print(f"   Status: {response.status_code}")
        
        if response.status_code != 200:
            return f"HTTP {response.status_code}", ""
        
        html = response.text
        print(f"   HTML length: {len(html)} chars")
        
        # Parse with BeautifulSoup
        soup = BeautifulSoup(html, 'html.parser')
        
        # Remove unwanted elements
        for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
            element.decompose()
        
        # Get all paragraph text
        paragraphs = soup.find_all(['p', 'div'], string=True)
        texts = [p.get_text() for p in paragraphs if len(p.get_text().strip()) > 20]
        
        if not texts:
            # Fallback: get all text
            text = soup.get_text()
        else:
            text = ' '.join(texts)
        
        text = clean_text(text)
        print(f"   Extracted text: {len(text)} chars")
        
        if len(text) > 100:
            print(f"   Preview: {text[:200]}...")
            return "success", text
        else:
            return "too_short", text
            
    except Exception as e:
        print(f"   Error: {e}")
        return f"error_{type(e).__name__}", ""

# Test URLs
test_urls = [
    "https://www.yahoo.com/news/trump-picks-jared-kushners-father-225933590.html",
    "https://www.bbc.com/news/world-us-canada-67833542",
    "https://www.reuters.com/world/americas/",
    "https://www.cnn.com/2024/12/01/politics/trump-cabinet/index.html",
    "https://apnews.com/article/ukraine-russia-war-12345"
]

print("🧪 Testing Simple Article Scraper")
print("=" * 50)

for i, url in enumerate(test_urls, 1):
    print(f"\n{i}. Testing URL:")
    status, text = test_simple_scraper(url)
    print(f"   Result: {status}")
    
    if status == "success":
        print(f"   ✅ Success! Got {len(text)} characters")
    else:
        print(f"   ❌ Failed: {status}")

print("\n�� Test complete!") 