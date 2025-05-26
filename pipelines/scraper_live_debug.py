#!/usr/bin/env python3
"""
Debug script to analyze website structure for better scraping.
"""
import sys
import os
import requests
from bs4 import BeautifulSoup
import json
import time
import random

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.web_scraper import get_random_headers

def analyze_site(url, selector=None, find_links=True):
    """
    Download and analyze a webpage to understand its structure.
    
    Args:
        url: The URL to analyze
        selector: Optional CSS selector to focus on specific elements
        find_links: Whether to extract links from the page
    """
    print(f"Analyzing {url}...")
    
    try:
        # Make the request with a random user agent
        headers = get_random_headers()
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse the HTML
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Basic page info
        print(f"\nPage title: {soup.title.text if soup.title else 'No title'}")
        print(f"Response size: {len(response.content)} bytes")
        
        # Extract common structural elements
        print("\nCommon elements found:")
        for element_type in ['header', 'nav', 'main', 'article', 'section', 'div.content', 'div.main']:
            elements = soup.select(element_type)
            if elements:
                print(f"- {element_type}: {len(elements)} found")
        
        # If a specific selector is provided, analyze those elements
        if selector:
            elements = soup.select(selector)
            print(f"\nFound {len(elements)} elements matching '{selector}'")
            
            # Show the first few elements
            for i, element in enumerate(elements[:3]):
                print(f"\nElement {i+1}:")
                print(element.prettify()[:500] + "..." if len(element.prettify()) > 500 else element.prettify())
        
        # Find potential article elements
        if find_links:
            print("\nPotential article links:")
            link_patterns = [
                ('a[href*="article"]', []),
                ('a[href*="news"]', []),
                ('a[href*="story"]', []),
                ('h1 a, h2 a, h3 a', []),
                ('.title a, .headline a', [])
            ]
            
            for selector, results in link_patterns:
                links = soup.select(selector)
                for link in links[:5]:  # Show at most 5 examples
                    title = link.text.strip()
                    url = link.get('href', '')
                    if title and url and len(title) > 15:  # Likely an article title
                        results.append((title, url))
                        print(f"[{selector}] {title[:50]}... -> {url}")
            
            # Save the HTML for further analysis
            with open('debug_page.html', 'w', encoding='utf-8') as f:
                f.write(soup.prettify())
            print("\nSaved page HTML to debug_page.html for further analysis")
            
    except Exception as e:
        print(f"Error analyzing {url}: {str(e)}")

def main():
    """Main function to analyze websites."""
    sites = [
        {
            "name": "Taipei Times",
            "url": "https://www.taipeitimes.com/News/front",
            "selector": ".list, .story, article"
        },
        {
            "name": "SCMP",
            "url": "https://www.scmp.com/news/china",
            "selector": ".article-title, .story-title, [data-testid='article-title-link']"
        },
        {
            "name": "New York Times",
            "url": "https://www.nytimes.com/section/world",
            "selector": "article h2, .story-heading, .css-1kv6qi"
        }
    ]
    
    # Allow selecting a specific site to analyze
    if len(sys.argv) > 1:
        site_index = int(sys.argv[1])
        if 0 <= site_index < len(sites):
            sites = [sites[site_index]]
        else:
            print(f"Invalid site index. Choose 0-{len(sites)-1}")
            return
    
    for site in sites:
        analyze_site(site["url"], site["selector"])
        print("\n" + "="*50 + "\n")
        
        # Sleep between requests to avoid being blocked
        if len(sites) > 1:
            time.sleep(random.uniform(2.0, 4.0))

if __name__ == "__main__":
    main() 