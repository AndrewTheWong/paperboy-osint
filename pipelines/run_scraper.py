#!/usr/bin/env python3
"""
Script to run the web scraper and print results.
"""
import json
import sys
import os
import time
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipelines.web_scraper import scrape_all_sources

def json_serial(obj):
    """JSON serializer for objects not serializable by default json code"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")

def main():
    """Run scraper and display results."""
    print("Starting web scraper...")
    start_time = time.time()
    
    # Run the scraper
    articles = scrape_all_sources()
    
    # Calculate runtime
    runtime = time.time() - start_time
    
    # Print summary
    print(f"\nScraping completed in {runtime:.2f} seconds")
    print(f"Total articles scraped: {len(articles)}")
    
    # Print source breakdown
    sources = {}
    for article in articles:
        source = article['source']
        if source not in sources:
            sources[source] = 0
        sources[source] += 1
    
    print("\nArticles by source:")
    for source, count in sources.items():
        print(f"- {source}: {count} articles")
    
    # Print sample articles (first from each source)
    print("\nSample articles:")
    seen_sources = set()
    for article in articles:
        source = article['source']
        if source not in seen_sources:
            seen_sources.add(source)
            print(f"\n{article['source']} - {article['title']}")
            print(f"URL: {article['url']}")
    
    # Save to file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"articles_{timestamp}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(articles, f, indent=2, default=json_serial)
    
    print(f"\nFull results saved to {output_file}")

if __name__ == "__main__":
    main() 