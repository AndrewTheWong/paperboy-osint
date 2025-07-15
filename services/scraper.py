import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from services.async_scraper import RobustAsyncScraper

class Scraper:
    def __init__(self):
        self.scraper = RobustAsyncScraper()

    def scrape_sources(self, sources_path: str):
        sources = RobustAsyncScraper.load_sources_from_file(sources_path)
        import asyncio
        return asyncio.run(self.scraper.scrape_sources(sources))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the article scraper.")
    parser.add_argument('--sources', type=str, default="../sources/master_sources.json", help="Path to sources file")
    args = parser.parse_args()
    scraper = Scraper()
    results = scraper.scrape_sources(args.sources)
    print(f"Scraped {len(results)} articles.") 