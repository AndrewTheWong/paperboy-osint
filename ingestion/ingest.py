import feedparser
import requests
from bs4 import BeautifulSoup
import pdfplumber
import os
from storage.db import insert_osint_entry
from datetime import datetime

def fetch_rss_articles(feed_urls):
    """Fetch articles from RSS feeds."""
    articles = []
    for url in feed_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries:
                content = entry.get('summary', '') or entry.get('content', [{}])[0].get('value', '')
                articles.append({
                    "source_url": entry.link,
                    "content": content
                })
        except Exception as e:
            print(f"Error fetching RSS feed from {url}: {str(e)}")
    return articles

def scrape_static_sites(urls):
    """Scrape content from static websites."""
    articles = []
    for url in urls:
        try:
            resp = requests.get(url)
            soup = BeautifulSoup(resp.text, 'html.parser')
            paragraphs = soup.find_all('p')
            content = '\n'.join(p.text for p in paragraphs)
            articles.append({
                "source_url": url,
                "content": content
            })
        except Exception as e:
            print(f"Error scraping website {url}: {str(e)}")
    return articles

def parse_pdf_documents(pdf_paths):
    """Extract text from PDF documents."""
    articles = []
    for path in pdf_paths:
        try:
            with pdfplumber.open(path) as pdf:
                text = ''
                for page in pdf.pages:
                    text += page.extract_text() + '\n'
                articles.append({
                    "source_url": f"file://{path}",
                    "content": text
                })
        except Exception as e:
            print(f"Error parsing PDF {path}: {str(e)}")
    return articles

def ingest_sources():
    """Main function to ingest content from all sources."""
    # Example sources
    rss = fetch_rss_articles([
        "https://www.foreignaffairs.com/rss.xml", 
        "https://feeds.bbci.co.uk/news/world/rss.xml"
    ])
    
    static = scrape_static_sites([
        "https://www.state.gov/latest-updates/"
    ])
    
    # For PDF ingestion, check for PDFs in a data directory
    pdf_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    pdf_files = []
    if os.path.exists(pdf_dir):
        pdf_files = [os.path.join(pdf_dir, f) for f in os.listdir(pdf_dir) if f.endswith('.pdf')]
    
    pdfs = parse_pdf_documents(pdf_files)

    all_sources = rss + static + pdfs
    for item in all_sources:
        try:
            insert_osint_entry(item['source_url'], item['content'])
        except Exception as e:
            print(f"Error inserting entry from {item['source_url']}: {str(e)}")
    
    print(f"Ingested {len(all_sources)} entries.")

if __name__ == "__main__":
    ingest_sources() 