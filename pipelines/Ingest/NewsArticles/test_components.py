#!/usr/bin/env python3
"""
Test Multiprocessing Pipeline

Simple test to verify the multiprocessing pipeline works with basic sources.
"""

import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.append(str(project_root))

def test_processors():
    """Test individual processors."""
    print(" Testing processors...")
    
    try:
        from pipelines.Ingest.NewsArticles.processors.embedding_processor import EmbeddingProcessor
        embedder = EmbeddingProcessor()
        
        # Test embedding
        test_text = "This is a test article about international news."
        embedding = embedder.create_embedding(test_text)
        
        if embedding and len(embedding) == 384:
            print(" Embedding processor working (384 dimensions)")
        else:
            print(" Embedding processor failed")
            
    except Exception as e:
        print(f" Embedding processor error: {e}")
        
    try:
        from pipelines.Ingest.NewsArticles.processors.article_tagger import UnifiedArticleTagger
        tagger = UnifiedArticleTagger()
        
        # Test tagging
        result = tagger.tag_article("Test Title", "This is test content about China and the United States.")
        
        if result and 'keywords' in result:
            print(" Article tagger working")
        else:
            print(" Article tagger failed")
            
    except Exception as e:
        print(f" Article tagger error: {e}")
        
    try:
        from pipelines.Ingest.NewsArticles.processors.geographic_tagger import GeographicTagger
        geo_tagger = GeographicTagger()
        
        # Test geographic extraction
        result = geo_tagger.extract_geographic_info("Breaking news from Beijing, China about tensions with Taiwan.")
        
        if result and result.get('primary_country'):
            print(" Geographic tagger working")
        else:
            print(" Geographic tagger failed")
            
    except Exception as e:
        print(f" Geographic tagger error: {e}")

def test_simple_scraping():
    """Test simple web scraping."""
    print(" Testing web scraping...")
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        url = "https://httpbin.org/html"
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        if soup.find('title'):
            print(" Web scraping working")
        else:
            print(" Web scraping failed")
            
    except Exception as e:
        print(f" Web scraping error: {e}")

def main():
    """Run all tests."""
    print(" Testing Multiprocessing Pipeline Components\n")
    
    test_processors()
    print()
    test_simple_scraping()
    
    print("\n Basic tests completed!")
    print("\nTo run the full pipeline:")
    print("python pipelines/Ingest/NewsArticles/multiprocessing_pipeline.py")

if __name__ == "__main__":
    main()
