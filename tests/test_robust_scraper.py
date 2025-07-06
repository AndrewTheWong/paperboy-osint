import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
import asyncio
from services.async_scraper import RobustAsyncScraper

@pytest.mark.asyncio
async def test_multi_source_scraping():
    sources = [
        {"name": "Test News", "url": "https://example.com/news1", "type": "news"},
        {"name": "Test Blog", "url": "https://example.com/blog1", "type": "blog"}
    ]
    scraper = RobustAsyncScraper()
    results = await scraper.scrape_sources(sources)
    assert isinstance(results, list)
    assert all("url" in r for r in results)

@pytest.mark.asyncio
async def test_metadata_extraction():
    html = "<html><head><title>Test Title</title></head><body>Content</body></html>"
    scraper = RobustAsyncScraper()
    meta = scraper.extract_metadata(html, "https://example.com")
    assert meta["title"] == "Test Title"

@pytest.mark.asyncio
async def test_language_detection():
    scraper = RobustAsyncScraper()
    assert scraper.detect_language("This is English.") == "en"
    assert scraper.detect_language("这是中文。") in ("zh", "zh-cn")

@pytest.mark.asyncio
async def test_error_handling():
    sources = [{"name": "Bad URL", "url": "https://nonexistent.example.com", "type": "news"}]
    scraper = RobustAsyncScraper()
    results = await scraper.scrape_sources(sources)
    assert results == []

@pytest.mark.asyncio
async def test_redis_queueing(monkeypatch):
    class DummyRedis:
        def __init__(self): self.queued = []
        async def lpush(self, queue, item): self.queued.append((queue, item))
    scraper = RobustAsyncScraper(redis_client=DummyRedis())
    await scraper.queue_article({"title": "Test", "content": "Test content", "url": "https://example.com"})
    assert scraper.redis_client.queued 