"""
Article Ingestion Agent for StraitWatch

Continuously monitors news sources and ingests articles into the database.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from pathlib import Path

from .base_agent import BaseAgent
from pipelines.Ingest.NewsIngest import NewsIngest
from utils.supabase_client import get_supabase

class IngestionAgent(BaseAgent):
    """Agent responsible for article ingestion from various news sources"""
    
    def __init__(self, sources_config_path: str = "config/sources_config.json"):
        super().__init__("ingestion")
        
        self.news_ingest = NewsIngest()
        self.sources_config_path = sources_config_path
        self.sources_config = self.load_sources_config()
        
    def load_sources_config(self) -> Dict[str, Any]:
        """Load news sources configuration"""
        try:
            with open(self.sources_config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            self.logger.warning(f"Sources config not found at {self.sources_config_path}, using defaults")
            return self.get_default_sources()
    
    def get_default_sources(self) -> Dict[str, Any]:
        """Default news sources focused on Taiwan Strait"""
        return {
            "rss_feeds": [
                "https://feeds.reuters.com/reuters/world",
                "https://rss.cnn.com/rss/edition.world.rss",
                "https://feeds.bbci.co.uk/news/world/rss.xml"
            ],
            "taiwan_sources": [
                "https://focustaiwan.tw/rss/news.xml",
                "https://www.taipeitimes.com/xml/news.rss"
            ],
            "keywords": [
                "taiwan strait", "china taiwan", "taiwan military",
                "south china sea", "indo-pacific", "strait of taiwan"
            ]
        }
    
    async def run(self) -> Dict[str, Any]:
        """Main ingestion workflow"""
        
        # Get articles from last 4 hours to avoid duplicates
        cutoff_time = datetime.now() - timedelta(hours=4)
        
        ingested_count = 0
        error_count = 0
        
        # Process RSS feeds
        for feed_url in self.sources_config.get("rss_feeds", []):
            try:
                articles = await self.fetch_from_rss(feed_url, cutoff_time)
                ingested_count += await self.store_articles(articles)
            except Exception as e:
                self.logger.error(f"Error processing RSS feed {feed_url}: {e}")
                error_count += 1
        
        # Process Taiwan-specific sources
        for source_url in self.sources_config.get("taiwan_sources", []):
            try:
                articles = await self.fetch_from_source(source_url, cutoff_time)
                ingested_count += await self.store_articles(articles)
            except Exception as e:
                self.logger.error(f"Error processing Taiwan source {source_url}: {e}")
                error_count += 1
        
        # Keyword-based search from news APIs
        try:
            keyword_articles = await self.fetch_by_keywords(cutoff_time)
            ingested_count += await self.store_articles(keyword_articles)
        except Exception as e:
            self.logger.error(f"Error in keyword search: {e}")
            error_count += 1
        
        return {
            "ingested_count": ingested_count,
            "error_count": error_count,
            "sources_processed": len(self.sources_config.get("rss_feeds", [])) + 
                              len(self.sources_config.get("taiwan_sources", []))
        }
    
    async def fetch_from_rss(self, feed_url: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Fetch articles from RSS feed"""
        try:
            articles = await self.news_ingest.fetch_rss_articles(feed_url)
            
            # Filter by time and relevance
            filtered_articles = []
            for article in articles:
                if article.get('published_at') and article['published_at'] > cutoff_time:
                    if self.is_relevant_article(article):
                        filtered_articles.append(article)
            
            self.logger.info(f"Fetched {len(filtered_articles)} relevant articles from {feed_url}")
            return filtered_articles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch from RSS {feed_url}: {e}")
            return []
    
    async def fetch_from_source(self, source_url: str, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Fetch articles from specific news source"""
        try:
            articles = await self.news_ingest.fetch_source_articles(source_url)
            
            # Filter by time
            filtered_articles = [
                article for article in articles 
                if article.get('published_at') and article['published_at'] > cutoff_time
            ]
            
            self.logger.info(f"Fetched {len(filtered_articles)} articles from {source_url}")
            return filtered_articles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch from source {source_url}: {e}")
            return []
    
    async def fetch_by_keywords(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Fetch articles using keyword search"""
        all_articles = []
        
        for keyword in self.sources_config.get("keywords", []):
            try:
                articles = await self.news_ingest.search_by_keyword(keyword, cutoff_time)
                all_articles.extend(articles)
            except Exception as e:
                self.logger.error(f"Failed keyword search for '{keyword}': {e}")
        
        # Remove duplicates based on URL
        unique_articles = {}
        for article in all_articles:
            url = article.get('url')
            if url and url not in unique_articles:
                unique_articles[url] = article
        
        self.logger.info(f"Fetched {len(unique_articles)} unique articles via keyword search")
        return list(unique_articles.values())
    
    def is_relevant_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is relevant to Taiwan Strait monitoring"""
        content = f"{article.get('title', '')} {article.get('content', '')}".lower()
        
        # Taiwan Strait specific keywords
        relevant_keywords = [
            'taiwan', 'china', 'strait', 'south china sea', 'indo-pacific',
            'military', 'defense', 'navy', 'air force', 'exercise',
            'tension', 'conflict', 'diplomacy', 'trade war', 'sanctions'
        ]
        
        # Must contain at least 2 relevant keywords
        keyword_count = sum(1 for keyword in relevant_keywords if keyword in content)
        
        return keyword_count >= 2
    
    async def store_articles(self, articles: List[Dict[str, Any]]) -> int:
        """Store articles in Supabase database"""
        if not articles:
            return 0
        
        stored_count = 0
        
        for article in articles:
            try:
                # Check if article already exists
                existing = self.supabase.table("articles")\
                    .select("id")\
                    .eq("url", article.get("url"))\
                    .execute()
                
                if existing.data:
                    continue  # Skip duplicates
                
                # Prepare article data
                article_data = {
                    "source": article.get("source", "unknown"),
                    "url": article.get("url"),
                    "title": article.get("title"),
                    "content": article.get("content"),
                    "published_at": article.get("published_at"),
                    "language": article.get("language", "en")
                }
                
                # Insert into database
                result = self.supabase.table("articles").insert(article_data).execute()
                
                if result.data:
                    stored_count += 1
                    self.logger.debug(f"Stored article: {article.get('title', 'No title')[:50]}...")
                
            except Exception as e:
                self.logger.error(f"Failed to store article {article.get('url', 'unknown')}: {e}")
        
        self.logger.info(f"Successfully stored {stored_count} new articles")
        return stored_count
    
    async def cleanup_old_articles(self, days_old: int = 180):
        """Remove articles older than specified days"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_old)
            
            result = self.supabase.table("articles")\
                .delete()\
                .lt("created_at", cutoff_date.isoformat())\
                .execute()
            
            self.logger.info(f"Cleaned up {len(result.data)} old articles")
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old articles: {e}")

# CLI interface for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = IngestionAgent()
        result = await agent.safe_run()
        print(f"Ingestion result: {result}")
    
    asyncio.run(main())