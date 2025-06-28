"""
Article Ingest Agent for StraitWatch
Uses the fixed comprehensive NewsIngest pipeline
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent

class ArticleIngestAgent(BaseAgent):
    """Agent responsible for ingesting real articles using the fixed pipeline"""
    
    def __init__(self):
        super().__init__("article_ingest_agent")
        
        # Taiwan Strait keywords for filtering relevance
        self.keywords = [
            "taiwan", "china", "strait", "beijing", "taipei", 
            "cross-strait", "military", "defense", "navy", "air force",
            "south china sea", "indo-pacific", "pla", "escalation",
            "tensions", "conflict", "diplomacy", "trade war"
        ]
        
    async def run(self) -> Dict[str, Any]:
        """Main ingestion workflow using the fixed comprehensive pipeline"""
        self.logger.info("Starting StraitWatch article ingestion with fixed pipeline")
        
        try:
            # Import the fixed comprehensive pipeline
            from pipelines.Ingest.NewsIngest import run_comprehensive_ingest
            
            # Run the fixed ingestion pipeline
            self.logger.info("Running comprehensive article ingestion...")
            result = await run_comprehensive_ingest(max_sources=25)
            
            # Extract statistics
            stats = result.get('pipeline_stats', {})
            scraped = stats.get('articles_scraped', 0)
            uploaded = stats.get('articles_uploaded', 0)
            duration = result.get('duration_seconds', 0)
            success_rate = result.get('success_rate', 0)
            
            # Filter articles for relevance
            relevant_count = await self.filter_relevant_articles()
            
            # Get total article count
            total_count = await self.get_total_article_count()
            
            self.logger.info(f"Ingestion completed: {uploaded} articles uploaded, {relevant_count} marked relevant")
            
            return {
                "success": True,
                "articles_scraped": scraped,
                "articles_uploaded": uploaded,
                "total_articles": total_count,
                "relevant_articles": relevant_count,
                "duration_seconds": duration,
                "success_rate": success_rate,
                "sources_processed": stats.get('sources_processed', 0),
                "message": f"Successfully ingested {uploaded} articles in {duration:.1f}s"
            }
            
        except Exception as e:
            self.logger.error(f"Article ingestion failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "articles_scraped": 0,
                "articles_uploaded": 0,
                "total_articles": 0,
                "relevant_articles": 0
            }
    
    async def get_total_article_count(self) -> int:
        """Get total count of articles in database"""
        try:
            result = self.supabase.table("articles")\
                .select("id", count="exact")\
                .execute()
            
            return result.count or 0
            
        except Exception as e:
            self.logger.error(f"Error getting total article count: {e}")
            return 0
    
    async def filter_relevant_articles(self) -> int:
        """Filter existing articles for Taiwan Strait relevance"""
        try:
            # Get recent unprocessed articles
            articles = self.supabase.table("articles")\
                .select("id", "title", "content")\
                .is_("relevant", "null")\
                .limit(100)\
                .execute()
            
            if not articles.data:
                return 0
            
            relevant_count = 0
            
            for article in articles.data:
                if await self.is_relevant_article(article):
                    # Mark as relevant for StraitWatch
                    self.supabase.table("articles")\
                        .update({"relevant": True})\
                        .eq("id", article["id"])\
                        .execute()
                    relevant_count += 1
                else:
                    # Mark as not relevant
                    self.supabase.table("articles")\
                        .update({"relevant": False})\
                        .eq("id", article["id"])\
                        .execute()
            
            self.logger.info(f"Marked {relevant_count} articles as relevant to Taiwan Strait")
            return relevant_count
            
        except Exception as e:
            self.logger.error(f"Relevance filtering error: {e}")
            return 0
    
    async def is_relevant_article(self, article: Dict[str, Any]) -> bool:
        """Check if article is relevant to Taiwan Strait"""
        title = article.get('title', '') or ''
        content = article.get('content', '') or ''
        text_to_check = f"{title} {content[:1000]}"
        text_lower = text_to_check.lower()
        
        # Must contain Taiwan Strait related keywords
        relevant_keywords = 0
        for keyword in self.keywords:
            if keyword.lower() in text_lower:
                relevant_keywords += 1
                
        # At least 2 relevant keywords for high confidence
        return relevant_keywords >= 2

async def main():
    """Test the article ingest agent"""
    agent = ArticleIngestAgent()
    result = await agent.run()
    print(f"Ingestion result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 