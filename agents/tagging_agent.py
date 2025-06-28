"""
Tagging Agent for StraitWatch
Performs NLP analysis on articles using existing tagging pipeline
"""

import asyncio
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional

from .base_agent import BaseAgent
from tagging.comprehensive_tagging_pipeline import ComprehensiveTaggingPipeline
from tagging.enhanced_tagging_layer import EnhancedTaggingLayer

class TaggingAgent(BaseAgent):
    """Agent responsible for NLP tagging of articles using existing pipeline"""
    
    def __init__(self):
        super().__init__("tagging_agent")
        
        # Initialize existing tagging components
        self.comprehensive_pipeline = ComprehensiveTaggingPipeline()
        self.enhanced_tagger = EnhancedTaggingLayer()
        
        self.batch_size = 5
        
        # For test compatibility
        self.event_patterns = {
            'military_exercise': ['exercise', 'drill', 'maneuver', 'training'],
            'diplomatic_meeting': ['meeting', 'summit', 'talks', 'diplomacy'],
            'trade_dispute': ['trade', 'tariff', 'sanctions', 'embargo'],
            'territorial_dispute': ['territory', 'claims', 'sovereignty', 'waters'],
            'military_movement': ['deployment', 'ships', 'aircraft', 'forces']
        }
        
        self.escalation_keywords = [
            'conflict', 'tension', 'threat', 'aggressive', 'hostile',
            'military', 'invasion', 'attack', 'war', 'crisis',
            'escalation', 'provocation', 'confrontation', 'dispute'
        ]
        
    async def run(self) -> Dict[str, Any]:
        """Main tagging workflow using existing pipeline"""
        self.logger.info("Starting NLP tagging using existing pipeline")
        
        # Get unprocessed articles
        unprocessed = await self.get_unprocessed_articles()
        
        if not unprocessed:
            self.logger.info("No unprocessed articles found")
            return {"success": True, "processed_count": 0}
        
        self.logger.info(f"Processing {len(unprocessed)} articles")
        
        processed_count = 0
        error_count = 0
        
        # Process in batches
        for i in range(0, len(unprocessed), self.batch_size):
            batch = unprocessed[i:i + self.batch_size]
            
            for article in batch:
                try:
                    await self.process_article_with_pipeline(article)
                    processed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing article {article['id']}: {e}")
                    error_count += 1
            
            # Small delay between batches
            await asyncio.sleep(0.5)
        
        return {
            "success": True,
            "processed_count": processed_count,
            "error_count": error_count,
            "total_articles": len(unprocessed)
        }
    
    async def get_unprocessed_articles(self) -> List[Dict[str, Any]]:
        """Get articles without NLP tags"""
        try:
            # Get articles from last 3 days without tags
            cutoff_date = datetime.now() - timedelta(days=3)
            
            # First get all articles  
            articles_result = self.supabase.table("articles")\
                .select("id, title, content, source, published_at")\
                .gte("created_at", cutoff_date.isoformat())\
                .neq("content", None)\
                .order("created_at", desc=True)\
                .limit(100)\
                .execute()
            
            if not articles_result.data:
                return []
            
            # Filter out articles that already have tags
            unprocessed = []
            for article in articles_result.data:
                tags_result = self.supabase.table("article_tags")\
                    .select("id")\
                    .eq("article_id", article["id"])\
                    .limit(1)\
                    .execute()
                
                if not tags_result.data:
                    unprocessed.append(article)
            
            return unprocessed
            
        except Exception as e:
            self.logger.error(f"Error fetching unprocessed articles: {e}")
            return []
    
    async def process_article_with_pipeline(self, article: Dict[str, Any]):
        """Process article using existing tagging pipeline"""
        article_id = article["id"]
        title = article.get("title", "")
        content = article.get("content", "")
        
        if not content:
            return
        
        try:
            # Use comprehensive tagging pipeline
            comprehensive_results = await asyncio.to_thread(
                self.comprehensive_pipeline.process_article,
                {"title": title, "content": content}
            )
            
            # Use enhanced tagging layer
            enhanced_results = await asyncio.to_thread(
                self.enhanced_tagger.process_article,
                {"title": title, "content": content, "source": article.get("source", "")}
            )
            
            # Store results in database
            await self.store_tagging_results(article_id, comprehensive_results, enhanced_results)
            
            self.logger.debug(f"Tagged article {article_id} using existing pipeline")
            
        except Exception as e:
            self.logger.error(f"Error tagging article {article_id}: {e}")
            raise
    
    async def store_tagging_results(self, article_id: str, comprehensive_results: Dict, enhanced_results: Dict):
        """Store tagging results in database"""
        try:
            # Store comprehensive results
            if comprehensive_results:
                await self.store_tag(article_id, "comprehensive_tags", comprehensive_results)
            
            # Store enhanced results  
            if enhanced_results:
                await self.store_tag(article_id, "enhanced_tags", enhanced_results)
                
                # Extract specific components for separate storage
                if 'entities' in enhanced_results:
                    await self.store_entities(article_id, enhanced_results['entities'])
                
                if 'events' in enhanced_results:
                    await self.store_events(article_id, enhanced_results['events'])
                
                if 'sentiment' in enhanced_results:
                    await self.store_sentiment(article_id, enhanced_results['sentiment'])
                
                if 'escalation_score' in enhanced_results:
                    await self.store_escalation_score(article_id, enhanced_results['escalation_score'])
            
        except Exception as e:
            self.logger.error(f"Error storing tagging results for article {article_id}: {e}")
            raise
    
    async def store_tag(self, article_id: str, tag_type: str, tag_data: Dict):
        """Store a tag in the article_tags table"""
        try:
            result = self.supabase.table("article_tags").insert({
                "article_id": article_id,
                "tag_type": tag_type,
                "tag_data": json.dumps(tag_data),
                "created_at": datetime.now().isoformat()
            }).execute()
            
            return result.data[0] if result.data else None
            
        except Exception as e:
            self.logger.error(f"Error storing tag {tag_type} for article {article_id}: {e}")
            raise
    
    async def store_entities(self, article_id: str, entities: List[Dict]):
        """Store entities in the entities table"""
        try:
            for entity in entities:
                self.supabase.table("entities").insert({
                    "article_id": article_id,
                    "entity_text": entity.get("text", ""),
                    "entity_type": entity.get("label", ""),
                    "confidence": entity.get("confidence", 0.0),
                    "start_char": entity.get("start", 0),
                    "end_char": entity.get("end", 0),
                    "created_at": datetime.now().isoformat()
                }).execute()
                
        except Exception as e:
            self.logger.error(f"Error storing entities for article {article_id}: {e}")
    
    async def store_events(self, article_id: str, events: List[Dict]):
        """Store events in the events table"""
        try:
            for event in events:
                # Handle both dict format and enhanced Event object format
                if hasattr(event, '__dict__'):
                    # Convert Event object to dict
                    event_dict = event.__dict__ if hasattr(event, '__dict__') else {}
                    event_data = {
                        "article_id": article_id,
                        "event_type": event_dict.get("event_type", ""),
                        "event_description": event_dict.get("description", ""),
                        "escalation_score": event_dict.get("escalation_score", 0.0),
                        "escalation_analysis": event_dict.get("escalation_analysis", ""),
                        "participants": json.dumps(event_dict.get("participants", [])),
                        "location": event_dict.get("location", ""),
                        "date_extracted": event_dict.get("datetime", ""),
                        "severity_rating": event_dict.get("severity_rating", "medium"),
                        "keywords": json.dumps(event_dict.get("keywords", [])),
                        "confidence": event_dict.get("confidence_score", 0.0),
                        "created_at": datetime.now().isoformat()
                    }
                else:
                    # Handle dict format
                    event_data = {
                        "article_id": article_id,
                        "event_type": event.get("type", event.get("event_type", "")),
                        "event_description": event.get("description", ""),
                        "escalation_score": event.get("escalation_score", 0.0),
                        "escalation_analysis": event.get("escalation_analysis", ""),
                        "participants": json.dumps(event.get("participants", [])),
                        "location": event.get("location", ""),
                        "date_extracted": event.get("date", event.get("datetime", "")),
                        "severity_rating": event.get("severity_rating", "medium"),
                        "keywords": json.dumps(event.get("keywords", [])),
                        "confidence": event.get("confidence", event.get("confidence_score", 0.0)),
                        "created_at": datetime.now().isoformat()
                    }
                
                self.supabase.table("events").insert(event_data).execute()
                
        except Exception as e:
            self.logger.error(f"Error storing events for article {article_id}: {e}")
    
    async def store_sentiment(self, article_id: str, sentiment: Dict):
        """Store sentiment in the article_sentiment table"""
        try:
            self.supabase.table("article_sentiment").insert({
                "article_id": article_id,
                "sentiment_score": sentiment.get("score", 0.0),
                "sentiment_label": sentiment.get("label", "neutral"),
                "confidence": sentiment.get("confidence", 0.0),
                "created_at": datetime.now().isoformat()
            }).execute()
            
        except Exception as e:
            self.logger.error(f"Error storing sentiment for article {article_id}: {e}")
    
    async def store_escalation_score(self, article_id: str, escalation_score: float):
        """Store escalation score as a tag"""
        try:
            await self.store_tag(article_id, "escalation", {
                "score": escalation_score,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Error storing escalation score for article {article_id}: {e}")

async def main():
    """Test the tagging agent"""
    agent = TaggingAgent()
    result = await agent.run()
    print(f"Tagging result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 