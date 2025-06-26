"""
NLP Processing Agent for StraitWatch

Processes untagged articles through NLP pipeline including:
- Named Entity Recognition
- Event Extraction  
- Relation Extraction
- Escalation Classification
"""

import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import json

from .base_agent import BaseAgent
from tagging.tagging_pipeline import tag_articles
from analytics.ner.named_entity_recognizer import NERProcessor
from analytics.ner.event_extraction import EventExtractor

class NLPAgent(BaseAgent):
    """Agent responsible for NLP processing of articles"""
    
    def __init__(self):
        super().__init__("nlp")
        
        # Initialize NLP processors
        self.ner_processor = None
        self.event_extractor = None
        self.escalation_classifier = None
        
        # Processing batch size
        self.batch_size = 10
        
    async def initialize_processors(self):
        """Initialize NLP processors lazily to avoid startup delays"""
        if not self.ner_processor:
            try:
                self.ner_processor = NERProcessor()
                self.logger.info("NER processor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize NER processor: {e}")
        
        if not self.event_extractor:
            try:
                self.event_extractor = EventExtractor()
                self.logger.info("Event extractor initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize event extractor: {e}")
        
        if not self.escalation_classifier:
            try:
                from analytics.inference.ensemble_predictor import EnsemblePredictor
                self.escalation_classifier = EnsemblePredictor()
                self.logger.info("Escalation classifier initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize escalation classifier: {e}")
    
    async def run(self) -> Dict[str, Any]:
        """Main NLP processing workflow"""
        
        # Initialize processors
        await self.initialize_processors()
        
        # Get unprocessed articles
        unprocessed_articles = await self.get_unprocessed_articles()
        
        if not unprocessed_articles:
            self.logger.info("No unprocessed articles found")
            return {"processed_count": 0}
        
        self.logger.info(f"Processing {len(unprocessed_articles)} articles")
        
        processed_count = 0
        error_count = 0
        
        # Process articles in batches
        for i in range(0, len(unprocessed_articles), self.batch_size):
            batch = unprocessed_articles[i:i + self.batch_size]
            
            try:
                await self.process_batch(batch)
                processed_count += len(batch)
                self.logger.info(f"Processed batch {i//self.batch_size + 1}/{(len(unprocessed_articles)-1)//self.batch_size + 1}")
                
            except Exception as e:
                self.logger.error(f"Error processing batch: {e}")
                error_count += len(batch)
        
        return {
            "processed_count": processed_count,
            "error_count": error_count,
            "total_articles": len(unprocessed_articles)
        }
    
    async def get_unprocessed_articles(self) -> List[Dict[str, Any]]:
        """Get articles that haven't been processed by NLP pipeline"""
        try:
            # Get articles without tags from the last 7 days
            cutoff_date = datetime.now() - timedelta(days=7)
            
            # Find articles without any tags
            result = self.supabase.table("articles")\
                .select("id, title, content, url, published_at")\
                .gte("created_at", cutoff_date.isoformat())\
                .is_("content", "not.null")\
                .execute()
            
            articles = result.data
            
            # Filter out articles that already have tags
            unprocessed = []
            for article in articles:
                existing_tags = self.supabase.table("article_tags")\
                    .select("id")\
                    .eq("article_id", article["id"])\
                    .execute()
                
                if not existing_tags.data:
                    unprocessed.append(article)
            
            self.logger.info(f"Found {len(unprocessed)} unprocessed articles")
            return unprocessed
            
        except Exception as e:
            self.logger.error(f"Failed to get unprocessed articles: {e}")
            return []
    
    async def process_batch(self, articles: List[Dict[str, Any]]):
        """Process a batch of articles through NLP pipeline"""
        
        for article in articles:
            try:
                await self.process_single_article(article)
            except Exception as e:
                self.logger.error(f"Failed to process article {article['id']}: {e}")
    
    async def process_single_article(self, article: Dict[str, Any]):
        """Process a single article through complete NLP pipeline"""
        
        article_id = article["id"]
        content = article.get("content", "")
        title = article.get("title", "")
        
        if not content:
            self.logger.warning(f"Article {article_id} has no content")
            return
        
        # Combine title and content for processing
        full_text = f"{title}. {content}"
        
        # Step 1: Named Entity Recognition
        entities = await self.extract_entities(full_text)
        await self.store_entities(article_id, entities)
        
        # Step 2: Event Extraction
        events = await self.extract_events(full_text, article)
        await self.store_events(article_id, events)
        
        # Step 3: Escalation Classification
        escalation_score = await self.classify_escalation(full_text)
        await self.store_escalation(article_id, escalation_score)
        
        # Step 4: Basic tagging using existing pipeline
        tagged_articles = tag_articles([article])
        if tagged_articles:
            await self.store_basic_tags(article_id, tagged_articles[0])
        
        self.logger.debug(f"Completed NLP processing for article {article_id}")
    
    async def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract named entities from text"""
        try:
            if self.ner_processor:
                entities = self.ner_processor.extract_entities(text)
                return entities
            else:
                self.logger.warning("NER processor not available")
                return []
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return []
    
    async def extract_events(self, text: str, article: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract events from text"""
        try:
            if self.event_extractor:
                events = self.event_extractor.extract_events(text)
                
                # Enrich events with article metadata
                for event in events:
                    event["article_date"] = article.get("published_at")
                    event["source_url"] = article.get("url")
                
                return events
            else:
                self.logger.warning("Event extractor not available")
                return []
        except Exception as e:
            self.logger.error(f"Event extraction failed: {e}")
            return []
    
    async def classify_escalation(self, text: str) -> Dict[str, Any]:
        """Classify escalation level of the text"""
        try:
            if self.escalation_classifier:
                prediction = self.escalation_classifier.predict_escalation(text)
                return {
                    "escalation_level": prediction.get("escalation_level", "low"),
                    "confidence": prediction.get("confidence", 0.0),
                    "escalation_score": prediction.get("escalation_score", 0.0)
                }
            else:
                self.logger.warning("Escalation classifier not available")
                return {"escalation_level": "unknown", "confidence": 0.0, "escalation_score": 0.0}
        except Exception as e:
            self.logger.error(f"Escalation classification failed: {e}")
            return {"escalation_level": "unknown", "confidence": 0.0, "escalation_score": 0.0}
    
    async def store_entities(self, article_id: str, entities: List[Dict[str, Any]]):
        """Store extracted entities in database"""
        for entity in entities:
            try:
                tag_data = {
                    "article_id": article_id,
                    "tag_type": "entity",
                    "tag_value": entity.get("text", ""),
                    "confidence": entity.get("confidence", 0.0),
                    "metadata": {
                        "entity_type": entity.get("label", ""),
                        "start_pos": entity.get("start", 0),
                        "end_pos": entity.get("end", 0)
                    }
                }
                
                self.supabase.table("article_tags").insert(tag_data).execute()
                
            except Exception as e:
                self.logger.error(f"Failed to store entity: {e}")
    
    async def store_events(self, article_id: str, events: List[Dict[str, Any]]):
        """Store extracted events in database"""
        for event in events:
            try:
                # Store in events table
                event_data = {
                    "article_id": article_id,
                    "event_type": event.get("event_type", "unknown"),
                    "event_date": event.get("event_date") or event.get("article_date"),
                    "location": event.get("location", ""),
                    "severity_score": event.get("severity", 0.0),
                    "confidence": event.get("confidence", 0.0),
                    "description": event.get("description", "")
                }
                
                self.supabase.table("events").insert(event_data).execute()
                
                # Also store as tag
                tag_data = {
                    "article_id": article_id,
                    "tag_type": "event",
                    "tag_value": event.get("event_type", "unknown"),
                    "confidence": event.get("confidence", 0.0),
                    "metadata": event
                }
                
                self.supabase.table("article_tags").insert(tag_data).execute()
                
            except Exception as e:
                self.logger.error(f"Failed to store event: {e}")
    
    async def store_escalation(self, article_id: str, escalation: Dict[str, Any]):
        """Store escalation classification in database"""
        try:
            tag_data = {
                "article_id": article_id,
                "tag_type": "escalation",
                "tag_value": escalation.get("escalation_level", "unknown"),
                "confidence": escalation.get("confidence", 0.0),
                "metadata": escalation
            }
            
            self.supabase.table("article_tags").insert(tag_data).execute()
            
        except Exception as e:
            self.logger.error(f"Failed to store escalation: {e}")
    
    async def store_basic_tags(self, article_id: str, tagged_article: Dict[str, Any]):
        """Store basic tags from tagging pipeline"""
        try:
            tags = tagged_article.get("tags", [])
            
            for tag in tags:
                tag_data = {
                    "article_id": article_id,
                    "tag_type": "keyword",
                    "tag_value": tag,
                    "confidence": 0.8,  # Default confidence for keyword tags
                    "metadata": {
                        "method": "keyword_matching",
                        "needs_review": tagged_article.get("needs_review", False)
                    }
                }
                
                self.supabase.table("article_tags").insert(tag_data).execute()
                
        except Exception as e:
            self.logger.error(f"Failed to store basic tags: {e}")

# CLI interface for standalone testing
if __name__ == "__main__":
    import asyncio
    
    async def main():
        agent = NLPAgent()
        result = await agent.safe_run()
        print(f"NLP processing result: {result}")
    
    asyncio.run(main())