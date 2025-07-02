#!/usr/bin/env python3
"""
Preprocessing task for article ingestion pipeline
"""

from celery import Celery
import logging
from typing import Optional
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('app.celery_config')

@celery_app.task(bind=True)
def preprocess_and_enqueue(self, article_id: str, title: str, body: str, 
                          region: Optional[str] = None, topic: Optional[str] = None, 
                          source_url: str = ""):
    """
    Preprocess article: clean HTML, embed, store to Supabase, queue for clustering
    
    Args:
        article_id: Unique article identifier
        title: Article title
        body: Article body (may contain HTML)
        region: Geographic region
        topic: Article topic
        source_url: Source URL
    """
    try:
        logger.info(f"🔄 Preprocessing article {article_id}")
        logger.info(f"📝 Article details: title='{title}', region='{region}', topic='{topic}'")
        
        # Import services
        from app.services.cleaning import clean_html_text
        from app.services.embedding import generate_embedding
        from app.services.redis_queue import push_to_clustering_queue
        
        # Clean HTML from body
        logger.info(f"🧹 Starting HTML cleaning for article {article_id}")
        cleaned_text = clean_html_text(body)
        logger.info(f"🧹 Cleaned text length: {len(cleaned_text)} characters")
        
        # Generate embedding
        logger.info(f"🔢 Starting embedding generation for article {article_id}")
        embedding = generate_embedding(cleaned_text)
        logger.info(f"🔢 Generated embedding: {len(embedding)} dimensions")
        
        # Push processed article data to clustering queue (storage happens after clustering)
        logger.info(f"📤 Pushing processed article {article_id} to clustering queue")
        
        # Create article data package for clustering
        article_data = {
            'article_id': article_id,
            'title': title,
            'raw_text': body,
            'cleaned_text': cleaned_text,
            'embedding': embedding.tolist() if hasattr(embedding, 'tolist') else embedding,
            'region': region,
            'topic': topic,
            'source_url': source_url
        }
        
        queue_success = push_to_clustering_queue(article_data)
        if queue_success:
            logger.info(f"📤 Successfully pushed article {article_id} to clustering queue")
        else:
            logger.error(f"❌ Failed to push article {article_id} to clustering queue")
        
        result = {
            "status": "success" if queue_success else "failed",
            "article_id": article_id,
            "cleaned_length": len(cleaned_text),
            "embedding_dimensions": len(embedding),
            "queued_for_clustering": queue_success
        }
        
        logger.info(f"✅ Preprocessing complete for article {article_id}: {result}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Error preprocessing article {article_id}: {e}")
        logger.exception("Full traceback:")
        raise self.retry(countdown=60, max_retries=3) 