#!/usr/bin/env python3
"""
Embedder Worker v2 - Pulls from Redis queue, embeds articles, stores in Supabase
"""

import logging
import asyncio
import json
from datetime import datetime
from celery import shared_task, Celery
from services.embedder import generate_embedding
from db.supabase_client_v2 import embed_article
from db.redis_queue import RedisQueue

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('config.celery_config')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _run_async_with_proper_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@shared_task(bind=True)
def run_embedder_pipeline(self, batch_size=10):
    """
    Pull articles from Redis embedding queue, embed them, and store in Supabase.
    Args:
        batch_size: Number of articles to process in each batch
    Returns:
        dict: Embedding results and performance metrics
    """
    try:
        logger.info(f"🚀 Starting embedder pipeline with batch size: {batch_size}")
        redis_queue = RedisQueue()
        
        # Get articles from embedding queue
        embedding_tasks = []
        for _ in range(batch_size):
            task = redis_queue.pop('embedding_queue')
            if task:
                embedding_tasks.append(task)
            else:
                break
        
        if not embedding_tasks:
            logger.info("📭 No articles in embedding queue")
            return {'status': 'success', 'articles_processed': 0, 'message': 'No articles to process'}
        
        logger.info(f"📋 Found {len(embedding_tasks)} articles in embedding queue")
        
        def embed_and_store():
            start = datetime.now()
            processed_count = 0
            
            for task in embedding_tasks:
                try:
                    article_id = task['article_id']
                    content = task.get('content', '')
                    title = task.get('title', '')
                    
                    if not content:
                        logger.warning(f"⚠️ Skipping {article_id}: no content to embed")
                        continue
                    
                    logger.info(f"🔢 Embedding article {article_id}: {title[:50]}...")
                    
                    # Get tags and entities if available
                    tags = task.get('tags', [])
                    entities = task.get('entities', [])
                    
                    # Generate enhanced embedding for better topic clustering
                    from services.embedder import generate_enhanced_embedding
                    embedding = generate_enhanced_embedding(
                        title=title,
                        content=content,
                        tags=tags,
                        entities=entities,
                        use_multimodal=True
                    )
                    
                    if embedding:
                        # Store embedding in Supabase
                        success = embed_article(
                            article_id=article_id,
                            embedding=embedding
                        )
                        
                        if success:
                            logger.info(f"✅ Embedded and stored article {article_id}")
                            processed_count += 1
                        else:
                            logger.error(f"❌ Failed to store embedding for article {article_id}")
                    else:
                        logger.error(f"❌ Failed to generate embedding for article {article_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing article {task.get('article_id', 'unknown')}: {e}")
                    continue
            
            duration = (datetime.now() - start).total_seconds()
            rate = processed_count / duration if duration > 0 else 0
            
            return {
                'status': 'success',
                'articles_processed': processed_count,
                'duration_seconds': duration,
                'articles_per_second': rate,
                'total_articles': len(embedding_tasks)
            }
        
        result = embed_and_store()
        logger.info(f"✅ Embedder pipeline completed: {result['articles_processed']} articles in {result['duration_seconds']:.2f}s ({result['articles_per_second']:.2f}/sec)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Embedder pipeline failed: {e}")
        return {'status': 'error', 'error': str(e)} 