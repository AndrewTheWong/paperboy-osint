#!/usr/bin/env python3
"""
Tagger Worker v2 - Pulls from Redis queue, tags articles, passes to embedding
"""

import logging
import asyncio
import json
from datetime import datetime
from celery import shared_task, Celery
from services.tagger import tag_article
from db.supabase_client_v2 import tag_article as tag_article_db
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
def run_tagger_pipeline(self, batch_size=10):
    """
    Pull articles from Redis tagging queue, tag them, and pass to embedding.
    Args:
        batch_size: Number of articles to process in each batch
    Returns:
        dict: Tagging results and performance metrics
    """
    try:
        logger.info(f"🚀 Starting tagger pipeline with batch size: {batch_size}")
        redis_queue = RedisQueue()
        
        # Get articles from tagging queue
        tagging_tasks = []
        for _ in range(batch_size):
            task = redis_queue.pop('tagging_queue')
            if task:
                tagging_tasks.append(task)
            else:
                break
        
        if not tagging_tasks:
            logger.info("📭 No articles in tagging queue")
            return {'status': 'success', 'articles_processed': 0, 'message': 'No articles to process'}
        
        logger.info(f"📋 Found {len(tagging_tasks)} articles in tagging queue")
        
        def tag_and_queue():
            start = datetime.now()
            processed_count = 0
            
            for task in tagging_tasks:
                try:
                    article_id = task['article_id']
                    content = task.get('translated_content') or task.get('content', '')
                    title = task.get('translated_title') or task.get('title', '')
                    
                    if not content:
                        logger.warning(f"⚠️ Skipping {article_id}: no content to tag")
                        continue
                    
                    logger.info(f"🏷️ Tagging article {article_id}: {title[:50]}...")
                    
                    # Tag the article (sync call)
                    tag_result = tag_article(content, title)
                    tags = tag_result.get('tags', [])
                    entities = tag_result.get('entities', [])
                    categories = tag_result.get('tag_categories', {})
                    
                    # Store tags in Supabase
                    success = tag_article_db(
                        article_id=article_id,
                        tags=tags,
                        entities=entities,
                        tag_categories=categories
                    )
                    
                    if success:
                        # Queue for embedding
                        embedding_task = {
                            'article_id': article_id,
                            'content': content,
                            'title': title,
                            'tags': tags,
                            'entities': entities
                        }
                        
                        # Send to Redis queue for embedding
                        redis_queue.push('embedding_queue', embedding_task)
                        logger.info(f"✅ Tagged and queued article {article_id} for embedding")
                        processed_count += 1
                    else:
                        logger.error(f"❌ Failed to store tags for article {article_id}")
                        
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
                'total_articles': len(tagging_tasks)
            }
        
        result = tag_and_queue()
        logger.info(f"✅ Tagger pipeline completed: {result['articles_processed']} articles in {result['duration_seconds']:.2f}s ({result['articles_per_second']:.2f}/sec)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Tagger pipeline failed: {e}")
        return {'status': 'error', 'error': str(e)} 