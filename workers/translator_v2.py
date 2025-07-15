#!/usr/bin/env python3
"""
Translator Worker v2 - Pulls from Supabase articles table
Translates articles and passes to tagging pipeline
"""

import logging
import asyncio
from datetime import datetime
from celery import shared_task, Celery
from services.translator import TranslationService
from db.supabase_client_v2 import get_unprocessed_articles, translate_article
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
def run_translator_pipeline(self, batch_size=10):
    """
    Pull unprocessed articles from Supabase, translate them, and pass to tagging.
    Args:
        batch_size: Number of articles to process in each batch
    Returns:
        dict: Translation results and performance metrics
    """
    try:
        logger.info(f"🚀 Starting translator pipeline with batch size: {batch_size}")
        translator = TranslationService()
        redis_queue = RedisQueue()
        
        # Get unprocessed articles from Supabase
        unprocessed_articles = get_unprocessed_articles(limit=batch_size)
        if not unprocessed_articles:
            logger.info("📭 No unprocessed articles found")
            return {'status': 'success', 'articles_processed': 0, 'message': 'No articles to process'}
        
        logger.info(f"📋 Found {len(unprocessed_articles)} unprocessed articles")
        
        async def translate_and_queue():
            start = datetime.now()
            processed_count = 0
            
            for article in unprocessed_articles:
                try:
                    article_id = article['id']
                    content = article['text_content']
                    title = article['title']
                    language = article.get('language', 'unknown')
                    
                    # Skip if already in English
                    if language == 'en':
                        logger.info(f"⏭️ Skipping {article_id}: already in English")
                        continue
                    
                    logger.info(f"🔄 Translating article {article_id}: {title[:50]}...")
                    
                    # Translate content and title (synchronous calls)
                    translated_content = translator.translate_text(content, source_lang=language, target_lang='en')
                    translated_title = translator.translate_text(title, source_lang=language, target_lang='en')
                    
                    # Store translation in Supabase
                    success = translate_article(
                        article_id=article_id,
                        translated_text=translated_content,
                        translated_title=translated_title,
                        source_language=language,
                        target_language='en'
                    )
                    
                    if success:
                        # Queue for tagging
                        tagging_task = {
                            'article_id': article_id,
                            'translated_content': translated_content,
                            'translated_title': translated_title,
                            'original_language': language
                        }
                        
                        # Send to Redis queue for tagging
                        redis_queue.push('tagging_queue', tagging_task)
                        logger.info(f"✅ Translated and queued article {article_id} for tagging")
                        processed_count += 1
                    else:
                        logger.error(f"❌ Failed to store translation for article {article_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing article {article.get('id', 'unknown')}: {e}")
                    continue
            
            duration = (datetime.now() - start).total_seconds()
            rate = processed_count / duration if duration > 0 else 0
            
            return {
                'status': 'success',
                'articles_processed': processed_count,
                'duration_seconds': duration,
                'articles_per_second': rate,
                'total_articles': len(unprocessed_articles)
            }
        
        result = _run_async_with_proper_loop(translate_and_queue())
        logger.info(f"✅ Translator pipeline completed: {result['articles_processed']} articles in {result['duration_seconds']:.2f}s ({result['articles_per_second']:.2f}/sec)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Translator pipeline failed: {e}")
        return {'status': 'error', 'error': str(e)} 