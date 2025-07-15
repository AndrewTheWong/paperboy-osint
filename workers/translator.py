#!/usr/bin/env python3
"""
Translation Worker for Paperboy Backend
Handles translation of articles from various languages to English
"""

import logging
from celery import shared_task
from typing import List, Dict, Any
from services.translator import translate_article_simple, translate_articles_batch_simple

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def translate_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Translate a single article
    
    Args:
        article: Article dictionary with title and content
        
    Returns:
        dict: Translated article with original and translated fields
    """
    try:
        logger.info(f"🔄 Translating article {article.get('article_id', 'unknown')}")
        
        # Translate the article
        translated_article = translate_article_simple(article)
        
        logger.info(f"✅ Translated article {article.get('article_id', 'unknown')}: "
                   f"{translated_article.get('title_language', 'en')}→en, "
                   f"{translated_article.get('content_language', 'en')}→en")
        
        return translated_article
        
    except Exception as e:
        logger.error(f"❌ Translation failed for article {article.get('article_id', 'unknown')}: {e}")
        raise self.retry(countdown=60, max_retries=3)

@shared_task(bind=True, max_retries=3)
def translate_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Translate a batch of articles
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        list: List of translated articles
    """
    try:
        logger.info(f"🔄 Starting batch translation of {len(articles)} articles")
        
        # Translate articles in batch
        translated_articles = translate_articles_batch_simple(articles)
        
        # Count translation statistics
        translated_count = 0
        language_stats = {}
        
        for article in translated_articles:
            title_lang = article.get('title_language', 'en')
            content_lang = article.get('content_language', 'en')
            
            if title_lang != 'en' or content_lang != 'en':
                translated_count += 1
            
            # Track language statistics
            language_stats[title_lang] = language_stats.get(title_lang, 0) + 1
            language_stats[content_lang] = language_stats.get(content_lang, 0) + 1
        
        logger.info(f"✅ Batch translation completed: {translated_count}/{len(articles)} articles translated")
        logger.info(f"📊 Language distribution: {language_stats}")
        
        return translated_articles
        
    except Exception as e:
        logger.error(f"❌ Batch translation failed: {e}")
        raise self.retry(countdown=120, max_retries=3)

@shared_task(bind=True, max_retries=3)
def translate_from_queue(self, queue_name: str = "translation_queue") -> Dict[str, Any]:
    """
    Translate articles from a Redis queue and store to Supabase
    
    Args:
        queue_name: Name of the Redis queue to process
        
    Returns:
        dict: Translation results
    """
    try:
        logger.info(f"🔄 Starting translation from queue: {queue_name}")
        
        # Import Redis queue functions and Supabase client
        from db.redis_queue import get_from_queue, get_queue_size, add_to_queue
        from db.supabase_client_v2 import translate_article
        
        # Get queue size
        queue_size = get_queue_size(queue_name)
        logger.info(f"📊 Found {queue_size} articles in {queue_name}")
        
        if queue_size == 0:
            logger.warning(f"⚠️ No articles found in {queue_name}")
            return {"status": "no_data", "translated_count": 0}
        
        # Process articles from queue
        articles_to_translate = []
        processed_count = 0
        max_articles = min(queue_size, 20)  # Process up to 20 articles at a time
        
        while processed_count < max_articles:
            article_data = get_from_queue(queue_name)
            if not article_data:
                break
            
            articles_to_translate.append(article_data)
            processed_count += 1
        
        if not articles_to_translate:
            logger.warning("⚠️ No articles retrieved from queue")
            return {"status": "no_articles", "translated_count": 0}
        
        # Translate the batch
        translated_articles = translate_articles_batch_simple(articles_to_translate)
        
        # Store translated articles to Supabase and pass to next queue
        stored_count = 0
        for article in translated_articles:
            article_id = article.get('article_id')
            if article_id:
                # Store translation data to Supabase
                success = translate_article(
                    article_id=article_id,
                    translated_text=article.get('content_translated', ''),
                    translated_title=article.get('title_translated', ''),
                    source_language=article.get('content_language', 'en'),
                    target_language='en'
                )
                if success:
                    stored_count += 1
                    # Pass to next queue
                    add_to_queue("tagging_queue", article)
        
        logger.info(f"✅ Translation from queue completed: {stored_count}/{len(translated_articles)} articles translated and stored")
        
        return {
            "status": "success",
            "translated_count": stored_count,
            "total_articles": len(translated_articles),
            "queue_processed": queue_name
        }
        
    except Exception as e:
        logger.error(f"❌ Translation from queue failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def detect_languages_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Detect languages for a batch of articles without translating
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        list: Articles with language detection added
    """
    try:
        logger.info(f"🔍 Detecting languages for {len(articles)} articles")
        
        from services.translator import get_translation_service
        
        service = get_translation_service()
        language_stats = {}
        
        for article in articles:
            title = article.get('title', '')
            content = article.get('body', article.get('content', ''))
            
            # Detect languages
            title_lang = service.detect_language(title)
            content_lang = service.detect_language(content)
            
            # Add language info to article
            article['title_language'] = title_lang
            article['content_language'] = content_lang
            article['title_original'] = title
            article['content_original'] = content
            
            # Track statistics
            language_stats[title_lang] = language_stats.get(title_lang, 0) + 1
            language_stats[content_lang] = language_stats.get(content_lang, 0) + 1
        
        logger.info(f"✅ Language detection completed: {language_stats}")
        
        return articles
        
    except Exception as e:
        logger.error(f"❌ Language detection failed: {e}")
        raise self.retry(countdown=60, max_retries=3) 