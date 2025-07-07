#!/usr/bin/env python3
"""
Tagging Worker for Paperboy Backend
Handles NER tagging and entity extraction from articles
"""

import logging
from celery import shared_task
from typing import List, Dict, Any
from services.tagger import tag_article_batch, tag_article

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def tag_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tag a single article with NER and entities
    
    Args:
        article: Article dictionary with title and content
        
    Returns:
        dict: Article with tags and entities added
    """
    try:
        logger.info(f"🏷️ Tagging article {article.get('article_id', 'unknown')}")
        
        # Extract text for tagging
        title = article.get('title_translated', article.get('title', ''))
        content = article.get('content_translated', article.get('body', article.get('content', '')))
        
        # Tag the article
        tag_result = tag_article(content, title)
        
        # Merge results
        article.update(tag_result)
        
        tags_count = len(article.get('tags', []))
        entities_count = len(article.get('entities', []))
        
        logger.info(f"✅ Tagged article {article.get('article_id', 'unknown')}: "
                   f"{tags_count} tags, {entities_count} entities")
        
        return article
        
    except Exception as e:
        logger.error(f"❌ Tagging failed for article {article.get('article_id', 'unknown')}: {e}")
        raise self.retry(countdown=60, max_retries=3)

@shared_task(bind=True, max_retries=3)
def tag_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tag a batch of articles
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        list: List of tagged articles
    """
    try:
        logger.info(f"🏷️ Starting batch tagging of {len(articles)} articles")
        
        # Prepare articles for batch tagging
        articles_for_tagging = []
        for article in articles:
            title = article.get('title_translated', article.get('title', ''))
            content = article.get('content_translated', article.get('body', article.get('content', '')))
            
            articles_for_tagging.append({
                'title': title,
                'content': content,
                'article_id': article.get('article_id', 'unknown')
            })
        
        # Tag articles in batch
        tagging_results = tag_article_batch(articles_for_tagging, text_key="content", title_key="title")
        
        # Merge results back to original articles
        for i, (article, tag_result) in enumerate(zip(articles, tagging_results)):
            article.update(tag_result)
        
        # Count statistics
        total_tags = sum(len(article.get('tags', [])) for article in articles)
        total_entities = sum(len(article.get('entities', [])) for article in articles)
        
        logger.info(f"✅ Batch tagging completed: {len(articles)} articles tagged")
        logger.info(f"📊 Total tags: {total_tags}, Total entities: {total_entities}")
        
        return articles
        
    except Exception as e:
        logger.error(f"❌ Batch tagging failed: {e}")
        raise self.retry(countdown=120, max_retries=3)

@shared_task(bind=True, max_retries=3)
def tag_from_queue(self, queue_name: str = "tagging_queue") -> Dict[str, Any]:
    """
    Tag articles from a Redis queue
    
    Args:
        queue_name: Name of the Redis queue to process
        
    Returns:
        dict: Tagging results
    """
    try:
        logger.info(f"🏷️ Starting tagging from queue: {queue_name}")
        
        # Import Redis queue functions
        from db.redis_queue import get_from_queue, get_queue_size, add_to_queue
        
        # Get queue size
        queue_size = get_queue_size(queue_name)
        logger.info(f"📊 Found {queue_size} articles in {queue_name}")
        
        if queue_size == 0:
            logger.warning(f"⚠️ No articles found in {queue_name}")
            return {"status": "no_data", "tagged_count": 0}
        
        # Process articles from queue
        articles_to_tag = []
        processed_count = 0
        max_articles = min(queue_size, 20)  # Process up to 20 articles at a time
        
        while processed_count < max_articles:
            article_data = get_from_queue(queue_name)
            if not article_data:
                break
            
            articles_to_tag.append(article_data)
            processed_count += 1
        
        if not articles_to_tag:
            logger.warning("⚠️ No articles retrieved from queue")
            return {"status": "no_articles", "tagged_count": 0}
        
        # Tag the batch
        tagged_articles = tag_articles_batch(articles_to_tag)
        
        # Store tagged articles back to queue for next step
        for article in tagged_articles:
            add_to_queue("embedding_queue", article)
        
        # Count statistics
        total_tags = sum(len(article.get('tags', [])) for article in tagged_articles)
        total_entities = sum(len(article.get('entities', [])) for article in tagged_articles)
        
        logger.info(f"✅ Tagging from queue completed: {len(tagged_articles)} articles tagged")
        logger.info(f"📊 Total tags: {total_tags}, Total entities: {total_entities}")
        
        return {
            "status": "success",
            "tagged_count": len(tagged_articles),
            "total_tags": total_tags,
            "total_entities": total_entities,
            "queue_processed": queue_name
        }
        
    except Exception as e:
        logger.error(f"❌ Tagging from queue failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def extract_entities_only(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract only entities without full tagging (faster for some use cases)
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        list: Articles with entities extracted
    """
    try:
        logger.info(f"🔍 Extracting entities from {len(articles)} articles")
        
        from services.tagger import extract_entities_batch
        
        # Extract entities in batch
        entities_results = extract_entities_batch(articles)
        
        # Merge results
        for i, (article, entities_result) in enumerate(zip(articles, entities_results)):
            article.update(entities_result)
        
        total_entities = sum(len(article.get('entities', [])) for article in articles)
        
        logger.info(f"✅ Entity extraction completed: {total_entities} entities found")
        
        return articles
        
    except Exception as e:
        logger.error(f"❌ Entity extraction failed: {e}")
        raise self.retry(countdown=60, max_retries=3)

@shared_task(bind=True, max_retries=3)
def tag_with_custom_categories(self, articles: List[Dict[str, Any]], 
                              custom_categories: List[str] = None) -> List[Dict[str, Any]]:
    """
    Tag articles with custom category focus
    
    Args:
        articles: List of article dictionaries
        custom_categories: List of custom categories to focus on
        
    Returns:
        list: Articles with custom tagging
    """
    try:
        logger.info(f"🏷️ Tagging {len(articles)} articles with custom categories: {custom_categories}")
        
        # Use custom tagging if categories provided
        if custom_categories:
            from services.tagger import tag_with_categories
            tagged_articles = tag_with_categories(articles, custom_categories)
        else:
            # Use standard batch tagging
            tagged_articles = tag_articles_batch(articles)
        
        logger.info(f"✅ Custom tagging completed: {len(tagged_articles)} articles processed")
        
        return tagged_articles
        
    except Exception as e:
        logger.error(f"❌ Custom tagging failed: {e}")
        raise self.retry(countdown=120, max_retries=3) 