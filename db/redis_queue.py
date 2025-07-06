#!/usr/bin/env python3
"""
Redis queue service for article processing pipeline
"""

import redis
import json
import logging
from typing import Optional, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Redis connection
_redis_client = None

def get_redis_client():
    """Get or create Redis client"""
    global _redis_client
    if _redis_client is None:
        try:
            _redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            # Test connection
            _redis_client.ping()
            logger.info("✅ Connected to Redis")
        except Exception as e:
            logger.error(f"❌ Error connecting to Redis: {e}")
            raise
    return _redis_client

def push_to_clustering_queue(article_data) -> bool:
    """
    Push article data to clustering queue
    
    Args:
        article_data: Article data dict or article ID string (for backward compatibility)
        
    Returns:
        bool: Success status
    """
    try:
        client = get_redis_client()
        
        # Handle both new data format (dict) and old format (string ID)
        if isinstance(article_data, dict):
            article_json = json.dumps(article_data)
            article_id = article_data.get('article_id', 'unknown')
        else:
            # Backward compatibility for string IDs
            article_json = article_data
            article_id = article_data
        
        result = client.lpush('clustering_queue', article_json)
        logger.info(f"📤 Pushed article {article_id} to clustering queue")
        return True
    except Exception as e:
        logger.error(f"❌ Error pushing to clustering queue: {e}")
        return False

def get_from_clustering_queue():
    """
    Get article data from clustering queue
    
    Returns:
        dict or str or None: Article data dict, article ID string (backward compatibility), or None if queue is empty
    """
    try:
        client = get_redis_client()
        result = client.rpop('clustering_queue')
        if result:
            # Try to parse as JSON (new format), fall back to string (old format)
            try:
                article_data = json.loads(result)
                article_id = article_data.get('article_id', 'unknown')
                logger.info(f"📥 Retrieved article data for {article_id} from clustering queue")
                return article_data
            except (json.JSONDecodeError, TypeError):
                # Old format - just a string ID
                logger.info(f"📥 Retrieved article {result} from clustering queue")
                return result
        return None
    except Exception as e:
        logger.error(f"❌ Error getting from clustering queue: {e}")
        return None

def get_queue_size() -> int:
    """
    Get size of clustering queue
    
    Returns:
        int: Number of items in queue
    """
    try:
        client = get_redis_client()
        size = client.llen('clustering_queue')
        return size
    except Exception as e:
        logger.error(f"❌ Error getting queue size: {e}")
        return 0

def clear_clustering_queue() -> bool:
    """
    Clear the clustering queue
    
    Returns:
        bool: Success status
    """
    try:
        client = get_redis_client()
        result = client.delete('clustering_queue')
        logger.info("🗑️ Cleared clustering queue")
        return True
    except Exception as e:
        logger.error(f"❌ Error clearing clustering queue: {e}")
        return False

def get_queue_items(limit: int = 10) -> List[str]:
    """
    Get items from queue without removing them
    
    Args:
        limit: Maximum number of items to retrieve
        
    Returns:
        List[str]: List of article IDs
    """
    try:
        client = get_redis_client()
        items = client.lrange('clustering_queue', 0, limit - 1)
        return items
    except Exception as e:
        logger.error(f"❌ Error getting queue items: {e}")
        return []

def push_batch_to_queue(article_ids: List[str]) -> int:
    """
    Push multiple article IDs to clustering queue
    
    Args:
        article_ids: List of article IDs to queue
        
    Returns:
        int: Number of items successfully queued
    """
    try:
        client = get_redis_client()
        if not article_ids:
            return 0
        
        # Use pipeline for batch operation
        pipe = client.pipeline()
        for article_id in article_ids:
            pipe.lpush('clustering_queue', article_id)
        
        results = pipe.execute()
        count = len([r for r in results if r > 0])
        
        logger.info(f"📤 Pushed {count} articles to clustering queue")
        return count
        
    except Exception as e:
        logger.error(f"❌ Error pushing batch to queue: {e}")
        return 0 