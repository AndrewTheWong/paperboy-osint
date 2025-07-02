#!/usr/bin/env python3

"""
Clear Redis queues to remove failed/stuck tasks
"""

import redis
import sys
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_redis_queues():
    """Clear all Redis queues and tasks"""
    try:
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        
        # Test connection
        r.ping()
        logger.info("✅ Connected to Redis")
        
        # Get all keys
        all_keys = r.keys('*')
        logger.info(f"📋 Found {len(all_keys)} Redis keys")
        
        if not all_keys:
            logger.info("✅ Redis is already clean")
            return
        
        # Show what we're going to delete
        celery_keys = [k for k in all_keys if any(x in k for x in ['celery', 'default', '_kombu', 'unacked'])]
        queue_keys = [k for k in all_keys if any(x in k for x in ['clustering', 'preprocessing', 'summarization'])]
        
        logger.info(f"🔍 Found {len(celery_keys)} Celery keys")
        logger.info(f"🔍 Found {len(queue_keys)} Pipeline queue keys")
        
        if celery_keys:
            logger.info("🗑️  Celery keys to delete:")
            for key in celery_keys[:10]:  # Show first 10
                logger.info(f"   - {key}")
            if len(celery_keys) > 10:
                logger.info(f"   ... and {len(celery_keys) - 10} more")
        
        if queue_keys:
            logger.info("🗑️  Queue keys to delete:")
            for key in queue_keys:
                logger.info(f"   - {key}")
        
        # Ask for confirmation
        confirm = input("\n⚠️  Delete all Redis keys? (y/N): ").strip().lower()
        if confirm != 'y':
            logger.info("❌ Cancelled")
            return
        
        # Delete all keys
        deleted_count = 0
        for key in all_keys:
            try:
                r.delete(key)
                deleted_count += 1
            except Exception as e:
                logger.error(f"❌ Error deleting key {key}: {e}")
        
        logger.info(f"✅ Deleted {deleted_count} Redis keys")
        
        # Verify cleanup
        remaining_keys = r.keys('*')
        if remaining_keys:
            logger.warning(f"⚠️  {len(remaining_keys)} keys still remain:")
            for key in remaining_keys:
                logger.warning(f"   - {key}")
        else:
            logger.info("🧹 Redis is now completely clean")
            
    except redis.ConnectionError:
        logger.error("❌ Cannot connect to Redis. Is Redis running on localhost:6379?")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error clearing Redis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("🧹 Redis Queue Cleaner")
    print("=" * 50)
    clear_redis_queues() 