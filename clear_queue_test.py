#!/usr/bin/env python3
"""
Clear the clustering queue and add fresh test articles
"""

import logging
from app.services.redis_queue import clear_clustering_queue, get_queue_size

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_and_test():
    """Clear queue and report status"""
    
    print("🗑️ Clearing clustering queue...")
    
    # Check initial queue size
    initial_size = get_queue_size()
    print(f"📊 Initial queue size: {initial_size}")
    
    # Clear the queue
    success = clear_clustering_queue()
    if success:
        print("✅ Queue cleared successfully")
    else:
        print("❌ Failed to clear queue")
    
    # Check final queue size
    final_size = get_queue_size()
    print(f"📊 Final queue size: {final_size}")

if __name__ == "__main__":
    clear_and_test() 