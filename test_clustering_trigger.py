#!/usr/bin/env python3
"""
Test script to manually trigger clustering for the new flow
"""

import logging
from app.tasks.cluster import run_clustering

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_clustering_trigger():
    """Manually trigger clustering to test the new flow"""
    
    print("🔍 Manually triggering clustering task...")
    print("=" * 50)
    
    try:
        # Import and call the clustering task directly
        result = run_clustering()
        
        print("✅ Clustering task completed!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"❌ Clustering task failed: {e}")
        logger.exception("Full traceback:")

if __name__ == "__main__":
    test_clustering_trigger() 