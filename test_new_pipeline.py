#!/usr/bin/env python3
"""
Test script to monitor the new pipeline flow
"""

import requests
import time
import logging
from app.services.redis_queue import get_queue_size, get_queue_items
from app.services.supabase import get_articles_count

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def monitor_new_pipeline():
    """Monitor the new pipeline flow"""
    
    print("🚀 Testing New Pipeline Flow: Preprocessing → Clustering → Storage")
    print("=" * 60)
    
    # Test API health
    try:
        response = requests.get("http://localhost:8000/health")
        print(f"✅ API Health: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ API Health check failed: {e}")
        return
    
    # Monitor queue and database for 60 seconds
    for i in range(12):  # 12 iterations * 5 seconds = 60 seconds
        print(f"\n--- Monitor Cycle {i+1}/12 ---")
        
        # Check clustering queue
        try:
            queue_size = get_queue_size()
            print(f"📊 Clustering Queue Size: {queue_size}")
            
            if queue_size > 0:
                queue_items = get_queue_items(3)  # Show first 3 items
                for j, item in enumerate(queue_items):
                    if len(item) > 100:
                        item_preview = item[:100] + "..."
                    else:
                        item_preview = item
                    print(f"   Item {j+1}: {item_preview}")
        except Exception as e:
            print(f"❌ Queue check failed: {e}")
        
        # Check database articles count
        try:
            articles_count = get_articles_count()
            print(f"💾 Articles in Database: {articles_count}")
        except Exception as e:
            print(f"❌ Database check failed: {e}")
        
        # Check ingest status via API
        try:
            response = requests.get("http://localhost:8000/ingest/status")
            if response.status_code == 200:
                status = response.json()
                print(f"📋 Ingest Status: {status}")
            else:
                print(f"⚠️ Ingest status: {response.status_code}")
        except Exception as e:
            print(f"❌ Ingest status check failed: {e}")
        
        time.sleep(5)
    
    print("\n🎉 Monitoring complete!")
    
    # Final summary
    print(f"\n{'='*60}")
    print("📊 FINAL SUMMARY")
    print(f"{'='*60}")
    
    try:
        final_queue_size = get_queue_size()
        final_articles_count = get_articles_count()
        print(f"📤 Final Queue Size: {final_queue_size}")
        print(f"💾 Final Articles Count: {final_articles_count}")
        
        if final_queue_size == 0 and final_articles_count > 0:
            print("✅ SUCCESS: Pipeline appears to be working! Article processed and stored.")
        elif final_queue_size > 0:
            print("⏳ IN PROGRESS: Articles still in queue, may need more time.")
        else:
            print("❓ UNKNOWN: Check logs for processing status.")
            
    except Exception as e:
        print(f"❌ Final summary failed: {e}")

if __name__ == "__main__":
    monitor_new_pipeline() 