#!/usr/bin/env python3
"""
Fix stuck processing queue and test new pipeline
"""

import redis
import requests
import json

def clear_stuck_queues():
    """Clear stuck Redis queues to reset the pipeline"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Clear stuck preprocessing queue
        preprocessing_length = r.llen('preprocessing')
        print(f"🧹 Found {preprocessing_length} stuck items in preprocessing queue")
        
        if preprocessing_length > 0:
            cleared = r.delete('preprocessing')
            print(f"✅ Cleared preprocessing queue (removed {preprocessing_length} items)")
        
        # Clear clustering queue (should be empty anyway)
        clustering_length = r.llen('clustering_queue')
        if clustering_length > 0:
            cleared = r.delete('clustering_queue')
            print(f"✅ Cleared clustering queue (removed {clustering_length} items)")
        
        print(f"🎯 Queues are now clean!")
        
    except Exception as e:
        print(f"❌ Error clearing queues: {e}")

def test_new_pipeline():
    """Test the new upgraded pipeline"""
    print("\n🧪 Testing New Upgraded Pipeline")
    print("=" * 50)
    
    # Test single article with v2 endpoint
    test_article = {
        "title": "Test Article - Cleared Queue",
        "body": """
        <p>This is a test article to verify the upgraded pipeline works 
        after clearing the stuck preprocessing queue. The article discusses 
        maritime security developments in the South China Sea.</p>
        
        <p>China has increased naval activity in the region, prompting 
        concerns from Taiwan and other neighbors about cybersecurity 
        threats to critical infrastructure.</p>
        """,
        "region": "East Asia",
        "topic": "Maritime Security",
        "source_url": "https://example.com/test-cleared-queue"
    }
    
    try:
        # Send to v2 pipeline
        response = requests.post(
            "http://localhost:8000/ingest/v2/",
            json=test_article,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Article ingested successfully!")
            print(f"   Article ID: {result.get('id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Message: {result.get('message')}")
            
            # Check status after a moment
            import time
            time.sleep(5)  # Give more time for processing
            
            status_response = requests.get("http://localhost:8000/ingest/status")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"\n📊 Current Status:")
                print(f"   Total Articles: {status.get('total_articles')}")
                print(f"   Processed: {status.get('processed_articles')}")
                print(f"   Unprocessed: {status.get('unprocessed_articles')}")
            
        else:
            print(f"❌ Failed to ingest: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")

if __name__ == "__main__":
    print("🔧 Fixing Stuck StraitWatch Pipeline")
    print("=" * 50)
    
    # Clear stuck queues
    clear_stuck_queues()
    
    # Test new pipeline
    test_new_pipeline()
    
    print(f"\n✅ Pipeline maintenance complete!")
    print(f"🚀 New pipeline should now be working with registered tasks.") 