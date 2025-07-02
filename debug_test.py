#!/usr/bin/env python3
"""
Debug test - Send one article and monitor processing
"""

import time
from scraper.ingest_client import IngestClient

def debug_single_article():
    """Send a single article and monitor processing"""
    
    print("🔍 Debug Test - Single Article Processing")
    print("=" * 50)
    
    client = IngestClient("http://localhost:8000")
    
    # Test article
    test_article = {
        "title": "Debug Test Article",
        "body": "This is a test article for debugging the pipeline. It contains some HTML <p>tags</p> and should be processed through the complete pipeline.",
        "source_url": "https://debug.test/article1",
        "region": "Debug Region",
        "topic": "Debug"
    }
    
    print(f"📤 Sending test article: {test_article['title']}")
    result = client.send_article(**test_article)
    print(f"📋 Result: {result}")
    
    if result.get('status') == 'queued':
        article_id = result['id']
        task_id = result.get('message', '').split('Task ID: ')[-1] if 'Task ID:' in result.get('message', '') else 'unknown'
        print(f"✅ Article queued with ID: {article_id}")
        print(f"🔄 Task ID: {task_id}")
        
        # Monitor for 30 seconds
        print("\n⏳ Monitoring for 30 seconds...")
        for i in range(6):
            print(f"[{i+1}/6] Checking status...")
            status = client.get_status()
            print(f"   Status: {status}")
            
            # Check queue size
            import subprocess
            try:
                result = subprocess.run([
                    'python', '-c', 
                    'from app.services.redis_queue import get_queue_size; print(f"Queue size: {get_queue_size()}")'
                ], capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    print(f"   {result.stdout.strip()}")
            except Exception as e:
                print(f"   Queue check failed: {e}")
            
            time.sleep(5)
    else:
        print(f"❌ Failed to queue article: {result}")
    
    print("\n🏁 Debug test complete")

if __name__ == "__main__":
    debug_single_article() 