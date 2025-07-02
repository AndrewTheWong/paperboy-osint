#!/usr/bin/env python3
"""
Test ingest API functionality
"""

import requests
import json
import time

def test_ingest_api():
    """Test the ingest API with a sample article"""
    
    # Sample article data
    test_article = {
        "title": "Test Article - Taiwan Strait Tensions",
        "content": "Recent developments in the Taiwan Strait have raised concerns about regional stability. Military exercises and diplomatic tensions continue to escalate between major powers in the region.",
        "source": "test_source",
        "url": "https://example.com/test-article",
        "published_at": "2025-07-01T12:00:00Z",
        "language": "en"
    }
    
    print("=" * 60)
    print("TESTING INGEST API")
    print("=" * 60)
    
    try:
        # Test ingest endpoint
        print("📤 Sending test article to ingest API...")
        response = requests.post(
            "http://localhost:8000/ingest/",
            json=test_article,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Ingest API is working!")
            
            # Wait a moment for processing
            print("⏳ Waiting 5 seconds for processing...")
            time.sleep(5)
            
            # Check if article was stored
            print("🔍 Checking if article was stored...")
            check_response = requests.get("http://localhost:8000/ingest/status")
            print(f"Status check: {check_response.status_code}")
            if check_response.status_code == 200:
                print(f"Status response: {check_response.json()}")
            
        else:
            print("❌ Ingest API failed")
            
    except requests.exceptions.ConnectionError:
        print("❌ Could not connect to API server. Is it running?")
    except Exception as e:
        print(f"❌ Error testing ingest API: {e}")

def test_celery_tasks():
    """Test if Celery tasks are working"""
    print("\n" + "=" * 60)
    print("TESTING CELERY TASKS")
    print("=" * 60)
    
    try:
        # Test clustering task
        print("🔍 Testing clustering task...")
        from app.tasks.cluster import run_clustering
        
        # This should trigger clustering on any articles in the queue
        result = run_clustering.delay()
        print(f"Clustering task queued: {result.id}")
        
        # Test summarization task
        print("📝 Testing summarization task...")
        from app.tasks.summarize import summarize_all_pending_clusters
        
        result = summarize_all_pending_clusters.delay()
        print(f"Summarization task queued: {result.id}")
        
        print("✅ Celery tasks are working!")
        
    except Exception as e:
        print(f"❌ Error testing Celery tasks: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ingest_api()
    test_celery_tasks() 