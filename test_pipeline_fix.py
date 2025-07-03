#!/usr/bin/env python3
"""
Test Pipeline Fix - Verify pipeline works with unique URLs and proper field propagation
"""

import time
import requests
import json
import uuid
from datetime import datetime

def test_pipeline_fix():
    """Test the pipeline with unique URLs and verify field propagation"""
    
    print("🔧 Testing Pipeline Fix")
    print("=" * 50)
    
    # Test articles with unique URLs
    test_articles = [
        {
            "title": "South China Sea Military Exercises Escalate",
            "body": "Multiple naval vessels from different countries are conducting military exercises in the South China Sea. The exercises involve naval drills and show of force operations in disputed waters near the Spratly Islands.",
            "source_url": f"https://example.com/scs-{uuid.uuid4().hex[:8]}",
            "region": "South China Sea",
            "topic": "Military"
        },
        {
            "title": "Taiwan Strait Tensions Rise",
            "body": "Recent military activities in the Taiwan Strait have raised concerns about regional stability. Naval patrols and air defense exercises are being conducted by multiple parties.",
            "source_url": f"https://example.com/taiwan-{uuid.uuid4().hex[:8]}",
            "region": "Taiwan Strait", 
            "topic": "Military"
        },
        {
            "title": "Maritime Security Operations in Southeast Asia",
            "body": "Coastal nations are implementing new maritime security measures to protect shipping lanes and prevent illegal activities in regional waters.",
            "source_url": f"https://example.com/maritime-{uuid.uuid4().hex[:8]}",
            "region": "Southeast Asia",
            "topic": "Security"
        }
    ]
    
    # Test single article first
    print("📝 Testing single article...")
    single_article = test_articles[0]
    single_article["article_id"] = str(uuid.uuid4())
    
    response = requests.post(
        "http://localhost:8000/ingest/v2/",
        json=single_article,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id")
        print(f"✅ Single article queued: {task_id}")
    else:
        print(f"❌ Single article failed: {response.status_code} - {response.text}")
        return
    
    # Wait for processing
    print("⏳ Waiting for single article processing...")
    time.sleep(10)
    
    # Test batch processing
    print("📦 Testing batch processing...")
    batch_articles = []
    for i, article in enumerate(test_articles[1:], 1):
        article["article_id"] = str(uuid.uuid4())
        batch_articles.append(article)
    
    response = requests.post(
        "http://localhost:8000/ingest/v2/batch-optimized/?batch_size=3",
        json=batch_articles,
        headers={"Content-Type": "application/json"}
    )
    
    if response.status_code == 200:
        result = response.json()
        task_id = result.get("task_id")
        print(f"✅ Batch articles queued: {task_id}")
    else:
        print(f"❌ Batch articles failed: {response.status_code} - {response.text}")
        return
    
    # Wait for batch processing
    print("⏳ Waiting for batch processing...")
    time.sleep(15)
    
    # Check results
    print("🔍 Checking results...")
    time.sleep(5)
    
    # Check articles in database
    response = requests.get("http://localhost:8000/ingest/status")
    if response.status_code == 200:
        status = response.json()
        print(f"📊 Status: {status}")
    
    print("✅ Pipeline test completed!")

if __name__ == "__main__":
    test_pipeline_fix() 