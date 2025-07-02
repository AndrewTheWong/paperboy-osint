#!/usr/bin/env python3
"""
Test UUID Pipeline - Verify pipeline works with proper UUIDs
"""

import requests
import json
import uuid
from datetime import datetime

def test_uuid_pipeline():
    """Test pipeline with proper UUIDs"""
    
    # Generate proper UUIDs
    article_id_1 = str(uuid.uuid4())
    article_id_2 = str(uuid.uuid4())
    article_id_3 = str(uuid.uuid4())
    
    # Test article with proper UUID
    test_article = {
        "id": article_id_1,
        "title": "UUID Test - Maritime Security with Proper UUID",
        "body": """
        <div class="article-content">
        <p>This is a test article using proper UUID format. 
        The pipeline should process this without UUID validation errors.</p>
        
        <p>Recent developments in East Asia have highlighted cybersecurity threats to 
        maritime infrastructure. China, Taiwan, Philippines, and Singapore are key 
        stakeholders monitoring naval operations in the South China Sea.</p>
        
        <p>Military experts analyze the geopolitical implications while commercial 
        shipping operators implement enhanced security measures for critical systems.</p>
        </div>
        """,
        "region": "East Asia",
        "topic": "Maritime Security",
        "source_url": "https://test.straitwatch.com/uuid-test"
    }
    
    print(f"🧪 Testing pipeline with proper UUID: {article_id_1}")
    
    # Test single article ingestion
    try:
        response = requests.post(
            "http://localhost:8000/ingest/v2/",
            json=test_article,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Single article ingested successfully")
            print(f"   Task ID: {result.get('task_id')}")
            print(f"   Status: {result.get('status')}")
        else:
            print(f"❌ Single article ingestion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing single article: {e}")
    
    # Test batch ingestion with proper UUIDs
    batch_articles = [
        {
            "id": article_id_2,
            "title": "Batch UUID Test 1 - Cybersecurity",
            "body": "Test article for batch processing with proper UUID format.",
            "region": "East Asia",
            "topic": "Cybersecurity",
            "source_url": "https://test.straitwatch.com/batch-1"
        },
        {
            "id": article_id_3,
            "title": "Batch UUID Test 2 - Naval Operations",
            "body": "Second test article for batch processing with proper UUID format.",
            "region": "Southeast Asia",
            "topic": "Naval Operations",
            "source_url": "https://test.straitwatch.com/batch-2"
        }
    ]
    
    print(f"\n🧪 Testing batch pipeline with proper UUIDs")
    
    try:
        response = requests.post(
            "http://localhost:8000/ingest/v2/batch-optimized/?batch_size=3",
            json=batch_articles,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch ingestion successful")
            print(f"   Task ID: {result.get('task_id')}")
            print(f"   Status: {result.get('status')}")
            print(f"   Article count: {result.get('article_count')}")
        else:
            print(f"❌ Batch ingestion failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing batch: {e}")
    
    # Check status
    print(f"\n📊 Checking pipeline status...")
    try:
        response = requests.get("http://localhost:8000/ingest/status")
        if response.status_code == 200:
            status = response.json()
            print(f"✅ Status check successful")
            print(f"   Total articles: {status.get('total_articles', 0)}")
            print(f"   Processing: {status.get('processing', 0)}")
            print(f"   Completed: {status.get('completed', 0)}")
        else:
            print(f"❌ Status check failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")

if __name__ == "__main__":
    test_uuid_pipeline() 