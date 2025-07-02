#!/usr/bin/env python3
"""
Test script to verify schema fix for the upgraded pipeline
"""

import requests
import json
import time

def test_schema_fix():
    """Test the upgraded pipeline with schema fix"""
    print("🧪 Testing Schema Fix - Upgraded Pipeline")
    print("=" * 50)
    
    # Test article that matches the simplified schema
    article = {
        "article_id": "schema-fix-test-001",
        "title": "Schema Fix Test Article", 
        "body": "<p>This article tests if the schema fix resolves the database storage issue. The article discusses maritime security developments in the South China Sea region.</p>",
        "region": "East Asia",
        "topic": "Maritime Security",
        "source_url": "https://example.com/schema-fix-test"
    }
    
    try:
        # Submit to v2 pipeline
        print(f"📤 Submitting article to v2 pipeline...")
        response = requests.post(
            "http://localhost:8000/ingest/v2/",
            json=article,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Article submitted successfully!")
            print(f"   Article ID: {result.get('id')}")
            print(f"   Status: {result.get('status')}")
            
            # Wait for processing
            print(f"⏳ Waiting 15 seconds for processing...")
            time.sleep(15)
            
            # Check status
            status_response = requests.get("http://localhost:8000/ingest/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                print(f"\n📊 Pipeline Status:")
                print(f"   Total articles: {status_data.get('total_articles', 0)}")
                print(f"   Processed articles: {status_data.get('processed_articles', 0)}")
                print(f"   Clusters: {status_data.get('total_clusters', 0)}")
                
                if status_data.get('processed_articles', 0) > 0:
                    print(f"🎉 SUCCESS! Articles are now being processed and stored!")
                else:
                    print(f"⚠️  Still processing... Check Celery worker logs")
            else:
                print(f"❌ Failed to get status: {status_response.status_code}")
                
        else:
            print(f"❌ Failed to submit article: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing schema fix: {e}")

if __name__ == "__main__":
    test_schema_fix() 