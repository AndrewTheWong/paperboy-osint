#!/usr/bin/env python3
"""
Simple test to verify the import fix for the upgraded pipeline
"""

import requests
import json
import time

def test_single_article_import_fix():
    """Test single article with import fix"""
    print("🧪 Testing Import Fix - Upgraded Pipeline")
    print("=" * 50)
    
    # Test article
    article = {
        "article_id": "import-fix-test-001",
        "title": "Import Fix Test Article", 
        "body": "<p>This article tests if the import fix resolves the storage issue in the upgraded pipeline.</p>",
        "region": "Test Region",
        "topic": "Technical Test",
        "source_url": "https://example.com/import-fix-test"
    }
    
    try:
        # Submit to v2 pipeline
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
            print("⏳ Waiting 15 seconds for pipeline to complete...")
            time.sleep(15)
            
            # Check status
            status_response = requests.get("http://localhost:8000/ingest/status")
            if status_response.status_code == 200:
                status = status_response.json()
                print(f"📊 Final Status:")
                print(f"   Total Articles: {status.get('total_articles', 0)}")
                print(f"   Processed Articles: {status.get('processed_articles', 0)}")
                print(f"   Unprocessed Articles: {status.get('unprocessed_articles', 0)}")
                
                if status.get('processed_articles', 0) > 0:
                    print("🎉 SUCCESS: Articles are now being processed!")
                else:
                    print("⚠️ Still processing or error occurred")
            
        else:
            print(f"❌ Failed to submit article: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing pipeline: {e}")

if __name__ == "__main__":
    test_single_article_import_fix() 