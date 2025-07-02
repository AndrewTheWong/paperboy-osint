#!/usr/bin/env python3

"""
Simple test with correct API field names
"""

import requests
import time
import json

def test_single_article():
    """Test single article with correct field names"""
    
    # Test article with correct API field names
    test_article = {
        "id": "test-correct-format-001",
        "title": "Correct Format Test - Maritime Security",
        "body": """
        <div class="article-content">
        <p>This is a test article using the correct API field names. 
        The pipeline should process this without validation errors.</p>
        
        <p>Recent developments in East Asia have highlighted cybersecurity threats to 
        maritime infrastructure. China, Taiwan, Philippines, and Singapore are key 
        stakeholders monitoring naval operations in the South China Sea.</p>
        
        <p>Military experts analyze the geopolitical implications while commercial 
        shipping operators implement enhanced security measures for critical systems.</p>
        </div>
        """,
        "source_url": "https://test.straitwatch.com/correct-format-test",
        "region": "East Asia",
        "topic": "Maritime Security"
    }
    
    print("🧪 Testing single article with correct format...")
    
    try:
        response = requests.post(
            "http://localhost:8000/ingest/v2/",
            json=test_article,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Single article test successful!")
            return True
        else:
            print("❌ Single article test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_articles():
    """Test batch processing with correct field names"""
    
    # Test articles with correct API field names
    test_articles = [
        {
            "id": "batch-test-001",
            "title": "Batch Test Article 1 - Cybersecurity",
            "body": "This is the first batch test article about cybersecurity in maritime operations.",
            "source_url": "https://test.straitwatch.com/batch-1",
            "region": "Global",
            "topic": "Cybersecurity"
        },
        {
            "id": "batch-test-002", 
            "title": "Batch Test Article 2 - Naval Operations",
            "body": "This is the second batch test article about naval operations in the Pacific.",
            "source_url": "https://test.straitwatch.com/batch-2",
            "region": "Pacific",
            "topic": "Naval Operations"
        },
        {
            "id": "batch-test-003",
            "title": "Batch Test Article 3 - Geopolitics", 
            "body": "This is the third batch test article about geopolitical tensions in Asia.",
            "source_url": "https://test.straitwatch.com/batch-3",
            "region": "Asia",
            "topic": "Geopolitics"
        }
    ]
    
    print("\n🧪 Testing batch processing with correct format...")
    
    try:
        response = requests.post(
            "http://localhost:8000/ingest/v2/batch-optimized/?batch_size=3",
            json=test_articles,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Batch test successful!")
            return True
        else:
            print("❌ Batch test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_status():
    """Check pipeline status"""
    print("\n📊 Checking pipeline status...")
    
    try:
        response = requests.get("http://localhost:8000/ingest/status")
        if response.status_code == 200:
            status = response.json()
            print(f"Pipeline Status: {status}")
        else:
            print(f"Status check failed: {response.status_code}")
    except Exception as e:
        print(f"Status check error: {e}")

def main():
    """Run all tests"""
    print("🚀 StraitWatch Pipeline Test - Correct Format")
    print("=" * 50)
    
    # Test 1: Single article
    single_success = test_single_article()
    
    # Wait a bit
    time.sleep(5)
    
    # Test 2: Batch articles
    batch_success = test_batch_articles()
    
    # Wait for processing
    print("\n⏳ Waiting for processing to complete...")
    time.sleep(30)
    
    # Check status
    check_status()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   Single Article: {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"   Batch Processing: {'✅ PASS' if batch_success else '❌ FAIL'}")
    
    if single_success and batch_success:
        print("\n🎉 All tests passed! Pipeline is working correctly.")
    else:
        print("\n⚠️ Some tests failed. Check the logs above.")

if __name__ == "__main__":
    main() 