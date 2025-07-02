#!/usr/bin/env python3

"""
Test with proper UUIDs for article IDs
"""

import requests
import time
import json
import uuid

def test_with_proper_uuids():
    """Test with proper UUID article IDs"""
    
    # Generate proper UUIDs
    article_id_1 = str(uuid.uuid4())
    article_id_2 = str(uuid.uuid4())
    
    # Test article with proper UUID
    test_article = {
        "id": article_id_1,
        "title": "UUID Test - Maritime Security with Proper UUID",
        "body": """
        <div class="article-content">
        <p>This is a test article using a proper UUID format. 
        The pipeline should process this without UUID validation errors.</p>
        
        <p>Recent developments in East Asia have highlighted cybersecurity threats to 
        maritime infrastructure. China, Taiwan, Philippines, and Singapore are key 
        stakeholders monitoring naval operations in the South China Sea.</p>
        
        <p>Military experts analyze the geopolitical implications while commercial 
        shipping operators implement enhanced security measures for critical systems.</p>
        </div>
        """,
        "source_url": "https://test.straitwatch.com/uuid-test",
        "region": "East Asia",
        "topic": "Maritime Security"
    }
    
    print("🧪 Testing with proper UUID...")
    print(f"Article ID: {article_id_1}")
    
    try:
        response = requests.post(
            "http://localhost:8000/ingest/v2/",
            json=test_article,
            timeout=30
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ UUID test successful!")
            return True
        else:
            print("❌ UUID test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_batch_with_uuids():
    """Test batch processing with proper UUIDs"""
    
    # Generate proper UUIDs for batch
    uuid_1 = str(uuid.uuid4())
    uuid_2 = str(uuid.uuid4())
    uuid_3 = str(uuid.uuid4())
    
    test_articles = [
        {
            "id": uuid_1,
            "title": "Batch UUID Test 1 - Cybersecurity",
            "body": "This is the first batch test article with proper UUID about cybersecurity in maritime operations.",
            "source_url": "https://test.straitwatch.com/batch-uuid-1",
            "region": "Global",
            "topic": "Cybersecurity"
        },
        {
            "id": uuid_2, 
            "title": "Batch UUID Test 2 - Naval Operations",
            "body": "This is the second batch test article with proper UUID about naval operations in the Pacific.",
            "source_url": "https://test.straitwatch.com/batch-uuid-2",
            "region": "Pacific",
            "topic": "Naval Operations"
        },
        {
            "id": uuid_3,
            "title": "Batch UUID Test 3 - Geopolitics", 
            "body": "This is the third batch test article with proper UUID about geopolitical tensions in Asia.",
            "source_url": "https://test.straitwatch.com/batch-uuid-3",
            "region": "Asia",
            "topic": "Geopolitics"
        }
    ]
    
    print(f"\n🧪 Testing batch with proper UUIDs...")
    print(f"UUIDs: {uuid_1[:8]}..., {uuid_2[:8]}..., {uuid_3[:8]}...")
    
    try:
        response = requests.post(
            "http://localhost:8000/ingest/v2/batch-optimized/?batch_size=3",
            json=test_articles,
            timeout=60
        )
        
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Batch UUID test successful!")
            return True
        else:
            print("❌ Batch UUID test failed")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_database():
    """Check if articles are stored in database"""
    print("\n📊 Checking database...")
    
    try:
        from utils.supabase_client import get_supabase_client
        supabase = get_supabase_client()
        
        result = supabase.table("articles").select("*").execute()
        
        print(f"Articles in database: {len(result.data)}")
        
        if result.data:
            print("Sample articles:")
            for i, article in enumerate(result.data[:3]):
                print(f"  {i+1}. ID: {article.get('id')}")
                print(f"     Original ID: {article.get('original_id')}")
                print(f"     Title: {article.get('title')[:50]}...")
                print(f"     Region: {article.get('region')}")
                print(f"     Topic: {article.get('topic')}")
                print(f"     Processed By: {article.get('processed_by')}")
                print()
        else:
            print("No articles found in database")
            
    except Exception as e:
        print(f"Database check error: {e}")

def main():
    """Run UUID tests"""
    print("🚀 StraitWatch Pipeline Test - Proper UUIDs")
    print("=" * 50)
    
    # Test 1: Single article with UUID
    single_success = test_with_proper_uuids()
    
    # Wait a bit
    time.sleep(5)
    
    # Test 2: Batch articles with UUIDs
    batch_success = test_batch_with_uuids()
    
    # Wait for processing
    print("\n⏳ Waiting for processing to complete...")
    time.sleep(45)
    
    # Check database
    check_database()
    
    # Summary
    print("\n📋 Test Summary:")
    print(f"   Single Article (UUID): {'✅ PASS' if single_success else '❌ FAIL'}")
    print(f"   Batch Processing (UUID): {'✅ PASS' if batch_success else '❌ FAIL'}")
    
    if single_success and batch_success:
        print("\n🎉 All UUID tests passed! Pipeline is working correctly.")
    else:
        print("\n⚠️ Some UUID tests failed. Check the logs above.")

if __name__ == "__main__":
    main() 