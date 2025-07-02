#!/usr/bin/env python3

"""
Test the updated pipeline with the new Supabase schema
"""

import sys
import os
sys.path.append('.')

import requests
import time
import json
from utils.supabase_client import get_supabase_client

def test_updated_schema_pipeline():
    """Test the pipeline with updated schema"""
    
    print("🧪 Testing Updated Schema Pipeline")
    print("=" * 50)
    
    # Test article with new schema fields
    test_article = {
        "article_id": "schema-updated-test-001",
        "title": "Updated Schema Test - Maritime Security in East Asia",
        "body": """
        <div class="article-content">
        <p>This is a comprehensive test article to verify the updated StraitWatch pipeline 
        works correctly with the new Supabase schema including all required fields.</p>
        
        <p>Recent developments in East Asia have highlighted cybersecurity threats to 
        maritime infrastructure. China, Taiwan, Philippines, and Singapore are key 
        stakeholders monitoring naval operations in the South China Sea.</p>
        
        <p>Military experts analyze the geopolitical implications while commercial 
        shipping operators implement enhanced security measures for critical systems.</p>
        </div>
        """,
        "url": "https://test.straitwatch.com/updated-schema-test",
        "source": "StraitWatch Test Suite",
        "region": "East Asia",
        "topic": "Maritime Security",
        "published_at": "2025-07-01T23:00:00Z"
    }
    
    # Test 1: Check API health
    print("1️⃣ Checking API health...")
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            print("✅ API is healthy")
        else:
            print("❌ API health check failed")
            return
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return
    
    # Test 2: Submit article to updated pipeline
    print("\n2️⃣ Submitting test article to v2 pipeline...")
    try:
        start_time = time.time()
        response = requests.post(
            "http://localhost:8000/ingest/v2/",
            json=test_article,
            timeout=60
        )
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"✅ Article submitted successfully")
            print(f"   Task ID: {response_data.get('task_id')}")
            print(f"   Response time: {time.time() - start_time:.2f}s")
        else:
            print(f"❌ Submission failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return
            
    except Exception as e:
        print(f"❌ Submission error: {e}")
        return
    
    # Test 3: Wait and check processing
    print("\n3️⃣ Waiting for processing to complete...")
    time.sleep(45)  # Give pipeline time to process
    
    # Test 4: Check Supabase for stored article
    print("\n4️⃣ Checking Supabase for stored article...")
    try:
        supabase = get_supabase_client()
        
        # Check if article was stored with new schema
        result = supabase.table("articles").select("*").eq("original_id", test_article["article_id"]).execute()
        
        if result.data:
            stored_article = result.data[0]
            print("✅ Article found in Supabase with updated schema:")
            print(f"   Database ID: {stored_article.get('id')}")
            print(f"   Original ID: {stored_article.get('original_id')}")
            print(f"   Title: {stored_article.get('title')[:50]}...")
            print(f"   Content: {len(stored_article.get('content', ''))} chars")
            print(f"   Cleaned: {len(stored_article.get('cleaned', ''))} chars")
            print(f"   URL: {stored_article.get('url')}")
            print(f"   Source: {stored_article.get('source')}")
            print(f"   Region: {stored_article.get('region')}")
            print(f"   Topic: {stored_article.get('topic')}")
            print(f"   Tags: {len(stored_article.get('tags', []))} tags")
            print(f"   Entities: {len(stored_article.get('entities', []))} entities")
            print(f"   Cluster ID: {stored_article.get('cluster_id')}")
            print(f"   Embedding Dimensions: {stored_article.get('embedding_dimensions')}")
            print(f"   Processed By: {stored_article.get('processed_by')}")
            print(f"   Inserted At: {stored_article.get('inserted_at')}")
            
            # Verify all expected fields are present
            expected_fields = [
                'id', 'original_id', 'title', 'content', 'cleaned', 'url', 
                'source', 'region', 'topic', 'tags', 'entities', 'cluster_id', 
                'embedding_dimensions', 'processed_by', 'inserted_at'
            ]
            
            missing_fields = []
            for field in expected_fields:
                if field not in stored_article or stored_article[field] is None:
                    if field not in ['confidence_score', 'published_at']:  # Optional fields
                        missing_fields.append(field)
            
            if missing_fields:
                print(f"⚠️  Missing fields: {missing_fields}")
            else:
                print("✅ All required schema fields are present")
                
        else:
            print("❌ Article not found in database")
            return
    
    except Exception as e:
        print(f"❌ Database check failed: {e}")
        return
    
    # Test 5: Check pipeline status
    print("\n5️⃣ Checking pipeline status...")
    try:
        response = requests.get("http://localhost:8000/ingest/status")
        if response.status_code == 200:
            status_data = response.json()
            print("✅ Pipeline status:")
            print(f"   Articles Count: {status_data.get('articles_count')}")
            print(f"   Status: {status_data.get('status')}")
        else:
            print("⚠️ Could not get pipeline status")
    except Exception as e:
        print(f"⚠️ Status check error: {e}")
    
    print("\n🎉 Updated schema pipeline test completed!")
    print("The pipeline is working with the new schema structure")

if __name__ == "__main__":
    test_updated_schema_pipeline() 