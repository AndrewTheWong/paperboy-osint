#!/usr/bin/env python3
"""
Test the updated pipeline that works with the existing basic database schema
"""

import sys
import os
sys.path.append('.')

import requests
import time
import json
from utils.supabase_client import get_supabase_client

def test_updated_pipeline():
    """Test the v2 pipeline with schema compatibility fix"""
    
    # Test article
    test_article = {
        "article_id": "schema-fix-test-v2",
        "title": "Schema Fix Test Article V2",
        "body": """
        <p>This is a test article to verify the updated pipeline works 
        with the existing basic database schema. The article discusses 
        maritime security developments in the South China Sea.</p>
        
        <p>China has increased naval activity in the region, prompting 
        concerns from Taiwan and other neighbors about cybersecurity 
        threats to critical infrastructure.</p>
        """,
        "region": "East Asia",
        "topic": "Maritime Security", 
        "source_url": "https://example.com/test-schema-fix-v2"
    }
    
    print("🧪 Testing Updated Pipeline with Schema Compatibility")
    print("=" * 60)
    
    # Test API endpoint
    try:
        print("📡 Sending article to v2 pipeline...")
        response = requests.post(
            "http://localhost:8000/ingest/v2/",
            json=test_article,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Article queued successfully!")
            print(f"   Task ID: {result.get('task_id')}")
            print(f"   Article ID: {result.get('article_id')}")
            
            # Wait for processing
            print("⏳ Waiting for pipeline to complete...")
            time.sleep(45)  # Give time for all steps
            
            # Check database
            print("🔍 Checking database...")
            supabase = get_supabase_client()
            
            # Look for our test article
            result = supabase.table("articles").select("*").order("created_at", desc=True).limit(5).execute()
            
            if result.data:
                print(f"✅ Found {len(result.data)} recent articles:")
                for i, article in enumerate(result.data, 1):
                    print(f"   {i}. ID: {article.get('id')}")
                    print(f"      Title: {article.get('title', 'N/A')}")
                    print(f"      Source: {article.get('source', 'N/A')}")
                    print(f"      Created: {article.get('created_at', 'N/A')}")
                    print()
                
                # Check if our test article is there
                test_found = any("Schema Fix Test Article V2" in article.get('title', '') for article in result.data)
                if test_found:
                    print("🎉 SUCCESS: Test article found in database!")
                    print("✅ Pipeline is working with existing schema")
                else:
                    print("⚠️  Test article not found, but pipeline may still be processing...")
            else:
                print("❌ No articles found in database")
                
        else:
            print(f"❌ API request failed: {response.status_code}")
            print(f"   Response: {response.text}")
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Network error: {e}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    test_updated_pipeline() 