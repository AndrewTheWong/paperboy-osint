#!/usr/bin/env python3
"""
Test cluster labeling in the pipeline
"""

import requests
import json
import uuid

def test_cluster_labeling():
    print("🧪 Testing cluster labeling in pipeline...")
    
    # Test article
    test_article = {
        "title": "South China Sea Naval Exercises Test Cluster Labeling",
        "body": "The Chinese Navy conducted extensive naval exercises in the South China Sea, involving multiple destroyers, frigates, and aircraft carriers. The exercises focused on anti-submarine warfare and air defense operations in disputed waters near the Spratly Islands.",
        "source_url": f"https://test.com/cluster-labeling-test-{uuid.uuid4()}"
    }
    
    # Send to API
    response = requests.post(
        "http://localhost:8000/ingest/v2/",
        json=test_article,
        timeout=120
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Article processed: {result}")
        
        # Wait a moment for processing
        import time
        time.sleep(5)
        
        # Check if cluster label was generated
        from utils.supabase_client import get_supabase_client
        supabase = get_supabase_client()
        
        # Get the article
        result = supabase.table('articles').select('title, cluster_id, cluster_label').eq('url', test_article['source_url']).execute()
        
        if result.data:
            article = result.data[0]
            print(f"📰 Article found:")
            print(f"   Title: {article['title']}")
            print(f"   Cluster ID: {article.get('cluster_id')}")
            print(f"   Cluster Label: {article.get('cluster_label')}")
            
            # Check clusters table
            if article.get('cluster_id'):
                cluster_result = supabase.table('clusters').select('cluster_id, theme').eq('cluster_id', article['cluster_id']).execute()
                if cluster_result.data:
                    cluster = cluster_result.data[0]
                    print(f"🔍 Cluster info:")
                    print(f"   Cluster ID: {cluster['cluster_id']}")
                    print(f"   Theme: {cluster.get('theme')}")
        else:
            print("❌ Article not found in database")
    else:
        print(f"❌ API error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    test_cluster_labeling() 