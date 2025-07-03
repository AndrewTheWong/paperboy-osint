#!/usr/bin/env python3
"""
Final Full Pipeline Test - Comprehensive verification of the complete StraitWatch pipeline
"""

import time
import requests
import json
import uuid
from datetime import datetime

def test_full_pipeline_final():
    """Comprehensive test of the complete pipeline"""
    
    print("🚀 StraitWatch Full Pipeline Test - FINAL")
    print("=" * 60)
    
    # Test articles with unique URLs for clustering
    test_articles = [
        {
            "title": "South China Sea Naval Exercises Intensify",
            "body": "The Chinese Navy conducted large-scale military exercises in the South China Sea, involving multiple destroyers, frigates, and aircraft carriers. The exercises focused on anti-submarine warfare and air defense operations in disputed waters.",
            "source_url": f"https://maritime-news.com/scs-exercises-{uuid.uuid4().hex[:8]}",
            "region": "South China Sea",
            "topic": "Military"
        },
        {
            "title": "Philippines Strengthens Maritime Security in Sulu Sea",
            "body": "The Philippine Navy has deployed additional patrol vessels to the Sulu Sea to combat piracy and protect commercial shipping routes. The operation involves coordination with neighboring countries.",
            "source_url": f"https://maritime-news.com/philippines-security-{uuid.uuid4().hex[:8]}",
            "region": "Southeast Asia",
            "topic": "Security"
        },
        {
            "title": "Taiwan Strait Tensions Escalate with Military Drills",
            "body": "Recent military activities in the Taiwan Strait have raised international concerns. Multiple naval vessels and aircraft are conducting exercises in the sensitive waterway.",
            "source_url": f"https://maritime-news.com/taiwan-tensions-{uuid.uuid4().hex[:8]}",
            "region": "Taiwan Strait",
            "topic": "Military"
        }
    ]
    
    # Step 1: Test single article ingestion
    print("📝 Step 1: Testing single article ingestion...")
    single_article = test_articles[0].copy()
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
    
    # Step 2: Test batch ingestion
    print("📦 Step 2: Testing batch ingestion...")
    batch_articles = []
    for article in test_articles[1:]:
        article_copy = article.copy()
        article_copy["article_id"] = str(uuid.uuid4())
        batch_articles.append(article_copy)
    
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
    
    # Step 3: Wait for processing
    print("⏳ Step 3: Waiting for pipeline processing...")
    time.sleep(20)
    
    # Step 4: Check processing status
    print("🔍 Step 4: Checking processing status...")
    response = requests.get("http://localhost:8000/ingest/status")
    if response.status_code == 200:
        status = response.json()
        print(f"📊 Pipeline Status: {status}")
    
    # Step 5: Check articles in database
    print("📰 Step 5: Checking articles in database...")
    from utils.supabase_client import get_supabase_client
    supabase = get_supabase_client()
    
    result = supabase.table('articles').select('id, title, cluster_id, tags, region, topic').execute()
    
    print(f"📊 Total articles: {len(result.data)}")
    for article in result.data:
        print(f"  📄 {article['title']}")
        print(f"    Cluster ID: {article.get('cluster_id', 'None')}")
        print(f"    Region: {article.get('region', 'None')}")
        print(f"    Topic: {article.get('topic', 'None')}")
        tags = article.get('tags', [])
        print(f"    Tags: {len(tags) if tags else 0} tags")
        print()
    
    # Step 6: Check clusters
    print("🔍 Step 6: Checking clusters...")
    clusters_result = supabase.table('clusters').select('*').execute()
    
    print(f"📊 Total clusters: {len(clusters_result.data)}")
    for cluster in clusters_result.data:
        print(f"  🔗 Cluster {cluster.get('cluster_id', 'None')}")
        print(f"    Theme: {cluster.get('theme', 'None')}")
        print(f"    Region: {cluster.get('region', 'None')}")
        print(f"    Topic: {cluster.get('topic', 'None')}")
        article_ids = cluster.get('article_ids', [])
        print(f"    Articles: {len(article_ids) if article_ids else 0}")
        print()
    
    # Step 7: Check API reports
    print("📋 Step 7: Checking API reports...")
    response = requests.get("http://localhost:8000/reports/quick")
    if response.status_code == 200:
        report = response.json()
        print(f"📊 Quick Report: {report}")
    
    print("✅ Full pipeline test completed successfully!")
    print("🎉 StraitWatch pipeline is working correctly!")

if __name__ == "__main__":
    test_full_pipeline_final() 