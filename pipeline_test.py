#!/usr/bin/env python3
"""
Complete Pipeline Test - Tests the full StraitWatch pipeline
"""

import time
import logging
import requests
from scraper.ingest_client import IngestClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_complete_pipeline():
    """Test the complete article processing pipeline"""
    
    print("🚀 Starting Complete Pipeline Test")
    print("=" * 50)
    
    # Create ingest client
    client = IngestClient("http://localhost:8000")
    
    # Test articles with variety for clustering
    test_articles = [
        {
            "title": "South China Sea Military Exercises Escalate",
            "body": "Multiple naval vessels from different countries are conducting military exercises in the South China Sea, raising tensions in the disputed waters. Fighter jets and warships have been spotted conducting drills near contested islands.",
            "source_url": "https://example.com/article1",
            "region": "South China Sea",
            "topic": "Military"
        },
        {
            "title": "Taiwan Strait Defense Posture Enhanced",
            "body": "Taiwan has enhanced its defense posture in the Taiwan Strait following increased military activity. New radar systems and missile defense platforms have been deployed along the coastline.",
            "source_url": "https://example.com/article2", 
            "region": "Taiwan Strait",
            "topic": "Military"
        },
        {
            "title": "Economic Sanctions Affect Regional Trade",
            "body": "New economic sanctions are disrupting trade routes throughout the Asia-Pacific region. Shipping companies report delays and increased costs as they navigate new compliance requirements.",
            "source_url": "https://example.com/article3",
            "region": "Asia-Pacific", 
            "topic": "Economic"
        },
        {
            "title": "Naval Patrol Activities Increase",
            "body": "Naval patrol activities have significantly increased across strategic waterways. Coast guard vessels and naval ships are conducting more frequent patrols in international waters.",
            "source_url": "https://example.com/article4",
            "region": "South China Sea",
            "topic": "Military"  
        },
        {
            "title": "Diplomatic Talks Resume on Trade Issues",
            "body": "High-level diplomatic talks have resumed focusing on resolving ongoing trade disputes. Economic ministers from multiple countries are meeting to discuss new frameworks for cooperation.",
            "source_url": "https://example.com/article5",
            "region": "Asia-Pacific",
            "topic": "Economic"
        }
    ]
    
    print("📊 Initial Pipeline Status:")
    initial_status = client.get_status()
    print(f"   Status: {initial_status}")
    print()
    
    # Send test articles
    print("📤 Sending Articles to Pipeline:")
    article_ids = []
    for i, article in enumerate(test_articles):
        print(f"   [{i+1}/5] Sending: {article['title'][:50]}...")
        result = client.send_article(**article)
        if result.get('status') == 'queued':
            article_ids.append(result['id'])
            print(f"      ✅ Queued with ID: {result['id']}")
        else:
            print(f"      ❌ Failed: {result}")
        time.sleep(1)
    
    print(f"\n📋 Successfully queued {len(article_ids)} articles")
    print()
    
    # Monitor processing
    print("⏳ Monitoring Pipeline Processing:")
    for i in range(12):  # Monitor for 2 minutes
        status = client.get_status()
        print(f"   [Check {i+1}] Total: {status.get('total_articles', 0)}, "
              f"Processed: {status.get('processed_articles', 0)}, "
              f"Unprocessed: {status.get('unprocessed_articles', 0)}")
        
        if status.get('unprocessed_articles', 0) == 0 and status.get('total_articles', 0) > 0:
            print("   ✅ All articles processed!")
            break
            
        time.sleep(10)
    
    print()
    
    # Check clusters
    print("🔍 Checking for Clusters:")
    try:
        response = requests.get("http://localhost:8000/reports/quick")
        if response.status_code == 200:
            report = response.json()
            print(f"   Quick Report Status: {report.get('status')}")
            if 'summary' in report:
                summary = report['summary']
                print(f"   Total Articles: {summary.get('total_articles', 0)}")
                print(f"   Recent Articles: {len(summary.get('recent_articles', []))}")
        else:
            print(f"   ❌ Failed to get report: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error getting report: {e}")
    
    print()
    
    # Check database directly
    print("💾 Database Status:")
    try:
        from utils.supabase_client import get_supabase_client
        supabase = get_supabase_client()
        
        # Check articles
        articles_result = supabase.table('articles').select('id, title, source').execute()
        articles_count = len(articles_result.data) if articles_result.data else 0
        print(f"   Articles in DB: {articles_count}")
        
        if articles_result.data:
            print("   Recent articles:")
            for article in articles_result.data[-3:]:
                print(f"      - {article.get('title', 'No title')[:40]}...")
        
        # Check clusters
        clusters_result = supabase.table('clusters').select('id, theme, status, article_ids').execute()
        clusters_count = len(clusters_result.data) if clusters_result.data else 0
        print(f"   Clusters created: {clusters_count}")
        
        if clusters_result.data:
            print("   Cluster status:")
            for cluster in clusters_result.data:
                article_count = len(cluster.get('article_ids', []))
                theme = cluster.get('theme', 'Unknown')[:30]
                status = cluster.get('status', 'unknown')
                print(f"      - {theme}... ({article_count} articles, {status})")
                
    except Exception as e:
        print(f"   ❌ Error checking database: {e}")
    
    print()
    
    # Final status
    print("📊 Final Pipeline Status:")
    final_status = client.get_status()
    print(f"   {final_status}")
    
    print()
    print("🎉 Pipeline Test Complete!")
    print("=" * 50)

if __name__ == "__main__":
    test_complete_pipeline() 