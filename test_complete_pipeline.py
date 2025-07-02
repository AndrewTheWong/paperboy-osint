#!/usr/bin/env python3
"""
Complete Pipeline Test - Tests the full StraitWatch pipeline end-to-end
"""

import time
import requests
from scraper.ingest_client import IngestClient

def test_complete_pipeline():
    """Test the complete article processing pipeline"""
    
    print("🚀 StraitWatch Complete Pipeline Test")
    print("=" * 60)
    
    # Create ingest client
    client = IngestClient("http://localhost:8000")
    
    # Test articles for clustering (similar topics for clustering)
    test_articles = [
        {
            "title": "South China Sea Military Exercises Escalate",
            "body": "Multiple naval vessels from different countries are conducting military exercises in the South China Sea. The exercises involve naval drills and show of force operations in disputed waters near the Spratly Islands.",
            "source_url": "https://example.com/scs1",
            "region": "South China Sea",
            "topic": "Military"
        },
        {
            "title": "Taiwan Strait Tensions Rise",
            "body": "Military aircraft from the People's Liberation Army conducted flights near Taiwan airspace. Defense officials report increased frequency of military activities in the Taiwan Strait region.",
            "source_url": "https://example.com/taiwan1", 
            "region": "Taiwan Strait",
            "topic": "Military"
        },
        {
            "title": "Maritime Security Operations in Southeast Asia",
            "body": "Coast guard vessels from multiple nations conducted joint maritime security operations. The operations focus on combating piracy and illegal fishing in Southeast Asian waters.",
            "source_url": "https://example.com/maritime1",
            "region": "Southeast Asia", 
            "topic": "Security"
        }
    ]
    
    print(f"📤 Sending {len(test_articles)} test articles...")
    
    # Send all articles
    sent_articles = []
    for i, article in enumerate(test_articles, 1):
        print(f"[{i}/{len(test_articles)}] Sending: {article['title']}")
        result = client.send_article(**article)
        if result.get('status') == 'queued':
            sent_articles.append(result['id'])
            print(f"   ✅ Queued with ID: {result['id']}")
        else:
            print(f"   ❌ Failed: {result}")
    
    print(f"\n📋 Successfully sent {len(sent_articles)} articles")
    
    # Wait for preprocessing
    print("\n⏳ Waiting for preprocessing (10 seconds)...")
    time.sleep(10)
    
    # Check status
    status = client.get_status()
    print(f"📊 Status after preprocessing: {status}")
    
    # Trigger clustering manually
    print("\n🔄 Triggering clustering...")
    try:
        from app.tasks.cluster import run_clustering
        cluster_task = run_clustering.delay()
        print(f"   ✅ Clustering task queued: {cluster_task.id}")
    except Exception as e:
        print(f"   ❌ Clustering failed: {e}")
    
    # Wait for clustering
    print("\n⏳ Waiting for clustering (15 seconds)...")
    time.sleep(15)
    
    # Check clustering results
    print("\n🔍 Checking clustering results...")
    try:
        response = requests.get("http://localhost:8000/reports/quick")
        if response.status_code == 200:
            report = response.json()
            print(f"📊 Quick report: {report}")
        else:
            print(f"❌ Report failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Report error: {e}")
    
    # Trigger summarization
    print("\n📝 Triggering summarization...")
    try:
        from app.tasks.summarize import summarize_all_pending_clusters
        summary_task = summarize_all_pending_clusters.delay()
        print(f"   ✅ Summarization task queued: {summary_task.id}")
    except Exception as e:
        print(f"   ❌ Summarization failed: {e}")
    
    # Wait for summarization
    print("\n⏳ Waiting for summarization (10 seconds)...")
    time.sleep(10)
    
    # Final status check
    print("\n🏁 Final Status Check")
    print("-" * 30)
    
    final_status = client.get_status()
    print(f"📊 Final status: {final_status}")
    
    # Check final report
    try:
        response = requests.get("http://localhost:8000/reports/quick")
        if response.status_code == 200:
            final_report = response.json()
            print(f"📋 Final report: {final_report}")
            
            if final_report.get('clusters'):
                print(f"✅ SUCCESS: Found {len(final_report['clusters'])} clusters")
                for i, cluster in enumerate(final_report['clusters'], 1):
                    print(f"   Cluster {i}: {cluster.get('theme', 'Unknown')} ({cluster.get('article_count', 0)} articles)")
            else:
                print("⚠️  No clusters found in final report")
        else:
            print(f"❌ Final report failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Final report error: {e}")
    
    print("\n🎉 Pipeline test complete!")

if __name__ == "__main__":
    test_complete_pipeline() 