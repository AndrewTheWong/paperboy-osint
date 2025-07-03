#!/usr/bin/env python3
"""
Simple Full Pipeline Test - Tests the complete StraitWatch pipeline end-to-end
"""

import time
import requests
import json

def test_full_pipeline():
    """Test the complete article processing pipeline"""
    
    print("🚀 StraitWatch Full Pipeline Test")
    print("=" * 50)
    
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
        try:
            response = requests.post("http://localhost:8000/ingest/", json=article)
            if response.status_code == 200:
                result = response.json()
                if result.get('status') == 'queued':
                    sent_articles.append(result['id'])
                    print(f"   ✅ Queued with ID: {result['id']}")
                else:
                    print(f"   ❌ Failed: {result}")
            else:
                print(f"   ❌ HTTP Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print(f"\n📋 Successfully sent {len(sent_articles)} articles")
    
    # Wait for preprocessing
    print("\n⏳ Waiting for preprocessing (15 seconds)...")
    time.sleep(15)
    
    # Check status
    try:
        status_response = requests.get("http://localhost:8000/ingest/status")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"📊 Status after preprocessing: {status}")
        else:
            print(f"❌ Status check failed: {status_response.status_code}")
    except Exception as e:
        print(f"❌ Status check error: {e}")
    
    # Trigger clustering manually
    print("\n🔄 Triggering clustering...")
    try:
        from app.tasks.cluster import run_clustering
        cluster_task = run_clustering.delay()
        print(f"   ✅ Clustering task queued: {cluster_task.id}")
    except Exception as e:
        print(f"   ❌ Clustering failed: {e}")
    
    # Wait for clustering
    print("\n⏳ Waiting for clustering (20 seconds)...")
    time.sleep(20)
    
    # Check clustering results
    print("\n🔍 Checking clustering results...")
    try:
        response = requests.get("http://localhost:8000/reports/quick")
        if response.status_code == 200:
            report = response.json()
            print(f"📊 Quick report: {json.dumps(report, indent=2)}")
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
    print("\n⏳ Waiting for summarization (15 seconds)...")
    time.sleep(15)
    
    # Final status check
    print("\n🏁 Final Status Check")
    print("-" * 30)
    
    try:
        final_status = requests.get("http://localhost:8000/ingest/status")
        if final_status.status_code == 200:
            status_data = final_status.json()
            print(f"📊 Final status: {json.dumps(status_data, indent=2)}")
        else:
            print(f"❌ Final status failed: {final_status.status_code}")
    except Exception as e:
        print(f"❌ Final status error: {e}")
    
    # Check final report
    try:
        response = requests.get("http://localhost:8000/reports/quick")
        if response.status_code == 200:
            final_report = response.json()
            print(f"📋 Final report: {json.dumps(final_report, indent=2)}")
            
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
    test_full_pipeline() 