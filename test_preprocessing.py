#!/usr/bin/env python3
"""
Test preprocessing pipeline
"""

import time
from app.tasks.preprocess import preprocess_and_enqueue
from app.tasks.cluster import run_clustering
from app.tasks.summarize import summarize_all_pending_clusters

def test_preprocessing():
    """Test the preprocessing pipeline"""
    
    print("=" * 60)
    print("TESTING PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Test article data
    test_article = {
        "article_id": "test-article-001",
        "title": "Test Article - Taiwan Strait Tensions",
        "body": "Recent developments in the Taiwan Strait have raised concerns about regional stability. Military exercises and diplomatic tensions continue to escalate between major powers in the region.",
        "region": "Asia Pacific",
        "topic": "Geopolitics",
        "source_url": "https://example.com/test-article"
    }
    
    try:
        print("📤 Queuing preprocessing task...")
        
        # Queue preprocessing task
        result = preprocess_and_enqueue.delay(
            article_id=test_article["article_id"],
            title=test_article["title"],
            body=test_article["body"],
            region=test_article["region"],
            topic=test_article["topic"],
            source_url=test_article["source_url"]
        )
        
        print(f"✅ Preprocessing task queued: {result.id}")
        
        # Wait for task to complete
        print("⏳ Waiting for preprocessing to complete...")
        task_result = result.get(timeout=60)
        print(f"📊 Preprocessing result: {task_result}")
        
        # Wait a moment then trigger clustering
        print("⏳ Waiting 5 seconds before clustering...")
        time.sleep(5)
        
        print("🔍 Triggering clustering...")
        cluster_result = run_clustering.delay()
        print(f"✅ Clustering task queued: {cluster_result.id}")
        
        # Wait for clustering to complete
        print("⏳ Waiting for clustering to complete...")
        cluster_task_result = cluster_result.get(timeout=120)
        print(f"📊 Clustering result: {cluster_task_result}")
        
        # Wait a moment then trigger summarization
        print("⏳ Waiting 5 seconds before summarization...")
        time.sleep(5)
        
        print("📝 Triggering summarization...")
        summary_result = summarize_all_pending_clusters.delay()
        print(f"✅ Summarization task queued: {summary_result.id}")
        
        # Wait for summarization to complete
        print("⏳ Waiting for summarization to complete...")
        summary_task_result = summary_result.get(timeout=120)
        print(f"📊 Summarization result: {summary_task_result}")
        
        print("\n🎉 Pipeline test completed!")
        
    except Exception as e:
        print(f"❌ Error in pipeline test: {e}")
        import traceback
        traceback.print_exc()

def check_celery_status():
    """Check if Celery is working"""
    print("\n" + "=" * 60)
    print("CHECKING CELERY STATUS")
    print("=" * 60)
    
    try:
        from celery import current_app
        
        # Check if Celery is connected
        inspect = current_app.control.inspect()
        active_tasks = inspect.active()
        registered_tasks = inspect.registered()
        
        print(f"Active tasks: {active_tasks}")
        print(f"Registered tasks: {registered_tasks}")
        
        if active_tasks:
            print("✅ Celery is working and has active tasks")
        else:
            print("⚠️ Celery is working but no active tasks")
            
    except Exception as e:
        print(f"❌ Error checking Celery status: {e}")

if __name__ == "__main__":
    check_celery_status()
    test_preprocessing() 