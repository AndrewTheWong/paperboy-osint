#!/usr/bin/env python3
"""
Manual Pipeline Test - Runs pipeline synchronously without Celery
"""

from app.tasks.preprocess import preprocess_and_enqueue
from app.tasks.cluster import run_clustering  
from app.tasks.summarize import summarize_all_pending_clusters
import requests

def test_manual_pipeline():
    """Test pipeline by calling functions directly (synchronously)"""
    
    print("🔧 Manual Pipeline Test (No Celery)")
    print("=" * 50)
    
    # Test 1: Manual preprocessing
    print("\n1️⃣ Testing Manual Preprocessing...")
    try:
        result = preprocess_and_enqueue(
            'manual-test-001',
            'Manual Test Article',
            'This is a test article for manual pipeline processing. It should be processed through cleaning, embedding, and storage.',
            'Asia',
            'Military',
            'http://manual.test/001'
        )
        print(f"✅ Preprocessing result: {result}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return
    
    # Test 2: Manual clustering
    print("\n2️⃣ Testing Manual Clustering...")
    try:
        cluster_result = run_clustering()
        print(f"✅ Clustering result: {cluster_result}")
    except Exception as e:
        print(f"❌ Clustering failed: {e}")
    
    # Test 3: Manual summarization  
    print("\n3️⃣ Testing Manual Summarization...")
    try:
        summary_result = summarize_all_pending_clusters()
        print(f"✅ Summarization result: {summary_result}")
    except Exception as e:
        print(f"❌ Summarization failed: {e}")
    
    # Test 4: Check API endpoints
    print("\n4️⃣ Testing API Endpoints...")
    try:
        status_response = requests.get("http://localhost:8000/ingest/status")
        print(f"📊 Status: {status_response.json()}")
        
        report_response = requests.get("http://localhost:8000/reports/quick")
        print(f"📋 Report: {report_response.json()}")
    except Exception as e:
        print(f"❌ API test failed: {e}")
    
    print("\n🏁 Manual pipeline test complete!")

if __name__ == "__main__":
    test_manual_pipeline() 