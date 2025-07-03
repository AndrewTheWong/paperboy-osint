#!/usr/bin/env python3
"""
Test cluster labeling function
"""

from app.services.faiss_cluster import label_cluster

def test_label_cluster():
    print("🔍 Testing cluster labeling function...")
    
    # Test with a known cluster ID
    cluster_id = 545
    try:
        label, description = label_cluster(cluster_id)
        print(f"✅ Cluster {cluster_id}:")
        print(f"   Label: {label}")
        print(f"   Description: {description}")
    except Exception as e:
        print(f"❌ Error labeling cluster {cluster_id}: {e}")
    
    # Test with another cluster ID
    cluster_id = 996
    try:
        label, description = label_cluster(cluster_id)
        print(f"✅ Cluster {cluster_id}:")
        print(f"   Label: {label}")
        print(f"   Description: {description}")
    except Exception as e:
        print(f"❌ Error labeling cluster {cluster_id}: {e}")

if __name__ == "__main__":
    test_label_cluster() 