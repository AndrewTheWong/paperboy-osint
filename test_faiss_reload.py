#!/usr/bin/env python3
"""
Test FAISS reload and cluster labeling
"""

from app.services.faiss_cluster import faiss_manager, label_cluster, reload_faiss

def test_faiss_reload():
    print("🔄 Testing FAISS reload and cluster labeling...")
    
    # Force reload FAISS
    print("📚 Reloading FAISS manager...")
    reload_faiss()
    
    print(f"✅ FAISS manager loaded:")
    print(f"   Total embeddings: {len(faiss_manager.embeddings)}")
    print(f"   Total articles: {len(faiss_manager.article_ids)}")
    print(f"   Total clusters: {len(set(faiss_manager.cluster_ids))}")
    print(f"   Articles with tags: {len([aid for aid, tags in faiss_manager.tags.items() if tags])}")
    
    # Test cluster labeling
    print("\n🔍 Testing cluster labeling...")
    
    cluster_id = "545"
    try:
        label, description = label_cluster(cluster_id)
        print(f"✅ Cluster {cluster_id}:")
        print(f"   Label: {label}")
        print(f"   Description: {description}")
    except Exception as e:
        print(f"❌ Error labeling cluster {cluster_id}: {e}")
    
    cluster_id = "996"
    try:
        label, description = label_cluster(cluster_id)
        print(f"✅ Cluster {cluster_id}:")
        print(f"   Label: {label}")
        print(f"   Description: {description}")
    except Exception as e:
        print(f"❌ Error labeling cluster {cluster_id}: {e}")

if __name__ == "__main__":
    test_faiss_reload() 