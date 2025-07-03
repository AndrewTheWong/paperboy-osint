#!/usr/bin/env python3
"""
Check clusters table
"""

from utils.supabase_client import get_supabase_client

def check_clusters():
    supabase = get_supabase_client()
    
    # Get all clusters
    result = supabase.table('clusters').select('*').execute()
    
    print("🔍 Clusters in database:")
    print("=" * 50)
    
    if not result.data:
        print("  No clusters found")
        return
    
    for cluster in result.data:
        print(f"  Cluster ID: {cluster.get('cluster_id', 'None')}")
        print(f"    Theme: {cluster.get('theme', 'None')}")
        print(f"    Region: {cluster.get('region', 'None')}")
        print(f"    Topic: {cluster.get('topic', 'None')}")
        article_ids = cluster.get('article_ids', [])
        print(f"    Articles: {len(article_ids) if article_ids else 0}")
        print()

if __name__ == "__main__":
    check_clusters() 