#!/usr/bin/env python3
"""
Check articles and their cluster assignments
"""

from utils.supabase_client import get_supabase_client

def check_articles():
    supabase = get_supabase_client()
    
    # Get all articles
    result = supabase.table('articles').select('id, title, cluster_id, tags, region, topic').execute()
    
    print("📰 Articles in database:")
    print("=" * 50)
    
    for article in result.data:
        print(f"  Title: {article['title']}")
        print(f"    Cluster ID: {article.get('cluster_id', 'None')}")
        print(f"    Region: {article.get('region', 'None')}")
        print(f"    Topic: {article.get('topic', 'None')}")
        tags = article.get('tags')
        if tags is None:
            tags = []
        print(f"    Tags: {len(tags)} tags")
        print()
    
    # Check clusters table
    clusters_result = supabase.table('clusters').select('*').execute()
    
    print("🔍 Clusters in database:")
    print("=" * 50)
    
    if clusters_result.data:
        for cluster in clusters_result.data:
            print(f"  Cluster ID: {cluster.get('cluster_id')}")
            print(f"    Theme: {cluster.get('theme', 'None')}")
            article_ids = cluster.get('article_ids')
            if article_ids is None:
                article_ids = []
            print(f"    Article count: {len(article_ids)}")
            print(f"    Status: {cluster.get('status', 'None')}")
            print()
    else:
        print("  No clusters found")

if __name__ == "__main__":
    check_articles() 