#!/usr/bin/env python3
"""
Check cluster labels in the database
"""

from utils.supabase_client import get_supabase_client

def check_cluster_labels():
    supabase = get_supabase_client()
    
    # Get articles with cluster information
    result = supabase.table('articles').select('title, cluster_id, cluster_label, tags').limit(10).execute()
    
    print("🔍 Articles with cluster labels:")
    print("=" * 50)
    
    articles_with_labels = 0
    for article in result.data:
        print(f"  Title: {article['title'][:50]}...")
        print(f"    Cluster ID: {article.get('cluster_id', 'None')}")
        cluster_label = article.get('cluster_label')
        if cluster_label:
            articles_with_labels += 1
            print(f"    Cluster Label: {cluster_label}")
        else:
            print(f"    Cluster Label: None")
        tags = article.get('tags', [])
        print(f"    Tags: {len(tags) if tags else 0} tags")
        print()
    
    print(f"📊 Summary: {articles_with_labels}/{len(result.data)} articles have cluster labels")
    
    # Check clusters table
    clusters_result = supabase.table('clusters').select('cluster_id, theme, article_ids').limit(5).execute()
    
    print("\n🔍 Clusters table:")
    print("=" * 50)
    
    for cluster in clusters_result.data:
        print(f"  Cluster ID: {cluster.get('cluster_id', 'None')}")
        print(f"    Theme: {cluster.get('theme', 'None')}")
        article_ids = cluster.get('article_ids', [])
        print(f"    Articles: {len(article_ids) if article_ids else 0}")
        print()

if __name__ == "__main__":
    check_cluster_labels() 