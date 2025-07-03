#!/usr/bin/env python3
"""
Check tags for clusters
"""

from utils.supabase_client import get_supabase_client

def check_cluster_tags():
    supabase = get_supabase_client()
    
    # Check cluster 545
    result = supabase.table('articles').select('title, tags, cluster_id').eq('cluster_id', '545').execute()
    
    print("🔍 Articles in cluster 545:")
    print("=" * 50)
    
    for article in result.data:
        title = article['title'][:50] + "..." if len(article['title']) > 50 else article['title']
        tags = article.get('tags', [])
        print(f"  Title: {title}")
        print(f"    Tags: {len(tags)} tags")
        if tags:
            print(f"    Sample tags: {tags[:3]}")
        print()
    
    # Check cluster 996
    result = supabase.table('articles').select('title, tags, cluster_id').eq('cluster_id', '996').execute()
    
    print("🔍 Articles in cluster 996:")
    print("=" * 50)
    
    for article in result.data:
        title = article['title'][:50] + "..." if len(article['title']) > 50 else article['title']
        tags = article.get('tags', [])
        print(f"  Title: {title}")
        print(f"    Tags: {len(tags)} tags")
        if tags:
            print(f"    Sample tags: {tags[:3]}")
        print()

if __name__ == "__main__":
    check_cluster_tags() 