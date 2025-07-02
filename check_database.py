#!/usr/bin/env python3
"""
Check database state for articles and clusters
"""

from utils.supabase_client import get_supabase_client
from datetime import datetime, timedelta

def check_database():
    """Check articles and clusters in database"""
    try:
        supabase = get_supabase_client()
        
        print("=" * 60)
        print("DATABASE STATUS CHECK")
        print("=" * 60)
        
        # Check articles
        print("\n📰 ARTICLES:")
        articles_result = supabase.table('articles').select('id, title, created_at').limit(10).execute()
        articles = articles_result.data if articles_result.data else []
        print(f"Total articles: {len(articles)}")
        
        if articles:
            print("Recent articles:")
            for article in articles[:5]:
                title = article.get('title', 'No title')[:50] + '...' if len(article.get('title', '')) > 50 else article.get('title', 'No title')
                created = article.get('created_at', 'Unknown')
                print(f"  - {title}")
                print(f"    ID: {article.get('id')}, Created: {created}")
        else:
            print("  No articles found")
        
        # Check clusters
        print("\n🔍 CLUSTERS:")
        clusters_result = supabase.table('clusters').select('id, theme, status, article_ids, created_at').limit(10).execute()
        clusters = clusters_result.data if clusters_result.data else []
        print(f"Total clusters: {len(clusters)}")
        
        if clusters:
            print("Recent clusters:")
            for cluster in clusters[:5]:
                theme = cluster.get('theme', 'No theme')[:50] + '...' if len(cluster.get('theme', '')) > 50 else cluster.get('theme', 'No theme')
                status = cluster.get('status', 'Unknown')
                article_count = len(cluster.get('article_ids', []))
                created = cluster.get('created_at', 'Unknown')
                print(f"  - {theme}")
                print(f"    ID: {cluster.get('id')}, Status: {status}, Articles: {article_count}, Created: {created}")
        else:
            print("  No clusters found")
        
        # Check cluster status distribution
        if clusters:
            status_counts = {}
            for cluster in clusters:
                status = cluster.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            print("\n📊 Cluster Status Distribution:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
        
        # Check recent activity (last 24 hours)
        print("\n⏰ RECENT ACTIVITY (Last 24 hours):")
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.isoformat()
        
        recent_articles = supabase.table('articles').select('id').gte('created_at', yesterday_str).execute()
        recent_clusters = supabase.table('clusters').select('id').gte('created_at', yesterday_str).execute()
        
        print(f"  Articles created: {len(recent_articles.data) if recent_articles.data else 0}")
        print(f"  Clusters created: {len(recent_clusters.data) if recent_clusters.data else 0}")
        
        print("\n" + "=" * 60)
        
    except Exception as e:
        print(f"❌ Error checking database: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_database() 