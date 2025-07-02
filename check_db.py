#!/usr/bin/env python3
"""
Check Database - Verify articles are stored in Supabase
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from utils.supabase_client import get_supabase_client
    supabase = get_supabase_client()
except ImportError:
    from supabase import create_client
    SUPABASE_URL = "http://localhost:54321"
    SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def check_database():
    """Check articles in database"""
    print("🔍 Checking Supabase database...")
    
    try:
        # Get total count
        count_result = supabase.table("articles").select("id", count="exact").execute()
        total_count = count_result.count if count_result.count is not None else 0
        
        print(f"📊 Total articles in database: {total_count}")
        
        if total_count > 0:
            # Get recent articles
            articles_result = supabase.table("articles").select(
                "id, title, region, topic, cluster_id, processed_by, inserted_at"
            ).order("inserted_at", desc=True).limit(5).execute()
            
            articles = articles_result.data if articles_result.data else []
            
            print(f"\n📋 Recent articles:")
            for i, article in enumerate(articles, 1):
                print(f"  {i}. {article.get('title', 'No title')}")
                print(f"     ID: {article.get('id')}")
                print(f"     Region: {article.get('region', 'Unknown')}")
                print(f"     Topic: {article.get('topic', 'Unknown')}")
                print(f"     Cluster: {article.get('cluster_id', 'None')}")
                print(f"     Processed by: {article.get('processed_by', 'Unknown')}")
                print(f"     Inserted: {article.get('inserted_at', 'Unknown')}")
                print()
        else:
            print("❌ No articles found in database")
            
    except Exception as e:
        print(f"❌ Error checking database: {e}")

if __name__ == "__main__":
    check_database() 