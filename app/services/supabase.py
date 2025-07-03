#!/usr/bin/env python3
"""
Supabase service for database operations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from app.utils.supabase_client import get_supabase_client
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_article(article_id: str, title: str, raw_text: str, cleaned_text: str, 
                 embedding: List[float], region: Optional[str] = None, 
                 topic: Optional[str] = None, source_url: str = "") -> bool:
    """
    Store article in Supabase articles table
    
    Args:
        article_id: Unique article identifier
        title: Article title
        raw_text: Original article text
        cleaned_text: Cleaned article text
        embedding: Article embedding vector (not stored in current schema)
        region: Geographic region
        topic: Article topic
        source_url: Source URL
        
    Returns:
        bool: Success status
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('articles').insert({
            'title': title,
            'content': cleaned_text,
            'url': source_url,
            'source': region or 'Unknown',
            'published_at': datetime.now().isoformat()
        }).execute()
        
        if result.data:
            db_id = result.data[0]['id']
            logger.info(f"💾 Stored article {article_id} to Supabase with DB ID {db_id}")
            return db_id  # Return the auto-generated database ID
        else:
            logger.error(f"❌ Failed to store article {article_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error storing article {article_id}: {e}")
        return False

def save_cluster(cluster_id: str, article_ids: List[str], status: str = 'pending') -> bool:
    """
    Save cluster to Supabase clusters table
    
    Args:
        cluster_id: Unique cluster identifier
        article_ids: List of article IDs in cluster
        status: Cluster status
        
    Returns:
        bool: Success status
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('clusters').insert({
            'id': cluster_id,
            'article_ids': article_ids,
            'status': status,
            'created_at': datetime.now().isoformat()
        }).execute()
        
        if result.data:
            logger.info(f"💾 Saved cluster {cluster_id} with {len(article_ids)} articles")
            return True
        else:
            logger.error(f"❌ Failed to save cluster {cluster_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error saving cluster {cluster_id}: {e}")
        return False

def update_cluster_summary(cluster_id: str, summary: str, status: str = 'complete') -> bool:
    """
    Update cluster with summary and status
    
    Args:
        cluster_id: Cluster identifier
        summary: Generated summary
        status: New status
        
    Returns:
        bool: Success status
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('clusters').update({
            'summary': summary,
            'status': status
        }).eq('id', cluster_id).execute()
        
        if result.data:
            logger.info(f"💾 Updated cluster {cluster_id} with summary")
            return True
        else:
            logger.error(f"❌ Failed to update cluster {cluster_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error updating cluster {cluster_id}: {e}")
        return False

def get_articles_count() -> int:
    """
    Get total count of articles in database
    
    Returns:
        int: Total number of articles
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles').select('id', count='exact').execute()
        return result.count or 0
    except Exception as e:
        logger.error(f"❌ Error getting articles count: {e}")
        return 0

def get_unprocessed_count() -> int:
    """
    Get count of unprocessed articles (articles not in any cluster)
    
    Returns:
        int: Number of unprocessed articles
    """
    try:
        supabase = get_supabase_client()
        
        # Get all article IDs
        all_articles = supabase.table('articles').select('id').execute()
        if not all_articles.data:
            return 0
            
        all_article_ids = set(article['id'] for article in all_articles.data)
        
        # Get article IDs that are in clusters
        clusters = supabase.table('clusters').select('article_ids').execute()
        clustered_article_ids = set()
        if clusters.data:
            for cluster in clusters.data:
                if cluster.get('article_ids'):
                    clustered_article_ids.update(cluster['article_ids'])
        
        # Count articles not in any cluster
        unprocessed_count = len(all_article_ids - clustered_article_ids)
        return unprocessed_count
        
    except Exception as e:
        logger.error(f"❌ Error getting unprocessed count: {e}")
        return 0

def get_embeddings_batch(article_ids: List[str]) -> List[Dict]:
    """
    Get embeddings for a batch of articles
    
    Args:
        article_ids: List of article IDs
        
    Returns:
        List[Dict]: List of {id, embedding} dictionaries
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles').select('id, embedding').in_('id', article_ids).execute()
        return result.data or []
    except Exception as e:
        logger.error(f"❌ Error getting embeddings batch: {e}")
        return []

def get_articles_texts(article_ids: List[str]) -> List[str]:
    """
    Get cleaned text for articles
    
    Args:
        article_ids: List of article IDs
        
    Returns:
        List[str]: List of cleaned texts
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles').select('content').in_('id', article_ids).execute()
        if result.data:
            return [article['content'] for article in result.data if article.get('content')]
        return []
    except Exception as e:
        logger.error(f"❌ Error getting articles texts: {e}")
        return []

def get_pending_clusters() -> List[Dict]:
    """
    Get clusters with pending status
    
    Returns:
        List[Dict]: List of pending clusters
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('clusters').select('*').eq('status', 'pending').execute()
        return result.data or []
    except Exception as e:
        logger.error(f"❌ Error getting pending clusters: {e}")
        return [] 