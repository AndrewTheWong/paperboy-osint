#!/usr/bin/env python3
"""
Supabase service for database operations
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.supabase_client import get_supabase_client
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def store_article(article_id: str, title: str, raw_text: str, cleaned_text: str, 
                 embedding: List[float], region: Optional[str] = None, 
                 topic: Optional[str] = None, source_url: str = "", 
                 tags: List[str] = None, tag_categories: Dict = None,
                 entities: List[str] = None, confidence_score: float = 0.0,
                 priority_level: str = "LOW", cluster_id: str = None,
                 cluster_label: str = None, cluster_description: str = None,
                 title_original: str = None, content_original: str = None,
                 title_language: str = "en", content_language: str = "en") -> bool:
    """
    Store article in Supabase articles table with enhanced schema
    """
    try:
        supabase = get_supabase_client()
        
        # Basic validation
        if not title or not cleaned_text:
            logger.warning(f"🚫 Skipping article {article_id}: missing title or content")
            return False
        
        # Only filter out obvious test content
        text_to_check = (title + " " + cleaned_text).lower()
        if 'stress test' in text_to_check or 'synthetic test' in text_to_check:
            logger.warning(f"🚫 Skipping article {article_id}: detected as stress test")
            return False
        
        # Prepare article data with clean schema including translation fields
        article_data = {
            'title': title,
            'body': raw_text,
            'cleaned': cleaned_text,
            'url': source_url,
            'source_name': region or 'Unknown',
            'published_at': datetime.now().isoformat(),
            'embedding': embedding,
            'tags': tags or [],
            'entities': entities or [],
            'cluster_id': cluster_id,
            'title_original': title_original or title,
            'content_original': content_original or raw_text,
            'title_language': title_language,
            'content_language': content_language
        }

        logger.info(f"[DEBUG] Upserting article_data: {article_data}")
        result = supabase.table('articles').upsert(article_data).execute()
        
        if result.data:
            db_id = result.data[0]['id']
            logger.info(f"💾 Stored article {article_id} to Supabase with DB ID {db_id} (tags: {len(tags or [])})")
            
            # DETAILED LOGGING: Print what was actually stored
            logger.info(f"🔍 [DETAILED-STORAGE] Article {article_id} stored with:")
            logger.info(f"   Title: {title}")
            logger.info(f"   Region: {region}")
            logger.info(f"   Topic: {topic}")
            logger.info(f"   Tags: {tags}")
            logger.info(f"   Entities: {entities}")
            logger.info(f"   Tag Categories: {tag_categories}")
            logger.info(f"   Priority: {priority_level}")
            logger.info(f"   Cluster ID: {cluster_id}")
            logger.info(f"   Cluster Label: {cluster_label}")
            logger.info(f"   Embedding Dimensions: {len(embedding) if embedding else 0}")
            
            return db_id
        else:
            logger.error(f"❌ Failed to store article {article_id}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error storing article {article_id}: {e}")
        return False

def save_cluster(cluster_id: str, article_ids: List[str], status: str = 'pending', theme: str = None, summary: str = None, top_tags=None, top_entities=None, representative_article_id=None) -> bool:
    """
    Save cluster to Supabase clusters table
    Args:
        cluster_id: Unique cluster identifier
        article_ids: List of article IDs in cluster
        status: Cluster status
        theme: Cluster theme/topic
        summary: Cluster summary text
        top_tags: List of top tags (from cluster analysis)
        top_entities: List of top entities (from cluster analysis)
        representative_article_id: UUID of representative article
    Returns:
        bool: Success status
    """
    try:
        supabase = get_supabase_client()
        
        # Convert cluster_id to integer if it's a string
        cluster_id_int = None
        if isinstance(cluster_id, str) and cluster_id.startswith('cluster_'):
            try:
                cluster_id_int = int(cluster_id.replace('cluster_', ''))
            except ValueError:
                cluster_id_int = None
        else:
            try:
                cluster_id_int = int(cluster_id)
            except (ValueError, TypeError):
                cluster_id_int = None
        
        if cluster_id_int is None:
            logger.error(f"❌ Invalid cluster_id: {cluster_id}")
            return False
        
        # Ensure top_tags and top_entities are lists
        if top_tags is None:
            top_tags = []
        if top_entities is None:
            top_entities = []
        
        cluster_data = {
            'cluster_id': cluster_id_int,
            'top_tags': top_tags,
            'top_entities': top_entities,
            'summary': summary or f"Cluster {cluster_id}",
            'member_count': len(article_ids),
            'topic': theme or 'Unknown',
            'representative_article_id': representative_article_id,
            'created_at': datetime.now().isoformat()
        }
        
        result = supabase.table('clusters').upsert(cluster_data).execute()
        if result.data:
            logger.info(f"💾 Saved cluster {cluster_id} with {len(article_ids)} articles (topic: {theme})")
            return True
        else:
            logger.error(f"❌ Failed to save cluster {cluster_id}. Supabase response: {result}")
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

def get_articles_with_embeddings(limit: int = 200) -> List[Dict]:
    """
    Get articles with embeddings from Supabase for clustering
    
    Args:
        limit: Maximum number of articles to fetch
        
    Returns:
        List[Dict]: List of articles with embeddings and metadata
    """
    try:
        supabase = get_supabase_client()
        
        # Fetch articles with embeddings, excluding those already in clusters
        result = supabase.table('articles').select(
            'id, title, content, region, topic, tags, embedding, url'
        ).not_.is_('embedding', 'null').limit(limit).execute()
        
        if not result.data:
            logger.warning("No articles with embeddings found")
            return []
        
        logger.info(f"📊 Fetched {len(result.data)} articles with embeddings")
        return result.data
        
    except Exception as e:
        logger.error(f"❌ Error fetching articles with embeddings: {e}")
        return []

def get_cluster_articles(cluster_id: str) -> List[Dict]:
    """
    Get all articles that belong to a specific cluster
    
    Args:
        cluster_id: The cluster ID to get articles for
        
    Returns:
        List[Dict]: List of articles in the cluster with full metadata
    """
    try:
        supabase = get_supabase_client()
        
        # First get the cluster to find its article_ids
        cluster_result = supabase.table('clusters').select('article_ids, theme, summary').eq('id', cluster_id).execute()
        
        if not cluster_result.data:
            logger.warning(f"No cluster found with ID: {cluster_id}")
            return []
        
        cluster = cluster_result.data[0]
        article_ids = cluster.get('article_ids', [])
        
        if not article_ids:
            logger.warning(f"Cluster {cluster_id} has no articles")
            return []
        
        # Get the articles using the article_ids
        articles_result = supabase.table('articles').select(
            'id, title, content, region, topic, tags, url, created_at'
        ).in_('id', article_ids).execute()
        
        if not articles_result.data:
            logger.warning(f"No articles found for cluster {cluster_id}")
            return []
        
        logger.info(f"📊 Retrieved {len(articles_result.data)} articles for cluster {cluster_id}")
        return articles_result.data
        
    except Exception as e:
        logger.error(f"❌ Error getting cluster articles for {cluster_id}: {e}")
        return []

def get_all_clusters_with_articles() -> List[Dict]:
    """
    Get all clusters with their articles
    
    Returns:
        List[Dict]: List of clusters with their articles
    """
    try:
        supabase = get_supabase_client()
        
        # Get all clusters
        clusters_result = supabase.table('clusters').select('id, cluster_id, theme, summary, article_ids, created_at').execute()
        
        if not clusters_result.data:
            logger.warning("No clusters found")
            return []
        
        clusters_with_articles = []
        
        for cluster in clusters_result.data:
            cluster_id = cluster['id']
            article_ids = cluster.get('article_ids', [])
            
            if article_ids:
                # Get articles for this cluster
                articles_result = supabase.table('articles').select(
                    'id, title, content, region, topic, tags, url, created_at'
                ).in_('id', article_ids).execute()
                
                cluster['articles'] = articles_result.data or []
            else:
                cluster['articles'] = []
            
            clusters_with_articles.append(cluster)
        
        logger.info(f"📊 Retrieved {len(clusters_with_articles)} clusters with articles")
        return clusters_with_articles
        
    except Exception as e:
        logger.error(f"❌ Error getting clusters with articles: {e}")
        return []

def save_article(article_data: Dict) -> bool:
    """
    Save article to database
    
    Args:
        article_data: Article data dictionary
        
    Returns:
        bool: Success status
    """
    try:
        import uuid
        supabase = get_supabase_client()
        
        # Ensure required fields
        if 'id' not in article_data:
            article_data['id'] = str(uuid.uuid4())
        
        result = supabase.table('articles').upsert(article_data).execute()
        
        if result.data:
            logger.info(f"💾 Saved article {article_data.get('id', 'unknown')}")
            return True
        else:
            logger.error(f"❌ Failed to save article")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error saving article: {e}")
        return False

def clear_databases():
    """
    Clear articles and clusters tables
    """
    try:
        supabase = get_supabase_client()
        
        logger.info("🗑️ Clearing databases...")
        
        # Clear clusters table
        logger.info("Clearing clusters table...")
        result = supabase.table('clusters').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        logger.info(f"✅ Cleared {len(result.data) if result.data else 0} clusters")
        
        # Clear articles table
        logger.info("Clearing articles table...")
        result = supabase.table('articles').delete().neq('id', '00000000-0000-0000-0000-000000000000').execute()
        logger.info(f"✅ Cleared {len(result.data) if result.data else 0} articles")
        
        logger.info("✅ Databases cleared successfully")
        
    except Exception as e:
        logger.error(f"❌ Error clearing databases: {e}")
        raise 

def get_all_articles() -> List[Dict]:
    """
    Get all articles from Supabase with all fields
    
    Returns:
        List[Dict]: List of all articles with full data
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles').select('*').execute()
        
        if result.data:
            logger.info(f"📊 Retrieved {len(result.data)} articles from Supabase")
            return result.data
        else:
            logger.warning("📊 No articles found in Supabase")
            return []
            
    except Exception as e:
        logger.error(f"❌ Error getting all articles: {e}")
        return []

def update_article_cluster(article_id: str, cluster_id: str, cluster_label: str = None, cluster_description: str = None) -> bool:
    """
    Update article with cluster assignment
    
    Args:
        article_id: Article identifier
        cluster_id: Cluster identifier
        cluster_label: Cluster label
        cluster_description: Cluster description
        
    Returns:
        bool: Success status
    """
    try:
        supabase = get_supabase_client()
        
        update_data = {
            'cluster_id': cluster_id
        }
        
        if cluster_label:
            update_data['cluster_label'] = cluster_label
        if cluster_description:
            update_data['cluster_description'] = cluster_description
        
        result = supabase.table('articles').update(update_data).eq('id', article_id).execute()
        
        if result.data:
            logger.info(f"💾 Updated article {article_id} with cluster {cluster_id}")
            return True
        else:
            logger.error(f"❌ Failed to update article {article_id} with cluster")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error updating article {article_id}: {e}")
        return False 