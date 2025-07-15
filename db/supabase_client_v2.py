#!/usr/bin/env python3
"""
Supabase service for database operations - Redesigned Schema v2
Separates raw scraped articles from processed enrichment data
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from config.supabase_client import get_supabase_client
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Batch processing configuration
BATCH_SIZE = 100  # Increased from 50 to 100 for better throughput

def store_raw_article(source_url: str, title: str, text_content: str, 
                     raw_html: str = None, language: str = None, 
                     published_date: datetime = None) -> Optional[str]:
    """
    Store raw scraped article to articles table
    Returns: article_id if successful, None if failed
    """
    try:
        supabase = get_supabase_client()
        
        # Basic validation
        if not source_url or not text_content:
            logger.warning(f"Skipping article: missing source_url or text_content")
            return None
        
        # Prepare raw article data
        article_data = {
            'source_url': source_url,
            'title': title,
            'text_content': text_content,
            'raw_html': raw_html,
            'language': language,
            'published_date': published_date.isoformat() if published_date else None,
            'created_at': datetime.now().isoformat()
        }

        logger.info(f"[RAW] Storing article: {title[:50]}...")
        result = supabase.table('articles').insert(article_data).execute()
        
        if result.data:
            article_id = result.data[0]['id']
            logger.info(f"✅ Stored raw article {article_id}")
            return article_id
        else:
            logger.error(f"❌ Failed to store raw article")
            return None
            
    except Exception as e:
        logger.error(f"Error storing raw article: {e}")
        return None

def store_raw_articles_batch(articles: List[Dict[str, Any]], batch_size: int = BATCH_SIZE) -> int:
    """
    Store a batch of raw articles in Supabase articles table.
    Processes articles in chunks for better performance.
    Returns: number of successfully inserted articles
    """
    try:
        supabase = get_supabase_client()
        if not articles:
            return 0
            
        # Filter out invalid articles
        valid_articles = [
            {
                'source_url': a.get('url'),
                'title': a.get('title'),
                'text_content': a.get('content'),
                'raw_html': a.get('raw_html'),
                'language': a.get('language'),
                'published_date': a.get('date'),
                'created_at': datetime.now().isoformat()
            }
            for a in articles
            if a.get('url') and a.get('content')
        ]
        
        if not valid_articles:
            return 0
            
        total_inserted = 0
        
        # Process in batches for better performance
        for i in range(0, len(valid_articles), batch_size):
            batch = valid_articles[i:i + batch_size]
            logger.info(f"[RAW-BATCH] Inserting batch {i//batch_size + 1}: {len(batch)} articles...")
            
            try:
                result = supabase.table('articles').insert(batch).execute()
                if result.data:
                    batch_inserted = len(result.data)
                    total_inserted += batch_inserted
                    logger.info(f"✅ Batch {i//batch_size + 1} inserted {batch_inserted} articles")
                else:
                    logger.error(f"❌ Batch {i//batch_size + 1} insert failed")
            except Exception as e:
                logger.error(f"Error in batch {i//batch_size + 1}: {e}")
                continue
                
        logger.info(f"✅ Total inserted: {total_inserted} articles")
        return total_inserted
        
    except Exception as e:
        logger.error(f"Error in batch insert: {e}")
        return 0

def upsert_processed_article(article_id: str, **kwargs) -> bool:
    """
    Upsert processed article data to articles_processed table
    Uses article_id as unique constraint
    """
    try:
        supabase = get_supabase_client()
        
        # Prepare processed article data
        processed_data = {
            'article_id': article_id,
            'updated_at': datetime.now().isoformat()
        }
        
        # Add provided fields
        for key, value in kwargs.items():
            if value is not None:
                processed_data[key] = value
        
        logger.info(f"[PROCESSED] Upserting article {article_id} with fields: {list(kwargs.keys())}")
        result = supabase.table('articles_processed').upsert(
            processed_data, 
            on_conflict='article_id'
        ).execute()
        
        if result.data:
            logger.info(f"✅ Upserted processed article {article_id}")
            return True
        else:
            logger.error(f"❌ Failed to upsert processed article {article_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error upserting processed article {article_id}: {e}")
        return False

def translate_article(article_id: str, translated_text: str, translated_title: str = None,
                    source_language: str = None, target_language: str = 'en') -> bool:
    """
    Update article with translation data
    """
    return upsert_processed_article(
        article_id=article_id,
        translated_text=translated_text,
        translated_title=translated_title,
        source_language=source_language,
        target_language=target_language
    )

def embed_article(article_id: str, embedding: List[float]) -> bool:
    """
    Update article with embedding data
    """
    return upsert_processed_article(
        article_id=article_id,
        embedding=embedding
    )

def tag_article(article_id: str, tags: List[str] = None, entities: List[str] = None,
               tag_categories: Dict = None) -> bool:
    """
    Update article with tagging and NER data
    """
    return upsert_processed_article(
        article_id=article_id,
        tags=tags or [],
        entities=entities or [],
        tag_categories=tag_categories or {}
    )

def cluster_article(article_id: str, cluster_id: str) -> bool:
    """
    Update article with cluster assignment
    """
    return upsert_processed_article(
        article_id=article_id,
        cluster_id=cluster_id
    )

def update_processed_article_cluster(processed_id: str, cluster_id: str) -> bool:
    """
    Update processed article with cluster assignment using processed_id
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles_processed').update({
            'cluster_id': cluster_id,
            'updated_at': datetime.now().isoformat()
        }).eq('id', processed_id).execute()
        
        if result.data:
            logger.info(f"✅ Assigned processed article {processed_id} to cluster {cluster_id}")
            return True
        else:
            logger.error(f"❌ Failed to assign processed article {processed_id} to cluster {cluster_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error assigning processed article {processed_id} to cluster {cluster_id}: {e}")
        return False

def summarize_article(article_id: str, summary: str) -> bool:
    """
    Update article with summary data
    """
    return upsert_processed_article(
        article_id=article_id,
        summary=summary
    )

def update_article_summary(article_id: str, summary: str) -> bool:
    """
    Update processed article with summary
    """
    return upsert_processed_article(
        article_id=article_id,
        summary=summary
    )

def get_raw_article(article_id: str) -> Optional[Dict]:
    """
    Get raw article by ID
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles').select('*').eq('id', article_id).single().execute()
        
        if result.data:
            return result.data
        else:
            logger.warning(f"Raw article {article_id} not found")
            return None
            
    except Exception as e:
        logger.error(f"Error getting raw article {article_id}: {e}")
        return None

def get_processed_article(article_id: str) -> Optional[Dict]:
    """
    Get processed article by article_id
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles_processed').select('*').eq('article_id', article_id).single().execute()
        
        if result.data:
            return result.data
        else:
            logger.warning(f"Processed article {article_id} not found")
            return None
            
    except Exception as e:
        logger.error(f"Error getting processed article {article_id}: {e}")
        return None

def get_articles_for_clustering(limit: int = 100) -> List[Dict]:
    """
    Get articles that have embeddings but no cluster assignment
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles_processed').select(
            'id, article_id, embedding, tags, entities'
        ).not_.is_('embedding', 'null').is_('cluster_id', 'null').limit(limit).execute()
        
        if result.data:
            logger.info(f"Found {len(result.data)} articles for clustering")
            return result.data
        else:
            logger.info("No articles found for clustering")
            return []
            
    except Exception as e:
        logger.error(f"Error getting articles for clustering: {e}")
        return []

def create_cluster(label: str, description: str = None, center_embedding: List[float] = None,
                  member_count: int = 0, representative_article_id: str = None) -> Optional[str]:
    """
    Create a new cluster
    Returns: cluster_id if successful, None if failed
    """
    try:
        supabase = get_supabase_client()
        
        cluster_data = {
            'label': label,
            'description': description,
            'center_embedding': center_embedding,
            'member_count': member_count,
            'representative_article_id': representative_article_id,
            'status': 'pending',
            'created_at': datetime.now().isoformat()
        }
        
        logger.info(f"[CLUSTER] Creating cluster: {label}")
        result = supabase.table('clusters').insert(cluster_data).execute()
        
        if result.data:
            cluster_id = result.data[0]['id']
            logger.info(f"✅ Created cluster {cluster_id}")
            return cluster_id
        else:
            logger.error(f"❌ Failed to create cluster")
            return None
            
    except Exception as e:
        logger.error(f"Error creating cluster: {e}")
        return None

def update_cluster_members(cluster_id: str, member_count: int, representative_article_id: str = None) -> bool:
    """
    Update cluster with member count and representative article
    """
    try:
        supabase = get_supabase_client()
        
        update_data = {
            'member_count': member_count,
            'updated_at': datetime.now().isoformat()
        }
        
        if representative_article_id:
            update_data['representative_article_id'] = representative_article_id
        
        result = supabase.table('clusters').update(update_data).eq('id', cluster_id).execute()
        
        if result.data:
            logger.info(f"✅ Updated cluster {cluster_id} with {member_count} members")
            return True
        else:
            logger.error(f"❌ Failed to update cluster {cluster_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating cluster {cluster_id}: {e}")
        return False

def get_pending_clusters() -> List[Dict]:
    """
    Get clusters that need summarization
    """
    try:
        supabase = get_supabase_client()
        result = supabase.table('clusters').select('*').eq('status', 'pending').execute()
        
        if result.data:
            logger.info(f"Found {len(result.data)} pending clusters")
            return result.data
        else:
            logger.info("No pending clusters found")
            return []
            
    except Exception as e:
        logger.error(f"Error getting pending clusters: {e}")
        return []

def update_cluster_summary(cluster_id: str, summary: str, status: str = 'complete') -> bool:
    """
    Update cluster with summary and status
    """
    try:
        supabase = get_supabase_client()
        
        result = supabase.table('clusters').update({
            'summary': summary,
            'status': status,
            'updated_at': datetime.now().isoformat()
        }).eq('id', cluster_id).execute()
        
        if result.data:
            logger.info(f"✅ Updated cluster {cluster_id} with summary")
            return True
        else:
            logger.error(f"❌ Failed to update cluster {cluster_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating cluster {cluster_id}: {e}")
        return False

def update_cluster_tags_and_entities(cluster_id: str, top_tags: List[str] = None, top_entities: List[str] = None) -> bool:
    """
    Update cluster with top tags and entities
    """
    try:
        supabase = get_supabase_client()
        
        update_data = {
            'updated_at': datetime.now().isoformat()
        }
        
        if top_tags is not None:
            update_data['top_tags'] = top_tags
        if top_entities is not None:
            update_data['top_entities'] = top_entities
        
        result = supabase.table('clusters').update(update_data).eq('id', cluster_id).execute()
        
        if result.data:
            logger.info(f"✅ Updated cluster {cluster_id} with tags and entities")
            return True
        else:
            logger.error(f"❌ Failed to update cluster {cluster_id}")
            return False
            
    except Exception as e:
        logger.error(f"Error updating cluster {cluster_id}: {e}")
        return False

def get_articles_count() -> int:
    """Get total count of raw articles"""
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles').select('id', count='exact').execute()
        return result.count or 0
    except Exception as e:
        logger.error(f"Error getting articles count: {e}")
        return 0

def get_processed_articles_count() -> int:
    """Get total count of processed articles"""
    try:
        supabase = get_supabase_client()
        result = supabase.table('articles_processed').select('id', count='exact').execute()
        return result.count or 0
    except Exception as e:
        logger.error(f"Error getting processed articles count: {e}")
        return 0

def get_clusters_count() -> int:
    """Get total count of clusters"""
    try:
        supabase = get_supabase_client()
        result = supabase.table('clusters').select('id', count='exact').execute()
        return result.count or 0
    except Exception as e:
        logger.error(f"Error getting clusters count: {e}")
        return 0 

def get_unprocessed_articles(limit: int = 50) -> List[Dict]:
    """
    Get articles that haven't been processed yet (no entry in articles_processed table)
    This prevents duplicate processing.
    
    Args:
        limit: Maximum number of articles to return
        
    Returns:
        List of article dictionaries
    """
    try:
        supabase = get_supabase_client()
        # Try RPC first
        try:
            query = f"""
            SELECT a.* 
            FROM articles a 
            LEFT JOIN articles_processed ap ON a.id = ap.article_id 
            WHERE ap.article_id IS NULL 
            ORDER BY a.created_at DESC 
            LIMIT {limit}
            """
            result = supabase.rpc('exec_sql', {'sql': query}).execute()
            print(f"[DIAG] Supabase RPC result: {result}")
            if result.data:
                logger.info(f"Found {len(result.data)} unprocessed articles (RPC)")
                return result.data
            else:
                logger.info("No unprocessed articles found (RPC)")
        except Exception as e:
            logger.error(f"[DIAG] RPC failed: {e}")
            print(f"[DIAG] RPC failed: {e}")
        # Fallback: Python-side diff
        raw = supabase.table('articles').select('*').order('created_at', desc=True).limit(limit).execute()
        processed = supabase.table('articles_processed').select('article_id').execute()
        processed_ids = {row['article_id'] for row in (processed.data or [])}
        unprocessed = [a for a in (raw.data or []) if a['id'] not in processed_ids]
        logger.info(f"Found {len(unprocessed)} unprocessed articles (fallback)")
        print(f"[DIAG] Fallback unprocessed: {unprocessed}")
        return unprocessed[:limit]
    except Exception as e:
        logger.error(f"Error getting unprocessed articles: {e}")
        print(f"[DIAG] Error in get_unprocessed_articles: {e}")
        return [] 

def store_article(*args, **kwargs):
    """Compatibility: Accept legacy article fields and map to store_raw_article."""
    mapping = {
        'source_url': ['source_url', 'url'],
        'title': ['title', 'title_translated', 'title_original'],
        'text_content': ['raw_text', 'content', 'cleaned_text'],
        'raw_html': ['raw_html'],
        'language': ['language', 'content_language'],
        'published_date': ['published_date'],
    }
    params = {}
    for k, aliases in mapping.items():
        for alias in aliases:
            if alias in kwargs:
                params[k] = kwargs[alias]
                break
    result = store_raw_article(**params)
    if isinstance(result, dict) and 'id' in result:
        return result['id']
    return result

def save_cluster(*args, **kwargs):
    """Compatibility: Accept legacy cluster fields and map to create_cluster."""
    mapping = {
        'label': ['theme', 'label', 'cluster_label'],
        'description': ['summary', 'description', 'cluster_description'],
        'center_embedding': ['center_embedding'],
        'member_count': ['member_count'],
        'representative_article_id': ['representative_article_id'],
    }
    params = {}
    for k, aliases in mapping.items():
        for alias in aliases:
            if alias in kwargs:
                params[k] = kwargs[alias]
                break
    result = create_cluster(**params)
    if isinstance(result, dict) and 'id' in result:
        return True
    return result 