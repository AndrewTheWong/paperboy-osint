#!/usr/bin/env python3
"""
Clusterer Worker v2 - Pulls processed articles with embeddings, clusters them, stores in clusters table
"""

import logging
import asyncio
import json
from datetime import datetime
from celery import shared_task, Celery
from services.embedder import apply_hdbscan_clustering
from db.supabase_client_v2 import get_articles_for_clustering, create_cluster, update_cluster_members
from db.redis_queue import RedisQueue

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('config.celery_config')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _run_async_with_proper_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(coro)
    finally:
        loop.close()

@shared_task(bind=True)
def run_clusterer_pipeline(self, batch_size=50, min_cluster_size=3):
    """
    Pull processed articles with embeddings, cluster them, and store cluster metadata.
    Args:
        batch_size: Number of articles to process in each batch
        min_cluster_size: Minimum cluster size for HDBSCAN
    Returns:
        dict: Clustering results and performance metrics
    """
    try:
        logger.info(f"🚀 Starting clusterer pipeline with batch size: {batch_size}")
        
        # Get processed articles with embeddings from Supabase
        articles = get_articles_for_clustering(limit=batch_size)
        if not articles:
            logger.info("📭 No processed articles with embeddings found")
            return {'status': 'success', 'clusters_created': 0, 'message': 'No articles to cluster'}
        
        logger.info(f"📋 Found {len(articles)} processed articles with embeddings")
        
        def cluster_and_store():
            start = datetime.now()
            clusters_created = 0
            
            # Extract embeddings and article IDs
            embeddings = []
            article_ids = []
            processed_ids = []  # IDs from articles_processed table
            for article in articles:
                embedding = article.get('embedding')
                if embedding and isinstance(embedding, str):
                    import json
                    embedding = json.loads(embedding)
                if embedding and len(embedding) == 384:
                    embeddings.append(embedding)
                    article_ids.append(article['article_id'])  # For reference to raw articles
                    processed_ids.append(article['id'])  # ID from articles_processed table
                else:
                    logger.warning(f"Skipping article {article.get('article_id')} due to invalid embedding dimension: {len(embedding) if embedding else 'None'}")
            
            if not embeddings:
                logger.warning("⚠️ No valid embeddings found in articles")
                return {
                    'status': 'success',
                    'clusters_created': 0,
                    'articles_processed': 0,
                    'duration_seconds': (datetime.now() - start).total_seconds()
                }
            
            # Debug: check embedding shapes
            logger.info(f"🔢 Clustering {len(embeddings)} articles with embeddings...")
            logger.info(f"📊 Embedding shapes: {[len(emb) for emb in embeddings[:3]]}")
            logger.info(f"📊 Sample embedding type: {type(embeddings[0]) if embeddings else 'None'}")
            
            # Perform clustering
            cluster_labels = apply_hdbscan_clustering(embeddings)
            
            # Group articles by cluster
            cluster_groups = {}
            for i, label in enumerate(cluster_labels):
                if label >= 0:  # Skip noise points (-1)
                    if label not in cluster_groups:
                        cluster_groups[label] = []
                    cluster_groups[label].append({
                        'article_id': article_ids[i],
                        'processed_id': processed_ids[i]
                    })
            
            logger.info(f"✅ Found {len(cluster_groups)} clusters")
            
            # Create cluster metadata and store in Supabase
            for cluster_id, members in cluster_groups.items():
                try:
                    # Get representative article for cluster info
                    representative_processed_id = members[0]['processed_id']
                    representative_article = next((a for a in articles if a['id'] == representative_processed_id), None)
                    
                    # Create cluster label and description
                    cluster_label = f"Cluster_{cluster_id}"
                    cluster_description = f"Cluster of {len(members)} articles"
                    
                    if representative_article:
                        # Get title from raw article if available
                        from db.supabase_client_v2 import get_raw_article
                        raw_article = get_raw_article(representative_article['article_id'])
                        if raw_article:
                            title = raw_article.get('title', '')
                            cluster_description = f"Cluster of {len(members)} articles. Representative: {title[:100]}..."
                    
                    # Calculate cluster center (mean of embeddings)
                    cluster_embeddings = [embeddings[i] for i, label in enumerate(cluster_labels) if label == cluster_id]
                    if cluster_embeddings:
                        import numpy as np
                        center_embedding = np.mean(cluster_embeddings, axis=0).tolist()
                    else:
                        center_embedding = None
                    
                    # Create cluster in Supabase using processed_id as representative
                    supabase_cluster_id = create_cluster(
                        label=cluster_label,
                        description=cluster_description,
                        center_embedding=center_embedding,
                        member_count=len(members),
                        representative_article_id=representative_processed_id
                    )
                    
                    if supabase_cluster_id:
                        logger.info(f"✅ Created cluster {supabase_cluster_id} with {len(members)} members")
                        clusters_created += 1
                        
                        # Update articles with cluster assignment
                        for member in members:
                            # Update the articles_processed table with cluster_id
                            from db.supabase_client_v2 import update_processed_article_cluster
                            update_processed_article_cluster(member['processed_id'], supabase_cluster_id)
                    else:
                        logger.error(f"❌ Failed to create cluster for group {cluster_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Error processing cluster {cluster_id}: {e}")
                    continue
            
            duration = (datetime.now() - start).total_seconds()
            rate = clusters_created / duration if duration > 0 else 0
            
            return {
                'status': 'success',
                'clusters_created': clusters_created,
                'articles_processed': len(embeddings),
                'duration_seconds': duration,
                'clusters_per_second': rate,
                'total_articles': len(articles)
            }
        
        result = cluster_and_store()
        logger.info(f"✅ Clusterer pipeline completed: {result['clusters_created']} clusters in {result['duration_seconds']:.2f}s ({result['clusters_per_second']:.2f}/sec)")
        return result
        
    except Exception as e:
        logger.error(f"❌ Clusterer pipeline failed: {e}")
        return {'status': 'error', 'error': str(e)} 