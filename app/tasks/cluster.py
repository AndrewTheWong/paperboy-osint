#!/usr/bin/env python3
"""
Clustering task for article analysis pipeline
"""

from celery import Celery
import logging
import numpy as np
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('app.celery_config')

@celery_app.task(bind=True)
def run_clustering(self):
    """
    Run clustering on articles in the queue
    
    NEW FLOW: Reads article data from Redis queue, applies clustering, 
    then stores both articles and clusters to database
    """
    try:
        logger.info("🔍 Starting clustering process")
        
        # Import services
        from app.services.redis_queue import get_from_clustering_queue, get_queue_size
        from app.services.supabase import store_article, save_cluster
        from app.services.embedding import apply_hdbscan_clustering
        import uuid
        
        # Get queue size
        queue_size = get_queue_size()
        if queue_size == 0:
            logger.info("📭 No articles in clustering queue")
            return {"status": "idle", "clusters_created": 0}
        
        logger.info(f"📊 Processing {queue_size} article data from clustering queue")
        
        # Get article data from queue (process in batches)
        batch_size = 50  # Smaller batches since we're handling full data
        total_clusters = 0
        total_articles_stored = 0
        
        while True:
            # Get batch of article data
            articles_data = []
            for _ in range(min(batch_size, queue_size)):
                article_data = get_from_clustering_queue()
                if article_data:
                    # Handle both new format (dict) and old format (string ID)
                    if isinstance(article_data, dict):
                        articles_data.append(article_data)
                    else:
                        logger.warning(f"⚠️ Skipping old format article ID: {article_data}")
            
            if not articles_data:
                break
            
            logger.info(f"🔄 Processing batch of {len(articles_data)} articles")
            
            # Extract embeddings for clustering
            embeddings = []
            for article in articles_data:
                embedding = article.get('embedding', [])
                if isinstance(embedding, list) and len(embedding) > 0:
                    embeddings.append(embedding)
                else:
                    logger.warning(f"⚠️ Invalid embedding for article {article.get('article_id')}")
            
            if not embeddings:
                logger.warning("⚠️ No valid embeddings in batch")
                continue
            
            # Apply HDBSCAN clustering
            cluster_labels = apply_hdbscan_clustering(embeddings)
            
            # Group articles by cluster and store to database
            clusters = {}
            article_db_ids = {}  # Map article_id to database ID
            
            for i, (article_data, label) in enumerate(zip(articles_data, cluster_labels)):
                # Store article to database first
                article_id = article_data.get('article_id')
                db_id = store_article(
                    article_id=article_id,
                    title=article_data.get('title', ''),
                    raw_text=article_data.get('raw_text', ''),
                    cleaned_text=article_data.get('cleaned_text', ''),
                    embedding=article_data.get('embedding', []),
                    region=article_data.get('region'),
                    topic=article_data.get('topic'),
                    source_url=article_data.get('source_url', '')
                )
                
                if db_id:
                    article_db_ids[article_id] = db_id
                    total_articles_stored += 1
                    
                    # Group by cluster
                    if label == -1:  # Noise points become single-article clusters
                        cluster_key = f"noise_{article_id}"
                    else:
                        cluster_key = f"cluster_{label}_{str(uuid.uuid4())[:8]}"
                    
                    if cluster_key not in clusters:
                        clusters[cluster_key] = []
                    clusters[cluster_key].append(db_id)
                else:
                    logger.error(f"❌ Failed to store article {article_id}")
            
            # Save clusters to database using database IDs
            for cluster_id, cluster_db_ids in clusters.items():
                if len(cluster_db_ids) >= 1:  # Save all clusters
                    success = save_cluster(
                        cluster_id=cluster_id,
                        article_ids=cluster_db_ids,
                        status='pending'
                    )
                    if success:
                        total_clusters += 1
                        
                        # Trigger summarization if cluster has 3+ articles
                        if len(cluster_db_ids) >= 3:
                            from app.tasks.summarize import summarize_cluster
                            summarize_cluster.delay(cluster_id, cluster_db_ids)
                            logger.info(f"📝 Triggered summarization for cluster {cluster_id}")
            
            logger.info(f"✅ Processed batch: {len(clusters)} clusters, {len(article_db_ids)} articles stored")
        
        logger.info(f"🎉 Clustering complete: {total_clusters} clusters created, {total_articles_stored} articles stored")
        
        return {
            "status": "success",
            "clusters_created": total_clusters,
            "articles_stored": total_articles_stored,
            "articles_processed": queue_size
        }
        
    except Exception as e:
        logger.error(f"❌ Error in clustering: {e}")
        raise self.retry(countdown=300, max_retries=3)

@celery_app.task(bind=True)
def cluster_single_batch(self, article_ids: List[str]):
    """
    Cluster a specific batch of articles
    
    Args:
        article_ids: List of article IDs to cluster
    """
    try:
        logger.info(f"🔍 Clustering batch of {len(article_ids)} articles")
        
        # Import services
        from app.services.supabase import get_embeddings_batch, save_cluster
        from app.services.embedding import apply_hdbscan_clustering
        
        # Fetch embeddings
        embeddings_data = get_embeddings_batch(article_ids)
        
        if not embeddings_data:
            logger.warning("⚠️ No embeddings found for articles")
            return {"status": "no_data", "clusters_created": 0}
        
        # Extract embeddings
        embeddings = [item['embedding'] for item in embeddings_data]
        batch_article_ids = [item['id'] for item in embeddings_data]
        
        # Apply clustering
        cluster_labels = apply_hdbscan_clustering(embeddings)
        
        # Group articles by cluster
        clusters = {}
        for i, label in enumerate(cluster_labels):
            if label == -1:
                clusters[f"noise_{batch_article_ids[i]}"] = [batch_article_ids[i]]
            else:
                cluster_key = f"cluster_{label}"
                if cluster_key not in clusters:
                    clusters[cluster_key] = []
                clusters[cluster_key].append(batch_article_ids[i])
        
        # Save clusters
        total_clusters = 0
        for cluster_id, cluster_article_ids in clusters.items():
            save_cluster(
                cluster_id=cluster_id,
                article_ids=cluster_article_ids,
                status='pending'
            )
            total_clusters += 1
            
            # Trigger summarization for large clusters
            if len(cluster_article_ids) >= 3:
                from app.tasks.summarize import summarize_cluster
                summarize_cluster.delay(cluster_id, cluster_article_ids)
        
        return {
            "status": "success",
            "clusters_created": total_clusters,
            "articles_processed": len(article_ids)
        }
        
    except Exception as e:
        logger.error(f"❌ Error clustering batch: {e}")
        raise self.retry(countdown=60, max_retries=3) 