#!/usr/bin/env python3
"""
Simple clustering task that uses fast_clustering service
"""

from celery import Celery
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('config.celery_config')

@celery_app.task(bind=True)
def run_clustering(self):
    """
    Run fast clustering on articles from Redis queue and store to Supabase
    """
    try:
        logger.info("🔍 Starting fast clustering process from Redis queue")
        
        # Import services
        from services.clusterer import cluster_articles_complete_with_summaries
        from db.redis_queue import get_from_clustering_queue, get_queue_size
        from db.supabase_client import store_article, save_cluster
        import uuid
        
        # Get articles from Redis queue
        queue_size = get_queue_size()
        logger.info(f"📊 Found {queue_size} articles in Redis queue")
        
        if queue_size == 0:
            logger.warning("⚠️ No articles found in Redis queue")
            return {"status": "no_data", "clusters_created": 0}
        
        # Process articles from queue
        articles_data = []
        embeddings = []
        articles_metadata = []
        
        processed_count = 0
        max_articles = min(queue_size, 50)  # Process up to 50 articles at a time
        
        while processed_count < max_articles:
            article_data = get_from_clustering_queue()
            if not article_data:
                break
                
            # Handle article data format
            if isinstance(article_data, dict):
                # New format with full article data
                article_id = article_data.get('article_id', f'unknown_{processed_count}')
                title = article_data.get('title', 'Unknown Title')
                cleaned_text = article_data.get('cleaned_text', '')
                embedding = article_data.get('embedding', [])
                region = article_data.get('region', 'Unknown')
                topic = article_data.get('topic', 'Unknown')
                source_url = article_data.get('source_url', '')
                
                # Store article to Supabase
                if embedding and len(embedding) > 0:
                    store_success = store_article(
                        article_id=article_id,
                        title=title,
                        raw_text=article_data.get('raw_text', ''),
                        cleaned_text=cleaned_text,
                        embedding=embedding,
                        region=region,
                        topic=topic,
                        source_url=source_url
                    )
                    
                    if store_success:
                        articles_data.append({
                            'id': article_id,
                            'title': title,
                            'content': cleaned_text,
                            'topic': topic,
                            'region': region,
                            'embedding': embedding
                        })
                        
                        embeddings.append(embedding)
                        articles_metadata.append({
                            'title': title,
                            'topic': topic,
                            'region': region,
                            'article_id': article_id,
                            'content': cleaned_text
                        })
                        
                        logger.info(f"✅ Processed and stored article {article_id}")
                    else:
                        logger.warning(f"⚠️ Failed to store article {article_id}")
                else:
                    logger.warning(f"⚠️ Article {article_id} has no embedding")
            else:
                # Old format - just article ID
                logger.warning(f"⚠️ Skipping old format article data: {article_data}")
            
            processed_count += 1
        
        if not embeddings:
            logger.warning("⚠️ No valid embeddings found in queue")
            return {"status": "no_embeddings", "clusters_created": 0}
        
        logger.info(f"📊 Processing {len(embeddings)} articles with embeddings")
        
        # Run fast clustering with summaries
        clustering_results = cluster_articles_complete_with_summaries(
            embeddings=embeddings,
            articles=articles_metadata,
            num_clusters=None,
            use_faiss=False,
            max_concurrent_summaries=3
        )
        
        clusters = clustering_results['clusters']
        summaries = clustering_results['summaries']
        
        # Save clusters to database
        total_clusters_saved = 0
        
        for cluster_id, cluster_indices in clusters.items():
            if len(cluster_indices) >= 3:
                cluster_db_ids = []
                for idx in cluster_indices:
                    if idx < len(articles_data):
                        article = articles_data[idx]
                        article_id = article.get('id')
                        if article_id:
                            cluster_db_ids.append(article_id)
                
                if len(cluster_db_ids) >= 3:
                    summary_info = summaries.get(cluster_id, {})
                    theme = summary_info.get('primary_topic', 'Unknown')
                    text_summary = summary_info.get('text_summary', 'No summary available')
                    
                    unique_cluster_id = f"cluster_{cluster_id}_{str(uuid.uuid4())[:8]}"
                    
                    success = save_cluster(
                        cluster_id=unique_cluster_id,
                        article_ids=cluster_db_ids,
                        status='complete',
                        theme=theme,
                        summary=text_summary
                    )
                    
                    if success:
                        total_clusters_saved += 1
                        logger.info(f"💾 Saved cluster {unique_cluster_id}: {theme} ({len(cluster_db_ids)} articles)")
        
        logger.info(f"✅ Fast clustering complete: {total_clusters_saved} clusters saved")
        
        return {
            "status": "success",
            "clusters_created": total_clusters_saved,
            "articles_processed": len(embeddings),
            "processing_time": clustering_results['processing_time']
        }
        
    except Exception as e:
        logger.error(f"❌ Error in fast clustering: {e}")
        raise self.retry(countdown=300, max_retries=3)

@celery_app.task(bind=True)
def maybe_trigger_clustering(self):
    """
    Check if clustering should be triggered based on queue size
    """
    try:
        logger.info("🔍 Checking if clustering should be triggered")
        
        # Import services
        from db.redis_queue import get_queue_size
        
        # Get queue size
        queue_size = get_queue_size()
        
        if queue_size >= 5:  # Trigger clustering if we have 5+ articles
            logger.info(f"📊 Found {queue_size} articles in queue, triggering clustering")
            # Trigger the main clustering task
            run_clustering.delay()
            return {"status": "triggered", "queue_size": queue_size}
        else:
            logger.info(f"📊 Only {queue_size} articles in queue, skipping clustering")
            return {"status": "skipped", "queue_size": queue_size}
            
    except Exception as e:
        logger.error(f"❌ Error in maybe_trigger_clustering: {e}")
        raise self.retry(countdown=60, max_retries=3) 