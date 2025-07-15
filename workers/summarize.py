#!/usr/bin/env python3
"""
Summarization task for article clusters
"""

from celery import Celery
import logging
from typing import List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery
celery_app = Celery('straitwatch')
celery_app.config_from_object('config.celery_config')

@celery_app.task(bind=True)
def summarize_cluster(self, cluster_id: str, article_ids: List[str]):
    """
    Summarize a cluster of articles
    
    Args:
        cluster_id: Unique cluster identifier
        article_ids: List of article IDs in the cluster
    """
    try:
        logger.info(f"📝 Summarizing cluster {cluster_id} with {len(article_ids)} articles")
        
        # Import services
        from db.supabase_client_v2 import get_articles_texts, update_cluster_summary
        from services.summarizer import generate_summary
        
        # Fetch article texts
        article_texts = get_articles_texts(article_ids)
        
        if not article_texts:
            logger.warning(f"⚠️ No texts found for cluster {cluster_id}")
            return {"status": "no_data", "cluster_id": cluster_id}
        
        # Combine texts for summarization
        combined_text = "\n\n".join(article_texts)
        
        # Generate summary
        summary = generate_summary(combined_text)
        
        logger.info(f"📄 Generated summary for cluster {cluster_id}: {len(summary)} characters")
        
        # Update cluster with summary and mark as complete
        update_cluster_summary(
            cluster_id=cluster_id,
            summary=summary,
            status='complete'
        )
        
        logger.info(f"✅ Updated cluster {cluster_id} with summary")
        
        return {
            "status": "success",
            "cluster_id": cluster_id,
            "summary_length": len(summary),
            "articles_processed": len(article_ids)
        }
        
    except Exception as e:
        logger.error(f"❌ Error summarizing cluster {cluster_id}: {e}")
        raise self.retry(countdown=120, max_retries=3)

@celery_app.task(bind=True)
def summarize_all_pending_clusters(self):
    """
    Summarize all pending clusters in the database
    """
    try:
        logger.info("📝 Starting batch summarization of pending clusters")
        
        # Import services
        from db.supabase_client_v2 import get_pending_clusters
        
        # Get all pending clusters
        pending_clusters = get_pending_clusters()
        
        if not pending_clusters:
            logger.info("📭 No pending clusters to summarize")
            return {"status": "idle", "clusters_summarized": 0}
        
        logger.info(f"📊 Found {len(pending_clusters)} pending clusters")
        
        # Process each cluster
        summarized_count = 0
        for cluster in pending_clusters:
            try:
                cluster_id = cluster['id']
                article_ids = cluster['article_ids']
                
                # Only summarize clusters with 3+ articles
                if len(article_ids) >= 3:
                    summarize_cluster.delay(cluster_id, article_ids)
                    summarized_count += 1
                    logger.info(f"📤 Queued cluster {cluster_id} for summarization")
                else:
                    # Mark small clusters as complete without summarization
                    from db.supabase_client_v2 import update_cluster_summary
                    update_cluster_summary(
                        cluster_id=cluster_id,
                        summary="Single article cluster - no summary needed",
                        status='complete'
                    )
                    logger.info(f"✅ Marked small cluster {cluster_id} as complete")
                    
            except Exception as e:
                logger.error(f"❌ Error processing cluster {cluster.get('id', 'unknown')}: {e}")
                continue
        
        logger.info(f"🎉 Batch summarization complete: {summarized_count} clusters queued")
        
        return {
            "status": "success",
            "clusters_summarized": summarized_count,
            "total_pending": len(pending_clusters)
        }
        
    except Exception as e:
        logger.error(f"❌ Error in batch summarization: {e}")
        raise self.retry(countdown=300, max_retries=3) 