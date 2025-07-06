#!/usr/bin/env python3
"""
Clustering Worker for Paperboy Backend
Handles article clustering and similarity grouping
"""

import logging
from celery import shared_task
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
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

@shared_task(bind=True, max_retries=3)
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

@shared_task(bind=True, max_retries=3)
def cluster_articles_batch(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Cluster a batch of articles
    
    Args:
        articles: List of articles with embeddings
        
    Returns:
        dict: Clustering results
    """
    try:
        logger.info(f"🔗 Starting batch clustering of {len(articles)} articles")
        
        from services.clusterer import cluster_articles_complete
        
        # Extract embeddings for clustering
        embeddings = [article.get('embedding', []) for article in articles]
        
        # Filter out articles without embeddings
        valid_articles = []
        valid_embeddings = []
        for article, embedding in zip(articles, embeddings):
            if embedding and len(embedding) > 0:
                valid_articles.append(article)
                valid_embeddings.append(embedding)
        
        if not valid_embeddings:
            logger.warning("⚠️ No valid embeddings for clustering")
            return {"status": "no_embeddings", "clusters_created": 0}
        
        # Perform clustering
        clustering_result = cluster_articles_complete(valid_embeddings, valid_articles)
        clusters = clustering_result.get('clusters', {})
        cluster_summaries = clustering_result.get('cluster_summaries', {})
        
        # Assign cluster IDs to articles
        for cluster_id, article_indices in clusters.items():
            for idx in article_indices:
                if idx < len(valid_articles):
                    article = valid_articles[idx]
                    article['cluster_id'] = f"cluster_{cluster_id}"
                    
                    # Add cluster metadata
                    if cluster_id in cluster_summaries:
                        summary = cluster_summaries[cluster_id]
                        article['cluster_label'] = summary.get('primary_topic', 'Unknown')
                        article['cluster_description'] = f"Cluster with {summary.get('size', 0)} articles"
        
        logger.info(f"✅ Batch clustering completed: {len(clusters)} clusters created")
        
        return {
            "status": "success",
            "clusters_created": len(clusters),
            "articles_processed": len(valid_articles),
            "clusters": clusters,
            "summaries": cluster_summaries
        }
        
    except Exception as e:
        logger.error(f"❌ Batch clustering failed: {e}")
        raise self.retry(countdown=180, max_retries=3)

@shared_task(bind=True, max_retries=3)
def cluster_from_queue(self, queue_name: str = "clustering_queue") -> Dict[str, Any]:
    """
    Cluster articles from a Redis queue
    
    Args:
        queue_name: Name of the Redis queue to process
        
    Returns:
        dict: Clustering results
    """
    try:
        logger.info(f"🔗 Starting clustering from queue: {queue_name}")
        
        # Import Redis queue functions
        from db.redis_queue import get_from_queue, get_queue_size, add_to_queue
        
        # Get queue size
        queue_size = get_queue_size(queue_name)
        logger.info(f"📊 Found {queue_size} articles in {queue_name}")
        
        if queue_size == 0:
            logger.warning(f"⚠️ No articles found in {queue_name}")
            return {"status": "no_data", "clusters_created": 0}
        
        # Process articles from queue
        articles_to_cluster = []
        processed_count = 0
        max_articles = min(queue_size, 50)  # Process up to 50 articles at a time
        
        while processed_count < max_articles:
            article_data = get_from_queue(queue_name)
            if not article_data:
                break
            
            articles_to_cluster.append(article_data)
            processed_count += 1
        
        if not articles_to_cluster:
            logger.warning("⚠️ No articles retrieved from queue")
            return {"status": "no_articles", "clusters_created": 0}
        
        # Cluster the batch
        clustering_result = cluster_articles_batch(articles_to_cluster)
        
        # Store clustered articles back to queue for storage
        for article in articles_to_cluster:
            add_to_queue("storage_queue", article)
        
        logger.info(f"✅ Clustering from queue completed: {clustering_result['clusters_created']} clusters created")
        
        return clustering_result
        
    except Exception as e:
        logger.error(f"❌ Clustering from queue failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def store_clusters_to_database(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Store clusters to database
    
    Args:
        articles: List of articles with cluster assignments
        
    Returns:
        dict: Storage results
    """
    try:
        logger.info(f"💾 Storing clusters to database for {len(articles)} articles")
        
        from db.supabase_client import save_cluster
        import uuid
        
        # Group articles by cluster
        clusters = {}
        for article in articles:
            cluster_id = article.get('cluster_id')
            if cluster_id:
                if cluster_id not in clusters:
                    clusters[cluster_id] = []
                clusters[cluster_id].append(article)
        
        # Save clusters to database
        saved_count = 0
        for cluster_id, cluster_articles in clusters.items():
            if len(cluster_articles) >= 3:  # Only save clusters with 3+ articles
                article_ids = [article.get('article_id', 'unknown') for article in cluster_articles]
                theme = cluster_articles[0].get('cluster_label', 'Unknown')
                summary = cluster_articles[0].get('cluster_description', 'No description')
                
                unique_cluster_id = f"{cluster_id}_{str(uuid.uuid4())[:8]}"
                
                success = save_cluster(
                    cluster_id=unique_cluster_id,
                    article_ids=article_ids,
                    status='complete',
                    theme=theme,
                    summary=summary
                )
                
                if success:
                    saved_count += 1
                    logger.info(f"💾 Saved cluster {unique_cluster_id}: {theme} ({len(article_ids)} articles)")
        
        logger.info(f"✅ Cluster storage completed: {saved_count} clusters saved")
        
        return {
            "status": "success",
            "clusters_saved": saved_count,
            "total_clusters": len(clusters)
        }
        
    except Exception as e:
        logger.error(f"❌ Cluster storage failed: {e}")
        raise self.retry(countdown=120, max_retries=3) 