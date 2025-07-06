#!/usr/bin/env python3
"""
Pipeline Orchestrator for Paperboy Backend
Coordinates all pipeline steps: Scraping → Translation → Tagging → Embedding → Clustering
"""

import logging
from celery import shared_task
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def run_full_pipeline(self, sources=None, max_articles_per_source=10) -> Dict[str, Any]:
    """
    Run the complete pipeline: Scraping → Translation → Tagging → Embedding → Clustering
    
    Args:
        sources: List of source dictionaries
        max_articles_per_source: Maximum articles per source
        
    Returns:
        dict: Complete pipeline results
    """
    try:
        logger.info("🚀 Starting full pipeline orchestration")
        start_time = time.time()
        
        # Step 1: Scraping
        logger.info("📰 Step 1: Starting scraping...")
        from workers.scraper import run_async_scraper
        scraping_result = run_async_scraper.delay(sources, max_articles_per_source)
        scraping_data = scraping_result.get()
        
        if not scraping_data or scraping_data.get('total_scraped', 0) == 0:
            logger.warning("⚠️ No articles scraped, stopping pipeline")
            return {"status": "no_articles_scraped", "pipeline_time": time.time() - start_time}
        
        logger.info(f"✅ Scraping completed: {scraping_data.get('total_scraped', 0)} articles")
        
        # Step 2: Translation
        logger.info("🌐 Step 2: Starting translation...")
        from workers.translator import translate_articles_batch
        articles = scraping_data.get('articles', [])
        translation_result = translate_articles_batch.delay(articles)
        translated_articles = translation_result.get()
        
        logger.info(f"✅ Translation completed: {len(translated_articles)} articles")
        
        # Step 3: Tagging
        logger.info("🏷️ Step 3: Starting tagging...")
        from workers.tagger import tag_articles_batch
        tagging_result = tag_articles_batch.delay(translated_articles)
        tagged_articles = tagging_result.get()
        
        total_tags = sum(len(article.get('tags', [])) for article in tagged_articles)
        total_entities = sum(len(article.get('entities', [])) for article in tagged_articles)
        logger.info(f"✅ Tagging completed: {total_tags} tags, {total_entities} entities")
        
        # Step 4: Embedding
        logger.info("🔢 Step 4: Starting embedding...")
        from workers.embedder import embed_articles_batch
        embedding_result = embed_articles_batch.delay(tagged_articles)
        embedded_articles = embedding_result.get()
        
        successful_embeddings = sum(1 for article in embedded_articles if article.get('embedding'))
        logger.info(f"✅ Embedding completed: {successful_embeddings}/{len(embedded_articles)} articles")
        
        # Step 5: Clustering
        logger.info("🔗 Step 5: Starting clustering...")
        from workers.cluster import cluster_articles_batch
        clustering_result = cluster_articles_batch.delay(embedded_articles)
        clustering_data = clustering_result.get()
        
        clusters_created = clustering_data.get('clusters_created', 0)
        logger.info(f"✅ Clustering completed: {clusters_created} clusters created")
        
        # Step 6: Storage
        logger.info("💾 Step 6: Starting storage...")
        from workers.cluster import store_clusters_to_database
        storage_result = store_clusters_to_database.delay(embedded_articles)
        storage_data = storage_result.get()
        
        clusters_saved = storage_data.get('clusters_saved', 0)
        logger.info(f"✅ Storage completed: {clusters_saved} clusters saved")
        
        # Calculate pipeline metrics
        pipeline_time = time.time() - start_time
        articles_per_second = len(embedded_articles) / pipeline_time if pipeline_time > 0 else 0
        
        logger.info(f"🎉 Full pipeline completed in {pipeline_time:.2f}s")
        logger.info(f"📊 Performance: {articles_per_second:.2f} articles/second")
        
        return {
            "status": "success",
            "pipeline_time": pipeline_time,
            "articles_scraped": scraping_data.get('total_scraped', 0),
            "articles_translated": len(translated_articles),
            "articles_tagged": len(tagged_articles),
            "articles_embedded": successful_embeddings,
            "clusters_created": clusters_created,
            "clusters_saved": clusters_saved,
            "total_tags": total_tags,
            "total_entities": total_entities,
            "articles_per_second": articles_per_second
        }
        
    except Exception as e:
        logger.error(f"❌ Full pipeline failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def run_pipeline_from_queue(self) -> Dict[str, Any]:
    """
    Run pipeline processing articles from Redis queues
    
    Returns:
        dict: Pipeline results
    """
    try:
        logger.info("🔄 Starting pipeline from queues")
        start_time = time.time()
        
        # Process each step from queues
        from workers.translator import translate_from_queue
        from workers.tagger import tag_from_queue
        from workers.embedder import embed_from_queue
        from workers.cluster import cluster_from_queue
        
        # Step 1: Translation from queue
        translation_result = translate_from_queue.delay("scraping_queue")
        translation_data = translation_result.get()
        
        # Step 2: Tagging from queue
        tagging_result = tag_from_queue.delay("translation_queue")
        tagging_data = tagging_result.get()
        
        # Step 3: Embedding from queue
        embedding_result = embed_from_queue.delay("tagging_queue")
        embedding_data = embedding_result.get()
        
        # Step 4: Clustering from queue
        clustering_result = cluster_from_queue.delay("embedding_queue")
        clustering_data = clustering_result.get()
        
        pipeline_time = time.time() - start_time
        
        logger.info(f"✅ Queue pipeline completed in {pipeline_time:.2f}s")
        
        return {
            "status": "success",
            "pipeline_time": pipeline_time,
            "translation": translation_data,
            "tagging": tagging_data,
            "embedding": embedding_data,
            "clustering": clustering_data
        }
        
    except Exception as e:
        logger.error(f"❌ Queue pipeline failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def run_continuous_pipeline(self, interval_minutes=30) -> Dict[str, Any]:
    """
    Run continuous pipeline with periodic intervals
    
    Args:
        interval_minutes: Minutes between pipeline runs
        
    Returns:
        dict: Continuous pipeline results
    """
    try:
        logger.info(f"🔄 Starting continuous pipeline (interval: {interval_minutes} minutes)")
        
        # Run one pipeline cycle
        pipeline_result = run_full_pipeline.delay()
        pipeline_data = pipeline_result.get()
        
        logger.info(f"✅ Continuous pipeline cycle completed")
        
        return {
            "status": "continuous_success",
            "interval_minutes": interval_minutes,
            "pipeline_results": pipeline_data
        }
        
    except Exception as e:
        logger.error(f"❌ Continuous pipeline failed: {e}")
        raise self.retry(countdown=interval_minutes * 60, max_retries=3)

@shared_task(bind=True, max_retries=3)
def run_pipeline_step(self, step_name: str, articles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Run a single pipeline step
    
    Args:
        step_name: Name of the step to run
        articles: Articles to process (optional, will use queue if not provided)
        
    Returns:
        dict: Step results
    """
    try:
        logger.info(f"🔄 Running pipeline step: {step_name}")
        
        if step_name == "translation":
            from workers.translator import translate_articles_batch
            if articles:
                result = translate_articles_batch.delay(articles)
            else:
                from workers.translator import translate_from_queue
                result = translate_from_queue.delay()
                
        elif step_name == "tagging":
            from workers.tagger import tag_articles_batch
            if articles:
                result = tag_articles_batch.delay(articles)
            else:
                from workers.tagger import tag_from_queue
                result = tag_from_queue.delay()
                
        elif step_name == "embedding":
            from workers.embedder import embed_articles_batch
            if articles:
                result = embed_articles_batch.delay(articles)
            else:
                from workers.embedder import embed_from_queue
                result = embed_from_queue.delay()
                
        elif step_name == "clustering":
            from workers.cluster import cluster_articles_batch
            if articles:
                result = cluster_articles_batch.delay(articles)
            else:
                from workers.cluster import cluster_from_queue
                result = cluster_from_queue.delay()
                
        else:
            raise ValueError(f"Unknown pipeline step: {step_name}")
        
        step_data = result.get()
        logger.info(f"✅ Pipeline step {step_name} completed")
        
        return {
            "status": "success",
            "step": step_name,
            "results": step_data
        }
        
    except Exception as e:
        logger.error(f"❌ Pipeline step {step_name} failed: {e}")
        raise self.retry(countdown=120, max_retries=3)

@shared_task(bind=True, max_retries=3)
def monitor_pipeline_health(self) -> Dict[str, Any]:
    """
    Monitor the health of all pipeline components
    
    Returns:
        dict: Health status of all components
    """
    try:
        logger.info("🏥 Monitoring pipeline health")
        
        health_status = {
            "timestamp": time.time(),
            "components": {}
        }
        
        # Check Redis queues
        try:
            from db.redis_queue import get_queue_size
            queue_sizes = {}
            for queue_name in ["scraping_queue", "translation_queue", "tagging_queue", 
                             "embedding_queue", "clustering_queue", "storage_queue"]:
                queue_sizes[queue_name] = get_queue_size(queue_name)
            
            health_status["components"]["redis_queues"] = {
                "status": "healthy",
                "queue_sizes": queue_sizes
            }
        except Exception as e:
            health_status["components"]["redis_queues"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Supabase connection
        try:
            from db.supabase_client import get_articles_count
            article_count = get_articles_count()
            health_status["components"]["supabase"] = {
                "status": "healthy",
                "article_count": article_count
            }
        except Exception as e:
            health_status["components"]["supabase"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Check Faiss index
        try:
            from services.faiss_index import get_faiss_service
            faiss_service = get_faiss_service()
            index_size = faiss_service.get_index_size()
            health_status["components"]["faiss"] = {
                "status": "healthy",
                "index_size": index_size
            }
        except Exception as e:
            health_status["components"]["faiss"] = {
                "status": "unhealthy",
                "error": str(e)
            }
        
        # Overall health assessment
        healthy_components = sum(1 for comp in health_status["components"].values() 
                               if comp["status"] == "healthy")
        total_components = len(health_status["components"])
        
        health_status["overall"] = {
            "status": "healthy" if healthy_components == total_components else "degraded",
            "healthy_components": healthy_components,
            "total_components": total_components
        }
        
        logger.info(f"✅ Health check completed: {healthy_components}/{total_components} components healthy")
        
        return health_status
        
    except Exception as e:
        logger.error(f"❌ Health monitoring failed: {e}")
        raise self.retry(countdown=300, max_retries=3) 