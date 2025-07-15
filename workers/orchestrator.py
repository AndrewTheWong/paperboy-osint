#!/usr/bin/env python3
"""
Pipeline Orchestrator for Paperboy Backend
Coordinates all pipeline steps: Scraping → Translation → Tagging → Embedding → Clustering
"""

import logging
from celery import shared_task, chain
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def run_full_pipeline(self, sources=None, max_articles_per_source=10):
    """
    Run the complete pipeline: Scraping → Translation → Tagging → Embedding → Clustering
    """
    try:
        logger.info("🚀 Starting full pipeline orchestration (async chain)")
        from workers.scraper import run_async_scraper
        from workers.translator import translate_articles_batch
        from workers.tagger import tag_articles_batch
        from workers.embedder import embed_articles_batch
        from workers.cluster import cluster_articles_batch, store_clusters_to_database

        # Build the async chain
        pipeline = chain(
            run_async_scraper.s(sources, max_articles_per_source),
            translate_articles_batch.s(),
            tag_articles_batch.s(),
            embed_articles_batch.s(),
            cluster_articles_batch.s(),
            store_clusters_to_database.s()
        )
        result = pipeline.apply_async()
        logger.info(f"✅ Pipeline chain triggered: {result.id}")
        return {"status": "pipeline_triggered", "task_id": result.id}
    except Exception as e:
        logger.error(f"❌ Full pipeline failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def run_pipeline_from_queue(self):
    try:
        logger.info("🔄 Starting pipeline from queues (async chain)")
        from workers.translator import translate_from_queue
        from workers.tagger import tag_from_queue
        from workers.embedder import embed_from_queue
        from workers.cluster import cluster_from_queue
        pipeline = chain(
            translate_from_queue.s("scraping_queue"),
            tag_from_queue.s("translation_queue"),
            embed_from_queue.s("tagging_queue"),
            cluster_from_queue.s("embedding_queue")
        )
        result = pipeline.apply_async()
        logger.info(f"✅ Queue pipeline chain triggered: {result.id}")
        return {"status": "queue_pipeline_triggered", "task_id": result.id}
    except Exception as e:
        logger.error(f"❌ Queue pipeline failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def run_continuous_pipeline(self, interval_minutes=30):
    try:
        logger.info(f"🔄 Starting continuous pipeline (interval: {interval_minutes} minutes)")
        result = run_full_pipeline.apply_async()
        logger.info(f"✅ Continuous pipeline triggered: {result.id}")
        return {"status": "continuous_pipeline_triggered", "interval_minutes": interval_minutes, "task_id": result.id}
    except Exception as e:
        logger.error(f"❌ Continuous pipeline failed: {e}")
        raise self.retry(countdown=interval_minutes * 60, max_retries=3)

@shared_task(bind=True, max_retries=3)
def run_pipeline_step(self, step_name: str, articles: List[Dict[str, Any]] = None):
    try:
        logger.info(f"🔄 Running pipeline step: {step_name}")
        if step_name == "translation":
            from workers.translator import translate_articles_batch
            result = translate_articles_batch.apply_async(args=[articles])
        elif step_name == "tagging":
            from workers.tagger import tag_articles_batch
            result = tag_articles_batch.apply_async(args=[articles])
        elif step_name == "embedding":
            from workers.embedder import embed_articles_batch
            result = embed_articles_batch.apply_async(args=[articles])
        elif step_name == "clustering":
            from workers.cluster import cluster_articles_batch
            result = cluster_articles_batch.apply_async(args=[articles])
        else:
            raise ValueError(f"Unknown pipeline step: {step_name}")
        logger.info(f"✅ Pipeline step {step_name} triggered: {result.id}")
        return {"status": "step_triggered", "step": step_name, "task_id": result.id}
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
            from db.supabase_client_v2 import get_articles_count
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