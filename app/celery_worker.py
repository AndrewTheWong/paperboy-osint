#!/usr/bin/env python3
"""
Celery worker for StraitWatch backend
"""

from celery import Celery
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('straitwatch')
celery_app.config_from_object('app.celery_config')

# IMPORTANT: Import all tasks to register them with Celery
logger.info("🔄 Importing tasks for registration...")

try:
    from app.tasks.preprocess import preprocess_and_enqueue
    logger.info("✅ Imported preprocess tasks")
except Exception as e:
    logger.error(f"❌ Failed to import preprocess tasks: {e}")

try:
    from app.tasks.cluster import run_clustering, cluster_single_batch
    logger.info("✅ Imported cluster tasks")
except Exception as e:
    logger.error(f"❌ Failed to import cluster tasks: {e}")

try:
    from app.tasks.summarize import summarize_cluster, summarize_all_pending_clusters
    logger.info("✅ Imported summarize tasks")
except Exception as e:
    logger.error(f"❌ Failed to import summarize tasks: {e}")

try:
    from app.tasks.pipeline_tasks import (
        preprocess_article, tag_article_ner, embed_and_cluster_article, 
        store_to_supabase, run_article_pipeline, run_batch_pipeline,
        store_batch_to_supabase, process_article_batch
    )
    logger.info("✅ Imported new pipeline tasks")
except Exception as e:
    logger.error(f"❌ Failed to import pipeline tasks: {e}")

# Auto-discover tasks as backup
celery_app.autodiscover_tasks(['app.tasks'])

logger.info(f"📋 Registered tasks: {list(celery_app.tasks.keys())}")

if __name__ == '__main__':
    logger.info("🚀 Starting StraitWatch Celery worker")
    celery_app.start() 