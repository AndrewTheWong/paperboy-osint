#!/usr/bin/env python3
"""
Celery worker for StraitWatch backend
"""

import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from celery import Celery
import logging

# Configure enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('celery_worker.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Celery app
celery_app = Celery('straitwatch')
celery_app.config_from_object('config.celery_config')

# IMPORTANT: Import all tasks to register them with Celery
logger.info("Importing tasks for registration...")

try:
    from workers.preprocess import preprocess_and_enqueue
    logger.info("SUCCESS: Imported preprocess tasks")
except Exception as e:
    logger.error(f"FAILED: Import preprocess tasks: {e}")
    logger.error(f"Python path: {sys.path}")
    logger.error(f"Current directory: {os.getcwd()}")

try:
    from workers.cluster import run_clustering
    logger.info("SUCCESS: Imported cluster tasks")
except Exception as e:
    logger.error(f"FAILED: Import cluster tasks: {e}")

try:
    from workers.summarize import summarize_cluster, summarize_all_pending_clusters
    logger.info("SUCCESS: Imported summarize tasks")
except Exception as e:
    logger.error(f"FAILED: Import summarize tasks: {e}")

try:
    from workers.pipeline import (
        preprocess_article, tag_article_ner, embed_and_cluster_article, 
        store_to_supabase, run_article_pipeline, run_batch_pipeline,
        store_batch_to_supabase, process_article_batch
    )
    logger.info("SUCCESS: Imported new pipeline tasks")
except Exception as e:
    logger.error(f"FAILED: Import pipeline tasks: {e}")

try:
    from workers.scraper import run_async_scraper, run_continuous_scraper
    logger.info("SUCCESS: Imported async scraper tasks")
except Exception as e:
    logger.error(f"FAILED: Import async scraper tasks: {e}")

try:
    from app.tasks.orchestrator import run_pipeline_orchestrator
    logger.info("SUCCESS: Imported orchestrator tasks")
except Exception as e:
    logger.error(f"FAILED: Import orchestrator tasks: {e}")

# Auto-discover tasks as backup
celery_app.autodiscover_tasks(['workers', 'app.tasks'])

logger.info(f"Registered tasks: {list(celery_app.tasks.keys())}")

if __name__ == '__main__':
    logger.info("Starting StraitWatch Celery worker")
    celery_app.start() 