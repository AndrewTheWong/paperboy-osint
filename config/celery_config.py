#!/usr/bin/env python3
"""
Celery configuration for StraitWatch backend
"""

# Broker settings
broker_url = 'redis://localhost:6379/0'
result_backend = 'redis://localhost:6379/0'

# Task settings
task_serializer = 'json'
accept_content = ['json']
result_serializer = 'json'
timezone = 'UTC'
enable_utc = True

# Task routing
# Assign tasks to specific queues for dedicated workers
task_routes = {
    # Scraping tasks
    'workers.scraper.run_async_scraper': {'queue': 'scrape'},
    'workers.scraper.run_continuous_scraper': {'queue': 'scrape'},
    'workers.scraper.run_stress_test_scraper': {'queue': 'scrape'},
    
    # Translation tasks
    'workers.translator.translate_single_article': {'queue': 'translate'},
    'workers.translator.translate_articles_batch': {'queue': 'translate'},
    'workers.translator.translate_from_queue': {'queue': 'translate'},
    'workers.translator.detect_languages_batch': {'queue': 'translate'},
    
    # Tagging tasks
    'workers.tagger.tag_single_article': {'queue': 'tag'},
    'workers.tagger.tag_articles_batch': {'queue': 'tag'},
    'workers.tagger.tag_from_queue': {'queue': 'tag'},
    'workers.tagger.extract_entities_only': {'queue': 'tag'},
    'workers.tagger.tag_with_custom_categories': {'queue': 'tag'},
    
    # Embedding tasks
    'workers.embedder.embed_single_article': {'queue': 'embed'},
    'workers.embedder.embed_articles_batch': {'queue': 'embed'},
    'workers.embedder.embed_from_queue': {'queue': 'embed'},
    'workers.embedder.store_embeddings_to_faiss': {'queue': 'embed'},
    'workers.embedder.embed_and_store_batch': {'queue': 'embed'},
    'workers.embedder.similarity_search': {'queue': 'embed'},
    
    # Clustering tasks
    'workers.cluster.run_clustering': {'queue': 'cluster'},
    'workers.cluster.cluster_articles_batch': {'queue': 'cluster'},
    'workers.cluster.cluster_from_queue': {'queue': 'cluster'},
    'workers.cluster.store_clusters_to_database': {'queue': 'cluster'},
    'workers.cluster.maybe_trigger_clustering': {'queue': 'cluster'},
    
    # Orchestration tasks
    'workers.orchestrator.run_full_pipeline': {'queue': 'orchestrate'},
    'workers.orchestrator.run_pipeline_from_queue': {'queue': 'orchestrate'},
    'workers.orchestrator.run_continuous_pipeline': {'queue': 'orchestrate'},
    'workers.orchestrator.run_pipeline_step': {'queue': 'orchestrate'},
    'workers.orchestrator.monitor_pipeline_health': {'queue': 'orchestrate'},
    'run_pipeline_orchestrator': {'queue': 'orchestrate'},
    
    # Legacy tasks
    'workers.preprocess.preprocess_and_enqueue': {'queue': 'preprocess'},
    'workers.pipeline.preprocess_article': {'queue': 'preprocess'},
    'workers.pipeline.tag_article_ner': {'queue': 'preprocess'},
}

# Queue settings
task_default_queue = 'default'
task_default_exchange = 'default'
task_default_routing_key = 'default'

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000
worker_disable_rate_limits = False

# Windows-specific settings
worker_pool = 'solo'  # Use solo pool for Windows to avoid multiprocessing issues

# Task execution settings
task_always_eager = False  # Set to True for testing
task_eager_propagates = True
task_ignore_result = False

# Result settings
result_expires = 3600  # 1 hour
result_persistent = True

# Beat settings (for periodic tasks)
beat_schedule = {
    'run-continuous-scraper': {
        'task': 'workers.scraper.run_continuous_scraper',
        'schedule': 300.0,  # Run every 5 minutes
    },
    # Removed clustering periodic task - using fast_clustering service directly
}

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s' 