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
# Assign tasks to specific queues
# (Uncomment and update for dedicated workers)
task_routes = {
    'workers.scraper.run_async_scraper': {'queue': 'scrape'},
    'workers.scraper.run_continuous_scraper': {'queue': 'scrape'},
    'workers.preprocess.preprocess_and_enqueue': {'queue': 'preprocess'},
    'workers.pipeline.preprocess_article': {'queue': 'preprocess'},
    'workers.pipeline.tag_article_ner': {'queue': 'preprocess'},
    'workers.cluster.run_clustering': {'queue': 'clustering'},
    # Add more as needed
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