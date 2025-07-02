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

# Task routing - TEMPORARILY DISABLED FOR DEBUGGING
# task_routes = {
#     'app.tasks.preprocess.preprocess_and_enqueue': {'queue': 'preprocessing'},
#     'app.tasks.cluster.run_clustering': {'queue': 'clustering'},
#     'app.tasks.cluster.cluster_single_batch': {'queue': 'clustering'},
#     'app.tasks.summarize.summarize_cluster': {'queue': 'summarization'},
#     'app.tasks.summarize.summarize_all_pending_clusters': {'queue': 'summarization'},
# }

# Queue settings
task_default_queue = 'default'
task_default_exchange = 'default'
task_default_routing_key = 'default'

# Worker settings
worker_prefetch_multiplier = 1
worker_max_tasks_per_child = 1000
worker_disable_rate_limits = False

# Task execution settings
task_always_eager = False  # Set to True for testing
task_eager_propagates = True
task_ignore_result = False

# Result settings
result_expires = 3600  # 1 hour
result_persistent = True

# Beat settings (for periodic tasks)
beat_schedule = {
    'run-clustering-every-5-minutes': {
        'task': 'app.tasks.cluster.run_clustering',
        'schedule': 300.0,  # 5 minutes
    },
    'summarize-pending-clusters-every-10-minutes': {
        'task': 'app.tasks.summarize.summarize_all_pending_clusters',
        'schedule': 600.0,  # 10 minutes
    },
}

# Logging
worker_log_format = '[%(asctime)s: %(levelname)s/%(processName)s] %(message)s'
worker_task_log_format = '[%(asctime)s: %(levelname)s/%(processName)s][%(task_name)s(%(task_id)s)] %(message)s' 