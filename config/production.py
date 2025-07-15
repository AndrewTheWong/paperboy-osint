#!/usr/bin/env python3
"""Production configuration for Paperboy pipeline"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "production")
DEBUG = ENVIRONMENT == "development"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Database
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Redis
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# CORS
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:8000",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000"
]

# Pipeline Configuration
MAX_ARTICLES_PER_BATCH = int(os.getenv("MAX_ARTICLES_PER_BATCH", "100"))
SCRAPING_TIMEOUT = int(os.getenv("SCRAPING_TIMEOUT", "30"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CLUSTERING_MIN_SAMPLES = int(os.getenv("CLUSTERING_MIN_SAMPLES", "5"))
CLUSTERING_EPSILON = float(os.getenv("CLUSTERING_EPSILON", "0.3"))

# Celery Configuration
CELERY_BROKER_URL = REDIS_URL
CELERY_RESULT_BACKEND = REDIS_URL
CELERY_TASK_SERIALIZER = "json"
CELERY_RESULT_SERIALIZER = "json"
CELERY_ACCEPT_CONTENT = ["json"]
CELERY_TIMEZONE = "UTC"
CELERY_ENABLE_UTC = True

# Worker Configuration
WORKER_PREFETCH_MULTIPLIER = 1
WORKER_MAX_TASKS_PER_CHILD = 1000
WORKER_POOL = "solo"  # For Windows compatibility

# Task Routing
TASK_ROUTES = {
    'workers.scraper.*': {'queue': 'scraper'},
    'workers.translator.*': {'queue': 'translate'},
    'workers.tagger.*': {'queue': 'tag'},
    'workers.embedder.*': {'queue': 'embed'},
    'workers.cluster.*': {'queue': 'cluster'},
    'workers.orchestrator.*': {'queue': 'orchestrate'},
}

# Beat Schedule
BEAT_SCHEDULE = {
    'run-continuous-scraper': {
        'task': 'workers.scraper.run_continuous_scraper',
        'schedule': 300.0,  # Every 5 minutes
    },
}

# Security
SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError("SECRET_KEY must be set in the environment for production.")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30 