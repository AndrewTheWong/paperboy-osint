#!/usr/bin/env python3
"""
StraitWatch Backend ML Pipeline Package

This package contains all the backend machine learning components for StraitWatch Phase 1:

Core Components:
- Ingest: Unified news scraping and ingestion pipeline
- inference_pipeline: Escalation prediction from article text
- tagging_pipeline: Keyword and ML-based article tagging
- cluster_articles: HDBSCAN-based article clustering
- train_forecasting_model: XGBoost time series forecasting
- evaluate_models: Comprehensive model evaluation
- run_backend_pipeline: Main orchestrator script

Usage:
    from pipelines import BackendPipelineOrchestrator
    from pipelines.inference_pipeline import predict_escalation
    from pipelines.tagging_pipeline import ArticleTaggingPipeline
    from pipelines.Ingest import NewsScraper, run_news_ingestion_pipeline
"""

__version__ = "1.0.0"
__author__ = "StraitWatch Team"

# Import main classes for convenience
try:
    from .run_backend_pipeline import BackendPipelineOrchestrator
    from .inference_pipeline import predict_escalation
    from .tagging_pipeline import ArticleTaggingPipeline, TagConfig
    from .cluster_articles import cluster_articles
    from .evaluate_models import ModelEvaluationPipeline
    
    # Import new Ingest module
    from .Ingest import NewsScraper, ScrapingConfig, run_news_ingestion_pipeline
    
    __all__ = [
        'BackendPipelineOrchestrator',
        'predict_escalation',
        'ArticleTaggingPipeline',
        'TagConfig',
        'cluster_articles',
        'ModelEvaluationPipeline',
        'NewsScraper',
        'ScrapingConfig',
        'run_news_ingestion_pipeline'
    ]
    
except ImportError as e:
    # Graceful degradation if dependencies are missing
    print(f"Warning: Some pipeline components could not be imported: {e}")
    __all__ = []

# Import key pipeline components for easier access
try:
    from .embedding_pipeline import embed_articles, save_embedded_articles
except ImportError:
    pass

try:
    from .translation_pipeline import translate_articles
except ImportError:
    pass

try:
    from .supabase_storage import upload_articles_to_supabase
except ImportError:
    pass 