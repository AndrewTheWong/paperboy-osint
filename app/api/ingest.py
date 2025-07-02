#!/usr/bin/env python3
"""
Ingest API Router for StraitWatch Backend
Handles article ingestion endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/ingest", tags=["ingestion"])

class ArticleIngest(BaseModel):
    """Article ingestion model"""
    id: Optional[str] = None
    title: str
    body: str
    region: Optional[str] = None
    topic: Optional[str] = None
    source_url: str

class IngestResponse(BaseModel):
    """Response model for ingestion"""
    id: str
    status: str
    message: str

@router.post("/", response_model=IngestResponse)
async def ingest_article(article: ArticleIngest):
    """
    Ingest a single article
    
    Args:
        article: Article data with id, title, body, region, topic, source_url
        
    Returns:
        IngestResponse: Confirmation of ingestion with article ID
    """
    try:
        # Generate ID if not provided
        if not article.id:
            article.id = str(uuid.uuid4())
        
        logger.info(f"📥 Ingesting article {article.id}: {article.title}")
        
        # Import Celery task
        from app.tasks.preprocess import preprocess_and_enqueue
        
        # Call the preprocessing task
        task = preprocess_and_enqueue.delay(
            article_id=article.id,
            title=article.title,
            body=article.body,
            region=article.region,
            topic=article.topic,
            source_url=article.source_url
        )
        
        logger.info(f"✅ Article {article.id} queued for preprocessing (task: {task.id})")
        
        return IngestResponse(
            id=article.id,
            status="queued",
            message=f"Article queued for preprocessing. Task ID: {task.id}"
        )
        
    except Exception as e:
        logger.error(f"❌ Error ingesting article: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest article: {str(e)}"
        )

@router.get("/status")
async def get_ingest_status():
    """
    Get ingestion pipeline status
    
    Returns:
        dict: Status information about the ingestion pipeline
    """
    try:
        from app.services.supabase import get_articles_count, get_unprocessed_count
        
        total_articles = get_articles_count()
        unprocessed_articles = get_unprocessed_count()
        processed_articles = total_articles - unprocessed_articles
        
        return {
            "status": "running" if unprocessed_articles > 0 else "idle",
            "total_articles": total_articles,
            "processed_articles": processed_articles,
            "unprocessed_articles": unprocessed_articles,
            "pipeline": "preprocess -> cluster -> summarize"
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting ingest status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# === UPGRADED PIPELINE ENDPOINTS ===

@router.post("/v2/", response_model=IngestResponse)
async def ingest_article_v2(article: ArticleIngest):
    """
    Ingest article using the upgraded pipeline: Preprocess → NER Tag → Embed+Cluster → Store
    
    Args:
        article: Article data with id, title, body, region, topic, source_url
        
    Returns:
        IngestResponse: Confirmation of ingestion with article ID
    """
    try:
        # Generate ID if not provided
        if not article.id:
            article.id = str(uuid.uuid4())
        
        logger.info(f"🚀 [V2] Ingesting article {article.id}: {article.title}")
        
        # Import the new pipeline task
        from app.tasks.pipeline_tasks import run_article_pipeline
        
        # Prepare article data for the new pipeline
        article_data = {
            "article_id": article.id,
            "title": article.title,
            "body": article.body,
            "region": article.region,
            "topic": article.topic,
            "source_url": article.source_url
        }
        
        # Run the upgraded pipeline
        task = run_article_pipeline.delay(article_data)
        
        logger.info(f"✅ [V2] Article {article.id} queued for upgraded pipeline (task: {task.id})")
        
        return IngestResponse(
            id=article.id,
            status="pipeline_started",
            message=f"Article queued for upgraded pipeline. Task ID: {task.id}"
        )
        
    except Exception as e:
        logger.error(f"❌ [V2] Error ingesting article: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest article with upgraded pipeline: {str(e)}"
        )

@router.post("/v2/batch/")
async def ingest_batch_v2(articles: List[ArticleIngest]):
    """
    Batch ingest articles using the upgraded pipeline (parallel processing)
    
    Args:
        articles: List of article data
        
    Returns:
        dict: Batch processing results
    """
    try:
        logger.info(f"🔄 [V2] Batch ingesting {len(articles)} articles")
        
        # Import the batch pipeline task
        from app.tasks.pipeline_tasks import run_batch_pipeline
        
        # Prepare articles data
        articles_data = []
        for article in articles:
            if not article.id:
                article.id = str(uuid.uuid4())
                
            articles_data.append({
                "article_id": article.id,
                "title": article.title,
                "body": article.body,
                "region": article.region,
                "topic": article.topic,
                "source_url": article.source_url
            })
        
        # Run batch pipeline
        task = run_batch_pipeline.delay(articles_data)
        
        logger.info(f"✅ [V2] Batch of {len(articles)} articles queued (task: {task.id})")
        
        return {
            "status": "batch_started",
            "article_count": len(articles),
            "task_id": task.id,
            "articles": [{"id": article.id, "title": article.title} for article in articles],
            "pipeline": "preprocess → ner_tag → embed_cluster → store"
        }
        
    except Exception as e:
        logger.error(f"❌ [V2] Error batch ingesting articles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to batch ingest articles: {str(e)}"
        )

@router.post("/v2/batch-optimized/")
async def ingest_batch_optimized(articles: List[ArticleIngest], batch_size: int = 10):
    """
    High-performance batch ingest with optimized Supabase storage
    
    Args:
        articles: List of article data
        batch_size: Number of articles to store in each Supabase batch (default: 10)
        
    Returns:
        dict: Optimized batch processing results
    """
    try:
        logger.info(f"🚀 [V2-OPTIMIZED] Batch ingesting {len(articles)} articles with batch_size={batch_size}")
        
        # Import the optimized batch processing task
        from app.tasks.pipeline_tasks import process_article_batch
        
        # Prepare articles data
        articles_data = []
        for article in articles:
            if not article.id:
                article.id = str(uuid.uuid4())
                
            articles_data.append({
                "article_id": article.id,
                "title": article.title,
                "body": article.body,
                "region": article.region,
                "topic": article.topic,
                "source_url": article.source_url
            })
        
        # Run optimized batch processing
        task = process_article_batch.delay(articles_data, batch_size)
        
        logger.info(f"✅ [V2-OPTIMIZED] Batch of {len(articles)} articles queued for optimized processing (task: {task.id})")
        
        return {
            "status": "optimized_batch_started",
            "article_count": len(articles),
            "batch_size": batch_size,
            "expected_db_batches": (len(articles) + batch_size - 1) // batch_size,
            "task_id": task.id,
            "articles": [{"id": article.id, "title": article.title} for article in articles],
            "pipeline": "batch_process → batch_tag → batch_embed_cluster → batch_store_supabase",
            "performance_note": f"Articles will be stored to Supabase in batches of {batch_size} for optimal performance"
        }
        
    except Exception as e:
        logger.error(f"❌ [V2-OPTIMIZED] Error optimized batch ingesting articles: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to optimized batch ingest articles: {str(e)}"
        ) 