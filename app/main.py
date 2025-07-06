#!/usr/bin/env python3
"""
FastAPI main application for Paperboy backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="StraitWatch Backend",
    description="Intelligence pipeline backend for news analysis",
    version="1.0.0"
)

# Mount static files
app.mount("/static", StaticFiles(directory="app/static"), name="static")

# Include routers
from app.api import report, ingest
app.include_router(report.router)
app.include_router(ingest.router)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class Article(BaseModel):
    title: str
    content: str
    url: str
    source: str
    published_at: Optional[str] = None

class IngestResponse(BaseModel):
    id: int
    status: str
    message: str

class StatusResponse(BaseModel):
    total_articles: int
    processed_articles: int
    unprocessed_articles: int
    status: str

class ReportRequest(BaseModel):
    report_type: str = "storage_based"
    include_raw_articles: bool = False
    cluster_threshold: int = 3

class ReportResponse(BaseModel):
    status: str
    message: str
    report_data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class ScraperRequest(BaseModel):
    sources: Optional[List[Dict[str, Any]]] = None
    max_articles_per_source: int = 10
    use_default_sources: bool = True

class ScraperResponse(BaseModel):
    status: str
    message: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "paperboy-backend"}

# Task status endpoint
@app.get("/tasks/{task_id}/status")
async def get_task_status(task_id: str):
    """Get the status of a Celery task"""
    try:
        from celery.result import AsyncResult
        from app.celery_worker import celery_app
        
        result = AsyncResult(task_id, app=celery_app)
        
        return {
            "task_id": task_id,
            "status": result.status,
            "ready": result.ready(),
            "successful": result.successful(),
            "failed": result.failed(),
            "info": result.info if result.ready() else None
        }
    except Exception as e:
        return {
            "task_id": task_id,
            "status": "ERROR",
            "error": str(e)
        }

# Article ingestion endpoint
@app.post("/ingest/articles", response_model=IngestResponse)
async def ingest_article(article: Article):
    """Ingest a single article"""
    try:
        # Import here to avoid circular imports
        from app.utils.supabase_client import get_supabase_client
        
        supabase = get_supabase_client()
        
        # Check if article already exists
        existing = supabase.table('articles').select('id').eq('url', article.url).execute()
        
        if existing.data:
            return IngestResponse(
                id=existing.data[0]['id'],
                status="exists",
                message="Article already exists"
            )
        
        # Insert article
        result = supabase.table('articles').insert({
            'title': article.title,
            'content': article.content,
            'url': article.url,
            'source': article.source,
            'published_at': article.published_at,
            'relevant': None  # Will be set by processing pipeline
        }).execute()
        
        if result.data:
            article_id = result.data[0]['id']
            logger.info(f"Ingested article {article_id}: {article.title}")
            
            return IngestResponse(
                id=article_id,
                status="ingested",
                message="Article ingested successfully"
            )
        else:
            raise HTTPException(status_code=500, detail="Failed to insert article")
            
    except Exception as e:
        logger.error(f"Error ingesting article: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Status endpoint


# Articles list endpoint
@app.get("/articles")
async def get_articles(limit: int = 100, offset: int = 0):
    """Get list of articles"""
    try:
        from app.utils.supabase_client import get_supabase_client
        
        supabase = get_supabase_client()
        
        result = supabase.table('articles').select('*').range(offset, offset + limit - 1).execute()
        
        return {
            "articles": result.data,
            "total": len(result.data),
            "limit": limit,
            "offset": offset
        }
        
    except Exception as e:
        logger.error(f"Error getting articles: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Report generation endpoint
@app.post("/reports/generate", response_model=ReportResponse)
async def generate_report(request: ReportRequest):
    """Generate intelligence report based on stored articles"""
    try:
        # Import the storage-based reporter
        from storage_based_reporter import StorageBasedReporter
        
        logger.info(f"🎯 Generating {request.report_type} report")
        
        # Create reporter instance
        reporter = StorageBasedReporter()
        
        # Generate report
        report_data = reporter.generate_storage_based_report()
        
        # Check for errors
        if 'error' in report_data:
            return ReportResponse(
                status="error",
                message="Report generation failed",
                error=report_data['error']
            )
        
        # Remove raw articles if not requested (to reduce response size)
        if not request.include_raw_articles and 'raw_articles' in report_data:
            del report_data['raw_articles']
        
        logger.info(f"✅ Report generated successfully: {report_data.get('metadata', {}).get('total_clusters', 0)} clusters")
        
        return ReportResponse(
            status="success",
            message="Report generated successfully",
            report_data=report_data
        )
        
    except Exception as e:
        logger.error(f"Error generating report: {e}")
        return ReportResponse(
            status="error",
            message="Report generation failed",
            error=str(e)
        )

# Quick report endpoint (simplified)
@app.get("/reports/quick")
async def get_quick_report():
    """Get a quick summary report without full analysis"""
    try:
        from app.utils.supabase_client import get_supabase_client
        
        supabase = get_supabase_client()
        
        # Get basic stats
        total_result = supabase.table('articles').select('id', count='exact').execute()
        total_articles = total_result.count or 0
        
        processed_result = supabase.table('articles').select('id', count='exact').not_.is_('relevant', 'null').execute()
        processed_articles = processed_result.count or 0
        
        # Get recent articles
        recent_articles = supabase.table('articles').select('title, source, created_at').order('created_at', desc=True).limit(5).execute()
        
        return {
            "status": "success",
            "summary": {
                "total_articles": total_articles,
                "processed_articles": processed_articles,
                "unprocessed_articles": total_articles - processed_articles,
                "recent_articles": recent_articles.data if recent_articles.data else []
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating quick report: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Report status endpoint
@app.get("/reports/status")
async def get_report_status():
    """Get status of report generation capabilities"""
    try:
        from app.utils.supabase_client import get_supabase_client
        
        supabase = get_supabase_client()
        
        # Check database connectivity
        test_result = supabase.table('articles').select('id').limit(1).execute()
        db_status = "connected" if test_result is not None else "disconnected"
        
        # Get article counts
        total_result = supabase.table('articles').select('id', count='exact').execute()
        total_articles = total_result.count or 0
        
        return {
            "status": "available",
            "database": db_status,
            "total_articles": total_articles,
            "report_types": [
                "storage_based",
                "quick_summary"
            ],
            "capabilities": [
                "thematic_clustering",
                "threat_assessment", 
                "executive_summary",
                "raw_article_analysis"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error checking report status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Scraper endpoints
@app.post("/scraper/run", response_model=ScraperResponse)
async def run_scraper(request: ScraperRequest):
    """Run the scraper to fetch articles and store in Supabase"""
    try:
        from app.tasks.scraper import run_async_scraper
        from app.services.scraper_service import TAIWAN_STRAIT_SOURCES
        
        logger.info("🚀 Starting async scraper task...")
        
        # Determine sources to use
        if request.use_default_sources:
            sources = TAIWAN_STRAIT_SOURCES
        elif request.sources:
            sources = request.sources
        else:
            sources = TAIWAN_STRAIT_SOURCES
        
        # Run scraper as Celery task
        task = run_async_scraper.delay(sources, request.max_articles_per_source)
        
        logger.info(f"✅ Async scraper task queued: {task.id}")
        
        return ScraperResponse(
            status="queued",
            message=f"Scraper task queued successfully. Task ID: {task.id}",
            results={"task_id": task.id}
        )
        
    except Exception as e:
        logger.error(f"Error queuing scraper task: {e}")
        return ScraperResponse(
            status="error",
            message="Failed to queue scraper task",
            error=str(e)
        )

@app.get("/scraper/status")
async def get_scraper_status():
    """Get scraper status and available sources"""
    try:
        from app.services.scraper_service import TAIWAN_STRAIT_SOURCES
        from app.utils.supabase_client import get_supabase_client
        
        supabase = get_supabase_client()
        
        # Get recent articles count
        recent_result = supabase.table('articles').select('id', count='exact').gte('created_at', datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat()).execute()
        today_articles = recent_result.count or 0
        
        # Get total articles
        total_result = supabase.table('articles').select('id', count='exact').execute()
        total_articles = total_result.count or 0
        
        return {
            "status": "available",
            "total_articles": total_articles,
            "today_articles": today_articles,
            "available_sources": len(TAIWAN_STRAIT_SOURCES),
            "default_sources": [
                {
                    "name": source["name"],
                    "url": source["url"],
                    "type": source["type"]
                }
                for source in TAIWAN_STRAIT_SOURCES
            ],
            "capabilities": [
                "parallel_scraping",
                "content_extraction",
                "duplicate_detection",
                "direct_supabase_storage"
            ]
        }
        
    except Exception as e:
        logger.error(f"Error checking scraper status: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

@app.get("/scraper/sources")
async def get_available_sources():
    """Get list of available scraping sources"""
    try:
        from app.services.scraper_service import TAIWAN_STRAIT_SOURCES
        
        return {
            "sources": TAIWAN_STRAIT_SOURCES,
            "total_sources": len(TAIWAN_STRAIT_SOURCES),
            "categories": {
                "taiwan_media": len([s for s in TAIWAN_STRAIT_SOURCES if "taiwan" in s["name"].lower()]),
                "chinese_media": len([s for s in TAIWAN_STRAIT_SOURCES if "china" in s["name"].lower()]),
                "international_media": len([s for s in TAIWAN_STRAIT_SOURCES if "china" not in s["name"].lower() and "taiwan" not in s["name"].lower()])
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting sources: {e}")
        return {
            "error": str(e)
        }

# Clustering endpoints
@app.post("/cluster/run")
async def run_clustering():
    """Run fast clustering on articles in database"""
    try:
        from app.services.fast_clustering import cluster_articles_complete_with_summaries
        from app.services.supabase import get_articles_with_embeddings, save_cluster
        import uuid
        
        logger.info("🚀 Starting fast clustering process...")
        
        # Get articles with embeddings
        articles_data = get_articles_with_embeddings(limit=200)
        
        if not articles_data:
            return {
                "status": "no_data",
                "message": "No articles with embeddings found",
                "clusters_created": 0
            }
        
        # Extract embeddings and metadata
        embeddings = []
        articles_metadata = []
        
        for article in articles_data:
            embedding = article.get('embedding', [])
            if isinstance(embedding, list) and len(embedding) > 0:
                embeddings.append(embedding)
                articles_metadata.append({
                    'title': article.get('title', ''),
                    'topic': article.get('topic', 'Unknown'),
                    'region': article.get('region', 'Unknown'),
                    'tags': article.get('tags', []),
                    'article_id': article.get('id', ''),
                    'content': article.get('content', '')
                })
        
        if not embeddings:
            return {
                "status": "no_embeddings",
                "message": "No valid embeddings found",
                "clusters_created": 0
            }
        
        # Run fast clustering with summaries
        clustering_results = cluster_articles_complete_with_summaries(
            embeddings=embeddings,
            articles=articles_metadata,
            num_clusters=None,
            use_faiss=False,
            max_concurrent_summaries=3
        )
        
        clusters = clustering_results['clusters']
        summaries = clustering_results['summaries']
        
        # Save clusters to database
        total_clusters_saved = 0
        
        for cluster_id, cluster_indices in clusters.items():
            if len(cluster_indices) >= 3:
                cluster_db_ids = []
                for idx in cluster_indices:
                    if idx < len(articles_data):
                        article = articles_data[idx]
                        article_id = article.get('id')
                        if article_id:
                            cluster_db_ids.append(article_id)
                
                if len(cluster_db_ids) >= 3:
                    summary_info = summaries.get(cluster_id, {})
                    theme = summary_info.get('primary_topic', 'Unknown')
                    text_summary = summary_info.get('text_summary', 'No summary available')
                    
                    unique_cluster_id = f"cluster_{cluster_id}_{str(uuid.uuid4())[:8]}"
                    
                    success = save_cluster(
                        cluster_id=unique_cluster_id,
                        article_ids=cluster_db_ids,
                        status='complete',
                        theme=theme,
                        summary=text_summary
                    )
                    
                    if success:
                        total_clusters_saved += 1
        
        logger.info(f"✅ Fast clustering complete: {total_clusters_saved} clusters saved")
        
        return {
            "status": "success",
            "message": f"Fast clustering completed successfully",
            "clusters_created": total_clusters_saved,
            "articles_processed": len(embeddings),
            "processing_time": clustering_results['processing_time']
        }
        
    except Exception as e:
        logger.error(f"Error in fast clustering: {e}")
        return {
            "status": "error",
            "message": "Fast clustering failed",
            "error": str(e)
        }

# Summarization endpoints
@app.post("/summarize/run")
async def run_summarization():
    """Trigger summarization of articles"""
    try:
        from app.tasks.summarize import summarize_articles_task
        
        logger.info("📝 Triggering summarization...")
        
        # Trigger summarization task
        task = summarize_articles_task.delay()
        
        return {
            "status": "success",
            "message": "Summarization task triggered",
            "task_id": task.id
        }
        
    except Exception as e:
        logger.error(f"Error triggering summarization: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Queue management endpoints
@app.post("/queue/clear")
async def clear_queue():
    """Clear the Redis queue"""
    try:
        import redis
        
        # Connect to Redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Clear all queues
        r.flushdb()
        
        logger.info("🗑️ Queue cleared")
        
        return {
            "status": "success",
            "message": "Queue cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"Error clearing queue: {e}")
        return {
            "status": "error",
            "error": str(e)
        }

# Enhanced status endpoint for dashboard
@app.get("/ingest/status")
async def get_enhanced_status():
    """Get enhanced status for dashboard"""
    try:
        from app.utils.supabase_client import get_supabase_client
        import redis
        
        supabase = get_supabase_client()
        
        # Get article counts
        total_result = supabase.table('articles').select('id', count='exact').execute()
        total_articles = total_result.count or 0
        
        processed_result = supabase.table('articles').select('id', count='exact').not_.is_('relevant', 'null').execute()
        processed_articles = processed_result.count or 0
        
        # Get cluster count
        cluster_result = supabase.table('clusters').select('id', count='exact').execute()
        clusters_created = cluster_result.count or 0
        
        # Get summary count (articles with summaries)
        summary_result = supabase.table('articles').select('id', count='exact').not_.is_('summary', 'null').execute()
        summaries_generated = summary_result.count or 0
        
        # Get queue size
        try:
            r = redis.Redis(host='localhost', port=6379, db=0)
            queue_size = r.llen('preprocess') if r.exists('preprocess') else 0
        except:
            queue_size = 0
        
        return {
            "articles_scraped": total_articles,
            "articles_processed": processed_articles,
            "clusters_created": clusters_created,
            "summaries_generated": summaries_generated,
            "queue_size": queue_size,
            "status": "active" if queue_size > 0 or processed_articles < total_articles else "inactive"
        }
        
    except Exception as e:
        logger.error(f"Error getting enhanced status: {e}")
        return {
            "articles_scraped": 0,
            "articles_processed": 0,
            "clusters_created": 0,
            "summaries_generated": 0,
            "queue_size": 0,
            "status": "error"
        }

# Root endpoint - serve dashboard
@app.get("/")
async def root():
    """Serve the dashboard"""
    return FileResponse("app/static/index.html")

# API info endpoint
@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Paperboy Backend API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "ingest": "/ingest/articles",
            "status": "/ingest/status",
            "articles": "/articles",
            "reports": {
                "generate": "/reports/generate",
                "quick": "/reports/quick", 
                "status": "/reports/status"
            },
            "daily_digest": {
                "today": "/report/today",
                "status": "/report/today/status"
            },
            "scraper": {
                "run": "/scraper/run",
                "status": "/scraper/status",
                "sources": "/scraper/sources"
            },
            "cluster": {
                "run": "/cluster/run"
            },
            "summarize": {
                "run": "/summarize/run"
            },
            "queue": {
                "clear": "/queue/clear"
            }
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    ) 