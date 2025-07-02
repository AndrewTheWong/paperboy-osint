#!/usr/bin/env python3
"""
FastAPI main application for Paperboy backend
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import asyncio
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="StraitWatch Backend",
    description="Intelligence pipeline backend for news analysis",
    version="1.0.0"
)

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

# Article ingestion endpoint
@app.post("/ingest/articles", response_model=IngestResponse)
async def ingest_article(article: Article):
    """Ingest a single article"""
    try:
        # Import here to avoid circular imports
        from utils.supabase_client import get_supabase_client
        
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
@app.get("/ingest/status", response_model=StatusResponse)
async def get_ingest_status():
    """Get ingestion pipeline status"""
    try:
        from utils.supabase_client import get_supabase_client
        
        supabase = get_supabase_client()
        
        # Get total articles
        total_result = supabase.table('articles').select('id', count='exact').execute()
        total_articles = total_result.count or 0
        
        # Get processed articles (those with relevant flag set)
        processed_result = supabase.table('articles').select('id', count='exact').not_.is_('relevant', 'null').execute()
        processed_articles = processed_result.count or 0
        
        unprocessed_articles = total_articles - processed_articles
        
        return StatusResponse(
            total_articles=total_articles,
            processed_articles=processed_articles,
            unprocessed_articles=unprocessed_articles,
            status="running" if unprocessed_articles > 0 else "idle"
        )
        
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Articles list endpoint
@app.get("/articles")
async def get_articles(limit: int = 100, offset: int = 0):
    """Get list of articles"""
    try:
        from utils.supabase_client import get_supabase_client
        
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
        from utils.supabase_client import get_supabase_client
        
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
        from utils.supabase_client import get_supabase_client
        
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
        from app.services.scraper_service import run_scraper, TAIWAN_STRAIT_SOURCES
        
        logger.info("🚀 Starting scraper...")
        
        # Determine sources to use
        if request.use_default_sources:
            sources = TAIWAN_STRAIT_SOURCES
        elif request.sources:
            sources = request.sources
        else:
            sources = TAIWAN_STRAIT_SOURCES
        
        # Run scraper
        results = await run_scraper(sources, request.max_articles_per_source)
        
        logger.info(f"✅ Scraper completed: {results['total_stored']} articles stored")
        
        return ScraperResponse(
            status="success",
            message="Scraper completed successfully",
            results=results
        )
        
    except Exception as e:
        logger.error(f"Error running scraper: {e}")
        return ScraperResponse(
            status="error",
            message="Scraper failed",
            error=str(e)
        )

@app.get("/scraper/status")
async def get_scraper_status():
    """Get scraper status and available sources"""
    try:
        from app.services.scraper_service import TAIWAN_STRAIT_SOURCES
        from utils.supabase_client import get_supabase_client
        
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

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
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