#!/usr/bin/env python3
"""
Report API Router for StraitWatch Backend
Handles report generation endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging

from app.pipelines.daily_digest import generate_daily_digest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/report", tags=["reports"])

class DigestResponse(BaseModel):
    """Response model for daily digest"""
    digest: str
    status: str = "success"
    message: str = "Daily digest generated successfully"

@router.get("/today", response_model=DigestResponse)
async def get_today_digest():
    """
    Get today's daily digest
    
    Returns:
        DigestResponse: Formatted markdown digest of completed clusters from last 24 hours
    """
    try:
        logger.info("📊 Generating today's daily digest")
        
        # Generate the daily digest
        digest_content = generate_daily_digest()
        
        logger.info("✅ Daily digest generated successfully")
        
        return DigestResponse(
            digest=digest_content,
            status="success",
            message="Daily digest generated successfully"
        )
        
    except Exception as e:
        logger.error(f"❌ Error generating daily digest: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate daily digest: {str(e)}"
        )

@router.get("/today/status")
async def get_digest_status():
    """
    Get status of daily digest generation capabilities
    
    Returns:
        dict: Status information about digest generation
    """
    try:
        from utils.supabase_client import get_supabase_client
        from datetime import datetime, timedelta
        
        supabase = get_supabase_client()
        
        # Check database connectivity
        test_result = supabase.table('clusters').select('id').limit(1).execute()
        db_status = "connected" if test_result is not None else "disconnected"
        
        # Get cluster counts
        yesterday = datetime.now() - timedelta(days=1)
        yesterday_str = yesterday.isoformat()
        
        # Total clusters
        total_result = supabase.table('clusters').select('id', count='exact').execute()
        total_clusters = total_result.count or 0
        
        # Completed clusters in last 24 hours
        completed_result = supabase.table('clusters').select(
            'id', count='exact'
        ).eq('status', 'complete').gte('created_at', yesterday_str).execute()
        completed_clusters = completed_result.count or 0
        
        return {
            "status": "available",
            "database": db_status,
            "total_clusters": total_clusters,
            "completed_clusters_24h": completed_clusters,
            "digest_type": "daily_summary",
            "time_range": "last_24_hours",
            "format": "markdown"
        }
        
    except Exception as e:
        logger.error(f"❌ Error checking digest status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "database": "unknown"
        } 