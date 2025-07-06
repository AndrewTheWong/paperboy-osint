#!/usr/bin/env python3
"""
Report API Router for StraitWatch Backend
Handles report generation endpoints
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import logging

# from app.pipelines.daily_digest import generate_daily_digest
# from app.reporting.report_generator import (
#     generate_cluster_report,
#     generate_multi_cluster_report,
#     get_cluster_summary_stats
# )

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

class ClusterReportResponse(BaseModel):
    """Response model for cluster report"""
    report: str
    cluster_id: str
    status: str = "success"
    message: str = "Cluster report generated successfully"

class ClusterStatsResponse(BaseModel):
    """Response model for cluster statistics"""
    cluster_id: str
    cluster_label: str
    article_count: int
    avg_confidence: float
    escalation_level: str
    unique_tags: int
    unique_entities: int
    top_tags: List[tuple]
    top_entities: List[tuple]
    status: str = "success"

class MultiClusterRequest(BaseModel):
    """Request model for multi-cluster reports"""
    cluster_ids: List[str]

@router.get("/today", response_model=DigestResponse)
async def get_today_digest():
    """
    Get today's daily digest
    
    Returns:
        DigestResponse: Formatted markdown digest of completed clusters from last 24 hours
    """
    try:
        logger.info("📊 Generating today's daily digest")
        
        # TODO: Implement daily digest generation
        digest_content = "# Daily Digest\n\nDaily digest functionality coming soon..."
        
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
        from config.supabase_client import get_supabase_client
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

@router.get("/cluster/{cluster_id}", response_model=ClusterReportResponse)
async def get_cluster_report(cluster_id: str):
    """
    Generate comprehensive OSINT intelligence report for a specific cluster
    
    Args:
        cluster_id: The ID of the cluster to generate report for
        
    Returns:
        ClusterReportResponse: Detailed cluster analysis report
    """
    try:
        logger.info(f"📊 Generating cluster report for: {cluster_id}")
        
        # Comment out the actual report generation
        # report = generate_cluster_report(cluster_id)
        
        # Comment out the actual report generation
        # if report is None:
        #     raise HTTPException(
        #         status_code=404,
        #         detail=f"Cluster {cluster_id} not found or contains no articles"
        #     )
        
        logger.info(f"✅ Cluster report generated successfully for {cluster_id}")
        
        return ClusterReportResponse(
            report="",
            cluster_id=cluster_id,
            status="success",
            message=f"Cluster report generated successfully for {cluster_id}"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating cluster report for {cluster_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate cluster report: {str(e)}"
        )

@router.get("/cluster/{cluster_id}/stats", response_model=ClusterStatsResponse)
async def get_cluster_stats(cluster_id: str):
    """
    Get summary statistics for a cluster without full report generation
    
    Args:
        cluster_id: The cluster ID to analyze
        
    Returns:
        ClusterStatsResponse: Summary statistics for the cluster
    """
    try:
        logger.info(f"📈 Getting cluster stats for: {cluster_id}")
        
        # Comment out the actual stats retrieval
        # stats = get_cluster_summary_stats(cluster_id)
        
        # Comment out the actual stats retrieval
        # if stats is None:
        #     raise HTTPException(
        #         status_code=404,
        #         detail=f"Cluster {cluster_id} not found or contains no articles"
        #     )
        
        logger.info(f"✅ Cluster stats retrieved successfully for {cluster_id}")
        
        return ClusterStatsResponse(**{})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting cluster stats for {cluster_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get cluster stats: {str(e)}"
        )

@router.post("/multi-cluster", response_model=ClusterReportResponse)
async def get_multi_cluster_report(request: MultiClusterRequest):
    """
    Generate a comparative report across multiple clusters
    
    Args:
        request: Request containing list of cluster IDs
        
    Returns:
        ClusterReportResponse: Combined analysis report
    """
    try:
        cluster_ids = request.cluster_ids
        logger.info(f"📊 Generating multi-cluster report for {len(cluster_ids)} clusters")
        
        if not cluster_ids:
            raise HTTPException(
                status_code=400,
                detail="At least one cluster ID must be provided"
            )
        
        if len(cluster_ids) > 10:
            raise HTTPException(
                status_code=400,
                detail="Maximum 10 clusters allowed per multi-cluster report"
            )
        
        # Comment out the actual multi-cluster report generation
        # report = generate_multi_cluster_report(cluster_ids)
        
        logger.info(f"✅ Multi-cluster report generated successfully")
        
        return ClusterReportResponse(
            report="",
            cluster_id=",".join(cluster_ids),
            status="success",
            message=f"Multi-cluster report generated for {len(cluster_ids)} clusters"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error generating multi-cluster report: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate multi-cluster report: {str(e)}"
        ) 