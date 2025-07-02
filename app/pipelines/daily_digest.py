#!/usr/bin/env python3
"""
Daily Digest Pipeline for StraitWatch Backend
Generates structured digest from completed clusters in the last 24 hours
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import defaultdict, Counter

from utils.supabase_client import get_supabase_client
from app.utils.formatter import format_digest

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_daily_digest() -> str:
    """
    Generate a markdown-style daily digest grouped by region and topic from completed clusters in Supabase.
    """
    # 1. Get clusters from last 24h
    since = (datetime.now() - timedelta(days=1)).isoformat()
    # REMOVE: clusters = fetch_completed_clusters(since)
    if not clusters:
        return "# Daily Digest\n\nNo completed clusters in the last 24 hours."

    # 2. For each cluster, fetch article metadata and assign region/topic
    grouped = defaultdict(lambda: defaultdict(list))
    for cluster in clusters:
        article_ids = cluster.get('article_ids') or []
        if not article_ids:
            continue
        # REMOVE: articles = fetch_article_metadata(article_ids)
        regions = [a.get('region') for a in articles if a.get('region')]
        topics = [a.get('topic') for a in articles if a.get('topic')]
        region = Counter(regions).most_common(1)[0][0] if regions else 'Unknown Region'
        topic = Counter(topics).most_common(1)[0][0] if topics else 'General'
        summary = cluster.get('summary', '(No summary)')
        grouped[region][topic].append(summary)

    # 3. Format as markdown
    return format_digest(grouped)

def _group_clusters_by_region_topic(clusters: List[Dict], supabase) -> Dict[str, Dict[str, List[Dict]]]:
    """
    Group clusters by region and topic
    
    Args:
        clusters: List of cluster data
        supabase: Supabase client
        
    Returns:
        Dict: Nested dictionary {region: {topic: [cluster_summaries]}}
    """
    digest_data = defaultdict(lambda: defaultdict(list))
    
    for cluster in clusters:
        try:
            # Get region and topic from cluster (they should be stored in the cluster)
            region = cluster.get('region', 'Unknown Region')
            topic = cluster.get('topic', 'General')
            
            # Get article details for this cluster
            article_ids = cluster.get('article_ids', [])
            if not article_ids:
                continue
                
            # Fetch articles for this cluster to get source information
            articles_result = supabase.table('articles').select(
                'id, title, source, published_at'
            ).in_('id', article_ids).execute()
            
            articles = articles_result.data if articles_result.data else []
            
            # Create cluster summary
            cluster_summary = {
                'theme': cluster.get('theme', 'Unknown Theme'),
                'summary': cluster.get('summary', 'No summary available'),
                'priority_level': cluster.get('priority_level', 'MEDIUM'),
                'escalation_score': cluster.get('escalation_score', 0),
                'article_count': len(articles),
                'sources': list(set([article.get('source', 'Unknown') for article in articles])),
                'created_at': cluster.get('created_at')
            }
            
            # Add to digest data
            digest_data[region][topic].append(cluster_summary)
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing cluster {cluster.get('id')}: {e}")
            continue
    
    return dict(digest_data)

def _format_digest(digest_data: Dict[str, Dict[str, List[Dict]]], total_clusters: int) -> str:
    """
    Format digest data into markdown string
    
    Args:
        digest_data: Nested dictionary of clusters by region and topic
        total_clusters: Total number of clusters processed
        
    Returns:
        str: Formatted markdown digest
    """
    today = datetime.now().strftime("%B %d, %Y")
    
    markdown = f"""# StraitWatch Daily Digest
## {today}

**Intelligence Summary**: {total_clusters} completed clusters analyzed in the last 24 hours

---

"""
    
    if not digest_data:
        markdown += "**No completed clusters found in the last 24 hours.**\n\n"
        return markdown
    
    # Sort regions and topics for consistent output
    for region in sorted(digest_data.keys()):
        markdown += f"## 🌍 {region}\n\n"
        
        region_data = digest_data[region]
        for topic in sorted(region_data.keys()):
            markdown += f"### 📋 {topic}\n\n"
            
            clusters = region_data[topic]
            for i, cluster in enumerate(clusters, 1):
                priority_emoji = _get_priority_emoji(cluster['priority_level'])
                escalation_emoji = _get_escalation_emoji(cluster['escalation_score'])
                
                markdown += f"#### {priority_emoji} {cluster['theme']} {escalation_emoji}\n\n"
                markdown += f"**Summary**: {cluster['summary']}\n\n"
                markdown += f"**Details**:\n"
                markdown += f"- Priority: {cluster['priority_level']}\n"
                markdown += f"- Escalation Score: {cluster['escalation_score']}\n"
                markdown += f"- Articles: {cluster['article_count']}\n"
                markdown += f"- Sources: {', '.join(cluster['sources'][:3])}{'...' if len(cluster['sources']) > 3 else ''}\n\n"
                
                if i < len(clusters):
                    markdown += "---\n\n"
        
        markdown += "---\n\n"
    
    # Add footer
    markdown += f"""
---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}*
*Total clusters analyzed: {total_clusters}*
"""
    
    return markdown

def _get_priority_emoji(priority: str) -> str:
    """Get emoji for priority level"""
    priority_map = {
        'HIGH': '🔴',
        'MEDIUM': '🟡', 
        'LOW': '🟢'
    }
    return priority_map.get(priority, '⚪')

def _get_escalation_emoji(score: float) -> str:
    """Get emoji for escalation score"""
    if score >= 8:
        return '🚨'
    elif score >= 6:
        return '⚠️'
    elif score >= 4:
        return '📈'
    else:
        return '📊'

def _format_empty_digest() -> str:
    """Format digest when no clusters are found"""
    today = datetime.now().strftime("%B %d, %Y")
    
    return f"""# StraitWatch Daily Digest
## {today}

**Intelligence Summary**: No completed clusters found in the last 24 hours

---

**Status**: No intelligence clusters were completed in the last 24 hours. This may indicate:
- All clusters are still in processing
- No new articles were ingested
- System is in maintenance mode

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}*
"""

def _format_error_digest(error: str) -> str:
    """Format digest when an error occurs"""
    today = datetime.now().strftime("%B %d, %Y")
    
    return f"""# StraitWatch Daily Digest
## {today}

**Error**: Failed to generate daily digest

**Details**: {error}

**Status**: Please check system logs and try again later.

---
*Generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}*
""" 