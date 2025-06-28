#!/usr/bin/env python3
"""
Metadata extraction module for cluster analysis.
Extracts keywords, topics, and other metadata from clustered articles.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import Counter, defaultdict
import re
import json
from datetime import datetime
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('metadata_extraction')

def extract_cluster_metadata(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract comprehensive metadata for each cluster.
    
    Args:
        articles: List of articles with cluster assignments
        
    Returns:
        Dictionary with cluster metadata
    """
    logger.info("Extracting cluster metadata...")
    
    # Group articles by cluster
    clusters = defaultdict(list)
    for article in articles:
        cluster_id = article.get('cluster_id', -1)
        clusters[cluster_id].append(article)
    
    # Extract metadata for each cluster
    cluster_metadata = {}
    
    for cluster_id, cluster_articles in clusters.items():
        if cluster_id == -1:
            continue  # Skip noise cluster
        
        metadata = _extract_single_cluster_metadata(cluster_articles, cluster_id)
        cluster_metadata[int(cluster_id)] = metadata
    
    # Add overall statistics
    overall_stats = {
        'total_articles': len(articles),
        'total_clusters': len(cluster_metadata),
        'noise_articles': len(clusters.get(-1, [])),
        'avg_cluster_size': np.mean([len(articles) for articles in cluster_metadata.values()]) if cluster_metadata else 0,
        'extraction_timestamp': datetime.now().isoformat()
    }
    
    result = {
        'clusters': cluster_metadata,
        'overall_stats': overall_stats
    }
    
    logger.info(f"Extracted metadata for {len(cluster_metadata)} clusters")
    return result

def _extract_single_cluster_metadata(articles: List[Dict[str, Any]], cluster_id: int) -> Dict[str, Any]:
    """Extract metadata for a single cluster."""
    
    # Basic cluster info
    metadata = {
        'cluster_id': cluster_id,
        'size': len(articles),
        'articles': []
    }
    
    # Extract text and tags for analysis
    all_text = []
    all_tags = []
    all_titles = []
    escalation_scores = []
    sources = []
    timestamps = []
    countries = []
    
    for article in articles:
        # Basic article info
        article_info = {
            'id': article.get('id'),
            'title': article.get('title', ''),
            'url': article.get('url', ''),
            'source': article.get('source', ''),
            'timestamp': article.get('timestamp', article.get('date', ''))
        }
        metadata['articles'].append(article_info)
        
        # Collect text data
        title = article.get('title', '')
        text = article.get('text', article.get('content', ''))
        all_titles.append(title)
        all_text.append(f"{title} {text}")
        
        # Collect tags
        tags = article.get('tags', [])
        if isinstance(tags, list):
            all_tags.extend(tags)
        
        # Collect other metadata
        escalation_score = article.get('escalation_score')
        if escalation_score is not None:
            escalation_scores.append(float(escalation_score))
        
        source = article.get('source', '')
        if source:
            sources.append(source)
        
        timestamp = article.get('timestamp', article.get('date', ''))
        if timestamp:
            timestamps.append(timestamp)
        
        country = article.get('country', article.get('ActionGeo_CountryCode', ''))
        if country:
            countries.append(country)
    
    # Extract keywords from titles and text
    keywords = get_cluster_keywords(all_text, top_k=20)
    title_keywords = get_cluster_keywords(all_titles, top_k=10)
    
    # Analyze tags
    tag_counts = Counter(all_tags)
    top_tags = tag_counts.most_common(10)
    
    # Calculate statistics
    metadata.update({
        'keywords': keywords,
        'title_keywords': title_keywords,
        'top_tags': top_tags,
        'unique_tags': len(set(all_tags)),
        'total_tag_mentions': len(all_tags)
    })
    
    # Escalation score statistics
    if escalation_scores:
        metadata['escalation_stats'] = {
            'mean': float(np.mean(escalation_scores)),
            'std': float(np.std(escalation_scores)),
            'min': float(np.min(escalation_scores)),
            'max': float(np.max(escalation_scores)),
            'median': float(np.median(escalation_scores))
        }
    
    # Source distribution
    if sources:
        source_counts = Counter(sources)
        metadata['top_sources'] = source_counts.most_common(5)
        metadata['unique_sources'] = len(set(sources))
    
    # Time span analysis
    if timestamps:
        valid_timestamps = [ts for ts in timestamps if ts]
        if valid_timestamps:
            try:
                # Try to parse timestamps
                parsed_times = []
                for ts in valid_timestamps:
                    if isinstance(ts, str):
                        # Try common formats
                        for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                            try:
                                parsed_times.append(datetime.strptime(ts[:len(fmt)], fmt))
                                break
                            except:
                                continue
                
                if parsed_times:
                    metadata['time_span'] = {
                        'earliest': min(parsed_times).isoformat(),
                        'latest': max(parsed_times).isoformat(),
                        'span_days': (max(parsed_times) - min(parsed_times)).days
                    }
            except Exception as e:
                logger.warning(f"Could not parse timestamps for cluster {cluster_id}: {e}")
    
    # Geographic distribution
    if countries:
        country_counts = Counter(countries)
        metadata['top_countries'] = country_counts.most_common(5)
        metadata['unique_countries'] = len(set(countries))
    
    # Topic detection using simple heuristics
    metadata['detected_topics'] = _detect_cluster_topics(all_text, all_tags)
    
    return metadata

def get_cluster_keywords(texts: List[str], top_k: int = 20, min_length: int = 3) -> List[Tuple[str, int]]:
    """
    Extract top keywords from cluster texts using simple frequency analysis.
    
    Args:
        texts: List of texts from cluster articles
        top_k: Number of top keywords to return
        min_length: Minimum keyword length
        
    Returns:
        List of tuples (keyword, frequency) sorted by frequency
    """
    # Combine all texts
    combined_text = ' '.join(texts).lower()
    
    # Remove common stopwords and extract words
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after',
        'above', 'below', 'under', 'between', 'is', 'are', 'was', 'were', 'be', 'been',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could',
        'can', 'may', 'might', 'must', 'shall', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them',
        'my', 'your', 'his', 'her', 'its', 'our', 'their', 'said', 'says', 'not'
    }
    
    # Extract words using regex
    words = re.findall(r'\b[a-zA-Z]+\b', combined_text)
    
    # Filter words
    filtered_words = [
        word for word in words 
        if len(word) >= min_length and word.lower() not in stopwords
    ]
    
    # Count frequencies
    word_counts = Counter(filtered_words)
    
    # Return top keywords
    return word_counts.most_common(top_k)

def _detect_cluster_topics(texts: List[str], tags: List[str]) -> List[str]:
    """
    Detect main topics in a cluster using keywords and tags.
    
    Args:
        texts: List of texts from cluster articles
        tags: List of all tags from cluster articles
        
    Returns:
        List of detected topics
    """
    topics = []
    
    # Topic categories based on common OSINT topics
    topic_keywords = {
        'Military': ['military', 'army', 'navy', 'air force', 'soldiers', 'troops', 'defense', 'weapons', 'missile', 'aircraft'],
        'Politics': ['government', 'president', 'minister', 'parliament', 'election', 'political', 'policy', 'diplomatic'],
        'Economics': ['economic', 'trade', 'business', 'market', 'finance', 'investment', 'gdp', 'inflation', 'currency'],
        'Technology': ['technology', 'cyber', 'digital', 'artificial intelligence', 'computer', 'internet', 'software'],
        'Security': ['security', 'terrorism', 'threat', 'intelligence', 'surveillance', 'border', 'police'],
        'International': ['international', 'foreign', 'global', 'world', 'nations', 'countries', 'bilateral'],
        'Conflict': ['conflict', 'war', 'violence', 'attack', 'battle', 'fight', 'tension', 'dispute'],
        'Diplomacy': ['diplomacy', 'negotiations', 'treaty', 'agreement', 'summit', 'talks', 'relations'],
        'Energy': ['energy', 'oil', 'gas', 'nuclear', 'renewable', 'electricity', 'power'],
        'Environment': ['environment', 'climate', 'pollution', 'greenhouse', 'carbon', 'renewable', 'green']
    }
    
    # Combine all text for analysis
    combined_text = ' '.join(texts).lower()
    tag_text = ' '.join(tags).lower()
    all_text = f"{combined_text} {tag_text}"
    
    # Check for topic keywords
    for topic, keywords in topic_keywords.items():
        keyword_count = sum(1 for keyword in keywords if keyword in all_text)
        if keyword_count >= 2:  # Require at least 2 keyword matches
            topics.append(topic)
    
    return topics

def generate_cluster_summary(cluster_metadata: Dict[str, Any]) -> str:
    """
    Generate a human-readable summary of a cluster.
    
    Args:
        cluster_metadata: Metadata dictionary for a single cluster
        
    Returns:
        Human-readable cluster summary
    """
    cluster_id = cluster_metadata.get('cluster_id', 'Unknown')
    size = cluster_metadata.get('size', 0)
    
    summary = f"Cluster {cluster_id} ({size} articles)\n"
    summary += "=" * 40 + "\n"
    
    # Top keywords
    keywords = cluster_metadata.get('keywords', [])
    if keywords:
        top_keywords = [word for word, count in keywords[:5]]
        summary += f"Key terms: {', '.join(top_keywords)}\n"
    
    # Top tags
    top_tags = cluster_metadata.get('top_tags', [])
    if top_tags:
        tag_names = [tag for tag, count in top_tags[:5]]
        summary += f"Main tags: {', '.join(tag_names)}\n"
    
    # Topics
    topics = cluster_metadata.get('detected_topics', [])
    if topics:
        summary += f"Topics: {', '.join(topics)}\n"
    
    # Escalation score
    escalation_stats = cluster_metadata.get('escalation_stats')
    if escalation_stats:
        mean_score = escalation_stats.get('mean', 0)
        summary += f"Avg escalation score: {mean_score:.2f}\n"
    
    # Time span
    time_span = cluster_metadata.get('time_span')
    if time_span:
        span_days = time_span.get('span_days', 0)
        summary += f"Time span: {span_days} days\n"
    
    # Top sources
    top_sources = cluster_metadata.get('top_sources', [])
    if top_sources:
        source_names = [source for source, count in top_sources[:3]]
        summary += f"Main sources: {', '.join(source_names)}\n"
    
    # Countries
    top_countries = cluster_metadata.get('top_countries', [])
    if top_countries:
        country_names = [country for country, count in top_countries[:3]]
        summary += f"Countries: {', '.join(country_names)}\n"
    
    summary += "\n"
    return summary

def save_cluster_metadata_report(metadata: Dict[str, Any], 
                                output_file: str = "analytics/cluster_metadata_report.json") -> None:
    """
    Save cluster metadata to a JSON report file.
    
    Args:
        metadata: Complete cluster metadata dictionary
        output_file: Output file path
    """
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Cluster metadata report saved to {output_file}")

def create_cluster_comparison_report(metadata: Dict[str, Any], 
                                   output_file: str = "analytics/cluster_comparison.txt") -> None:
    """
    Create a human-readable comparison report of all clusters.
    
    Args:
        metadata: Complete cluster metadata dictionary
        output_file: Output file path
    """
    import os
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("CLUSTER ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        # Overall statistics
        overall_stats = metadata.get('overall_stats', {})
        f.write(f"Total articles: {overall_stats.get('total_articles', 0)}\n")
        f.write(f"Total clusters: {overall_stats.get('total_clusters', 0)}\n")
        f.write(f"Noise articles: {overall_stats.get('noise_articles', 0)}\n")
        f.write(f"Average cluster size: {overall_stats.get('avg_cluster_size', 0):.1f}\n\n")
        
        # Individual cluster summaries
        clusters = metadata.get('clusters', {})
        for cluster_id in sorted(clusters.keys()):
            cluster_data = clusters[cluster_id]
            summary = generate_cluster_summary(cluster_data)
            f.write(summary)
    
    logger.info(f"Cluster comparison report saved to {output_file}")

def extract_trending_topics(articles: List[Dict[str, Any]], 
                          time_window_hours: int = 24,
                          min_cluster_size: int = 3) -> List[Dict[str, Any]]:
    """
    Extract trending topics based on recent cluster activity.
    
    Args:
        articles: List of articles with timestamps and cluster assignments
        time_window_hours: Time window to consider for trending analysis
        min_cluster_size: Minimum cluster size to consider
        
    Returns:
        List of trending topic dictionaries
    """
    logger.info(f"Extracting trending topics in last {time_window_hours} hours...")
    
    from datetime import datetime, timedelta
    
    # Filter recent articles
    now = datetime.now()
    cutoff_time = now - timedelta(hours=time_window_hours)
    
    recent_articles = []
    for article in articles:
        timestamp_str = article.get('timestamp', article.get('date', ''))
        if timestamp_str:
            try:
                # Try to parse timestamp
                for fmt in ['%Y-%m-%d', '%Y-%m-%d %H:%M:%S', '%Y-%m-%dT%H:%M:%S']:
                    try:
                        timestamp = datetime.strptime(timestamp_str[:len(fmt)], fmt)
                        if timestamp >= cutoff_time:
                            recent_articles.append(article)
                        break
                    except:
                        continue
            except:
                continue
    
    if not recent_articles:
        logger.warning("No recent articles found")
        return []
    
    # Group by cluster and analyze
    cluster_groups = defaultdict(list)
    for article in recent_articles:
        cluster_id = article.get('cluster_id', -1)
        if cluster_id != -1:  # Skip noise
            cluster_groups[cluster_id].append(article)
    
    # Filter clusters by minimum size
    trending_clusters = {
        cid: articles for cid, articles in cluster_groups.items() 
        if len(articles) >= min_cluster_size
    }
    
    # Extract metadata for trending clusters
    trending_topics = []
    for cluster_id, cluster_articles in trending_clusters.items():
        metadata = _extract_single_cluster_metadata(cluster_articles, cluster_id)
        
        # Add trending metrics
        trending_score = len(cluster_articles) / time_window_hours  # Articles per hour
        metadata['trending_score'] = trending_score
        metadata['time_window_hours'] = time_window_hours
        
        trending_topics.append(metadata)
    
    # Sort by trending score
    trending_topics.sort(key=lambda x: x['trending_score'], reverse=True)
    
    logger.info(f"Found {len(trending_topics)} trending topics")
    return trending_topics

if __name__ == "__main__":
    # Example usage
    sample_articles = [
        {
            'id': 1,
            'title': 'Military exercise in Taiwan',
            'text': 'Taiwan conducted military exercise with advanced weapons',
            'cluster_id': 0,
            'tags': ['military', 'taiwan', 'defense'],
            'source': 'Reuters',
            'timestamp': '2024-01-15',
            'escalation_score': 0.7
        },
        {
            'id': 2,
            'title': 'China responds to Taiwan drills',
            'text': 'China announced military response to Taiwan exercise',
            'cluster_id': 0,
            'tags': ['military', 'china', 'taiwan'],
            'source': 'BBC',
            'timestamp': '2024-01-15',
            'escalation_score': 0.8
        }
    ]
    
    # Test metadata extraction
    metadata = extract_cluster_metadata(sample_articles)
    print(json.dumps(metadata, indent=2))