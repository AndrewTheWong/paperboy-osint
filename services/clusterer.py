#!/usr/bin/env python3
"""
Fast clustering service using MiniBatchKMeans for article analysis
"""

import numpy as np
import time
import logging
from typing import List, Dict, Any, Optional
from sklearn.cluster import MiniBatchKMeans
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cluster_articles_fast(embeddings: List[List[float]], num_clusters: int = 100, batch_size: int = 512, 
                         similarity_threshold: float = 0.85) -> Dict[int, List[int]]:
    """
    Fast clustering using MiniBatchKMeans with strict similarity filtering.
    
    Args:
        embeddings: List of SBERT embedding vectors (384d)
        num_clusters: How many clusters to create (more = stricter)
        batch_size: Batch size for MiniBatchKMeans
        similarity_threshold: Minimum cosine similarity to belong to cluster (0.85 = very strict)
    
    Returns:
        A dict mapping cluster_id -> list of article indices
    """
    start = time.time()
    
    if not embeddings:
        logger.warning("No embeddings provided for clustering")
        return {}
    
    X = np.array(embeddings, dtype=np.float32)
    logger.info(f"🔄 Starting strict clustering: {len(embeddings)} embeddings, {num_clusters} clusters, threshold={similarity_threshold}")
    
    # Use MiniBatchKMeans for speed
    kmeans = MiniBatchKMeans(
        n_clusters=num_clusters, 
        batch_size=batch_size, 
        random_state=42,
        n_init=5,  # Increase for better quality
        max_iter=200  # Increase for better quality
    )
    
    labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_
    
    # Group articles by cluster with similarity filtering
    clusters = {}
    rejected_count = 0
    
    for idx, label in enumerate(labels):
        # Calculate cosine similarity to cluster center
        article_embedding = X[idx]
        cluster_center = cluster_centers[label]
        
        # Normalize vectors for cosine similarity
        article_norm = article_embedding / (np.linalg.norm(article_embedding) + 1e-8)
        center_norm = cluster_center / (np.linalg.norm(cluster_center) + 1e-8)
        
        # Calculate cosine similarity
        similarity = np.dot(article_norm, center_norm)
        
        # Only add to cluster if similarity is above threshold
        if similarity >= similarity_threshold:
            clusters.setdefault(label, []).append(idx)
        else:
            rejected_count += 1
            logger.debug(f"Rejected article {idx} from cluster {label}: similarity={similarity:.3f} < {similarity_threshold}")
    
    elapsed = time.time() - start
    logger.info(f"⏱️ Strict clustering completed in {elapsed:.2f}s")
    logger.info(f"📊 Created {len(clusters)} clusters from {len(embeddings)} articles")
    logger.info(f"🚫 Rejected {rejected_count} articles for low similarity")
    
    return clusters

def summarize_clusters(articles: List[Dict], clusters: Dict[int, List[int]]) -> Dict[int, Dict[str, Any]]:
    """
    Create summary metadata per cluster (top tags, representative article, etc.)
    
    Args:
        articles: List of article dictionaries with metadata
        clusters: Dict mapping cluster_id -> list of article indices
    
    Returns:
        Dict mapping cluster_id -> cluster summary info
    """
    cluster_info = {}
    
    for cid, indices in clusters.items():
        if not indices:
            continue
            
        # Collect tags from all articles in cluster
        tag_counts = Counter()
        topics = Counter()
        regions = Counter()
        
        for idx in indices:
            if idx < len(articles):
                article = articles[idx]
                # Count tags
                if 'tags' in article:
                    tag_counts.update(article['tags'])
                # Count topics
                if 'topic' in article:
                    topics[article['topic']] += 1
                # Count regions
                if 'region' in article:
                    regions[article['region']] += 1
        
        # Get most common tags, topics, regions
        top_tags = [tag for tag, count in tag_counts.most_common(5)]
        top_topic = topics.most_common(1)[0][0] if topics else "Unknown"
        top_region = regions.most_common(1)[0][0] if regions else "Unknown"
        
        # Representative article (first one)
        representative_title = articles[indices[0]]['title'] if indices[0] < len(articles) else "Unknown"
        
        cluster_info[cid] = {
            "size": len(indices),
            "top_tags": top_tags,
            "primary_topic": top_topic,
            "primary_region": top_region,
            "representative_title": representative_title,
            "article_ids": indices
        }
    
    logger.info(f"📝 Generated summaries for {len(cluster_info)} clusters")
    return cluster_info

def estimate_optimal_clusters(embeddings: List[List[float]], max_clusters: int = 150) -> int:
    """
    Estimate optimal number of clusters for strict clustering
    
    Args:
        embeddings: List of embedding vectors
        max_clusters: Maximum number of clusters to test
    
    Returns:
        Estimated optimal number of clusters (more clusters = stricter grouping)
    """
    if len(embeddings) < 10:
        return min(len(embeddings), 8)  # More clusters for small datasets
    
    # Use a more aggressive approach for stricter clustering
    n = len(embeddings)
    # More clusters: n/10 or 100, whichever is smaller
    optimal = min(max(n // 10, 20), max_clusters)
    
    logger.info(f"🎯 Estimated optimal clusters: {optimal} (from {n} articles) - STRICT MODE")
    return optimal

def cluster_with_fast_incremental(embeddings: List[List[float]], existing_clusters: Optional[Dict] = None) -> Dict[int, List[int]]:
    """
    Use fast clustering for incremental clustering (assign new points to existing clusters)
    
    Args:
        embeddings: New embeddings to cluster
        existing_clusters: Existing cluster centroids (optional)
    
    Returns:
        Dict mapping cluster_id -> list of article indices
    """
    # For now, just use fast clustering for all cases
    # In the future, we could implement a more sophisticated incremental approach
    return cluster_articles_fast(embeddings)

def cluster_articles_complete(embeddings: List[List[float]], articles: List[Dict], 
                           num_clusters: Optional[int] = None, use_faiss: bool = False) -> Dict[str, Any]:
    """
    Complete clustering pipeline with summarization
    
    Args:
        embeddings: List of embedding vectors
        articles: List of article metadata dictionaries
        num_clusters: Number of clusters (auto-estimate if None)
        use_faiss: Whether to use FAISS for incremental clustering
    
    Returns:
        Complete clustering results with summaries
    """
    start = time.time()
    
    # Estimate optimal number of clusters if not provided
    if num_clusters is None:
        num_clusters = estimate_optimal_clusters(embeddings)
    
    # Perform clustering
    if use_faiss:
        clusters = cluster_with_fast_incremental(embeddings)
    else:
        clusters = cluster_articles_fast(embeddings, num_clusters)
    
    # Generate summaries
    cluster_summaries = summarize_clusters(articles, clusters)
    
    total_time = time.time() - start
    logger.info(f"🎉 Complete clustering pipeline finished in {total_time:.2f}s")
    
    return {
        "clusters": clusters,
        "summaries": cluster_summaries,
        "total_articles": len(embeddings),
        "total_clusters": len(clusters),
        "processing_time": total_time
    }

def generate_batch_summaries(cluster_texts: Dict[int, str], max_concurrent: int = 3) -> Dict[int, str]:
    """
    Generate summaries for multiple clusters in batches for better performance
    
    Args:
        cluster_texts: Dict mapping cluster_id -> combined text
        max_concurrent: Maximum concurrent summary generations
    
    Returns:
        Dict mapping cluster_id -> summary text
    """
    try:
        from services.summarizer import generate_summary
        import asyncio
        import concurrent.futures
        
        logger.info(f"📝 Generating batch summaries for {len(cluster_texts)} clusters")
        
        summaries = {}
        
        # Use ThreadPoolExecutor for concurrent summary generation
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_concurrent) as executor:
            # Submit all summary generation tasks
            future_to_cluster = {
                executor.submit(generate_summary, text): cluster_id 
                for cluster_id, text in cluster_texts.items()
            }
            
            # Collect results as they complete
            for future in concurrent.futures.as_completed(future_to_cluster):
                cluster_id = future_to_cluster[future]
                try:
                    summary = future.result()
                    summaries[cluster_id] = summary
                    logger.info(f"✅ Generated summary for cluster {cluster_id}: {summary[:50]}...")
                except Exception as e:
                    logger.error(f"❌ Failed to generate summary for cluster {cluster_id}: {e}")
                    summaries[cluster_id] = f"Summary generation failed: {str(e)}"
        
        logger.info(f"🎉 Batch summary generation complete: {len(summaries)} summaries")
        return summaries
        
    except Exception as e:
        logger.error(f"❌ Batch summary generation failed: {e}")
        # Fallback to empty summaries
        return {cluster_id: "Summary generation failed" for cluster_id in cluster_texts.keys()}

def cluster_articles_complete_with_summaries(embeddings: List[List[float]], articles: List[Dict], 
                                          num_clusters: Optional[int] = None, use_faiss: bool = False,
                                          max_concurrent_summaries: int = 3) -> Dict[str, Any]:
    """
    Complete clustering pipeline with fast batch summary generation
    
    Args:
        embeddings: List of embedding vectors
        articles: List of article metadata dictionaries
        num_clusters: Number of clusters (auto-estimate if None)
        use_faiss: Whether to use FAISS for incremental clustering
        max_concurrent_summaries: Maximum concurrent summary generations
    
    Returns:
        Complete clustering results with summaries
    """
    start = time.time()
    
    # Estimate optimal number of clusters if not provided
    if num_clusters is None:
        num_clusters = estimate_optimal_clusters(embeddings)
    
    # Perform clustering
    if use_faiss:
        clusters = cluster_with_fast_incremental(embeddings)
    else:
        clusters = cluster_articles_fast(embeddings, num_clusters)
    
    # Generate metadata summaries
    cluster_summaries = summarize_clusters(articles, clusters)
    
    # Prepare texts for batch summary generation
    cluster_texts = {}
    for cluster_id, cluster_indices in clusters.items():
        if len(cluster_indices) >= 3:  # Only generate summaries for valid clusters
            # Combine all article texts for this cluster
            texts = []
            for idx in cluster_indices:
                if idx < len(articles):
                    article = articles[idx]
                    if 'content' in article:
                        texts.append(article['content'])
            
            if texts:
                combined_text = '\n\n'.join(texts)
                cluster_texts[cluster_id] = combined_text
    
    # Generate batch summaries
    if cluster_texts:
        text_summaries = generate_batch_summaries(cluster_texts, max_concurrent_summaries)
        
        # Add text summaries to cluster summaries
        for cluster_id, text_summary in text_summaries.items():
            if cluster_id in cluster_summaries:
                cluster_summaries[cluster_id]['text_summary'] = text_summary
    
    total_time = time.time() - start
    logger.info(f"🎉 Complete clustering with batch summaries finished in {total_time:.2f}s")
    
    return {
        "clusters": clusters,
        "summaries": cluster_summaries,
        "total_articles": len(embeddings),
        "total_clusters": len(clusters),
        "processing_time": total_time
    } 