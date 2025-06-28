#!/usr/bin/env python3
"""
Similarity scoring module for articles.
Calculates similarity matrices and finds similar articles.
"""
import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.preprocessing import normalize
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('similarity_scoring')

def calculate_similarity_matrix(articles: List[Dict[str, Any]], 
                              metric: str = 'cosine',
                              normalize_embeddings: bool = True) -> np.ndarray:
    """
    Calculate similarity matrix between articles based on their embeddings.
    
    Args:
        articles: List of articles with embeddings
        metric: Similarity metric ('cosine', 'euclidean', 'dot_product')
        normalize_embeddings: Whether to normalize embeddings before calculation
        
    Returns:
        Similarity matrix as numpy array
    """
    logger.info(f"Calculating {metric} similarity matrix for {len(articles)} articles...")
    
    # Extract embeddings
    embeddings = []
    for article in articles:
        embedding = article.get('embedding')
        if embedding and len(embedding) > 0:
            embeddings.append(embedding)
        else:
            logger.warning(f"Article {article.get('id', 'unknown')} has no embedding")
            return np.array([])  # Return empty if any article lacks embedding
    
    if not embeddings:
        logger.error("No articles with embeddings found")
        return np.array([])
    
    # Convert to numpy array
    X = np.array(embeddings)
    
    # Normalize embeddings if requested
    if normalize_embeddings and metric in ['cosine', 'dot_product']:
        X = normalize(X, norm='l2')
    
    # Calculate similarity matrix based on metric
    if metric == 'cosine':
        similarity_matrix = cosine_similarity(X)
    elif metric == 'euclidean':
        # Convert distances to similarities (higher = more similar)
        distances = euclidean_distances(X)
        # Normalize distances to [0,1] and invert (1 - normalized_distance)
        max_distance = np.max(distances)
        if max_distance > 0:
            similarity_matrix = 1 - (distances / max_distance)
        else:
            similarity_matrix = np.ones_like(distances)
    elif metric == 'dot_product':
        similarity_matrix = np.dot(X, X.T)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'cosine', 'euclidean', or 'dot_product'")
    
    logger.info(f"Similarity matrix calculated with shape: {similarity_matrix.shape}")
    logger.info(f"Similarity range: [{np.min(similarity_matrix):.3f}, {np.max(similarity_matrix):.3f}]")
    
    return similarity_matrix

def find_similar_articles(target_article: Dict[str, Any],
                         articles: List[Dict[str, Any]],
                         top_k: int = 5,
                         metric: str = 'cosine',
                         exclude_self: bool = True) -> List[Tuple[Dict[str, Any], float]]:
    """
    Find the most similar articles to a target article.
    
    Args:
        target_article: The article to find similarities for
        articles: List of all articles to search in
        top_k: Number of most similar articles to return
        metric: Similarity metric to use
        exclude_self: Whether to exclude the target article from results
        
    Returns:
        List of tuples (article, similarity_score) sorted by similarity (highest first)
    """
    logger.info(f"Finding top {top_k} similar articles...")
    
    # Get target embedding
    target_embedding = target_article.get('embedding')
    if not target_embedding or len(target_embedding) == 0:
        logger.error("Target article has no embedding")
        return []
    
    # Calculate similarities
    similarities = []
    target_id = target_article.get('id')
    
    for article in articles:
        # Skip self if requested
        if exclude_self and article.get('id') == target_id:
            continue
            
        article_embedding = article.get('embedding')
        if not article_embedding or len(article_embedding) == 0:
            continue
        
        # Calculate similarity
        if metric == 'cosine':
            similarity = cosine_similarity([target_embedding], [article_embedding])[0][0]
        elif metric == 'euclidean':
            distance = euclidean_distances([target_embedding], [article_embedding])[0][0]
            # Convert to similarity (higher = more similar)
            similarity = 1 / (1 + distance)
        elif metric == 'dot_product':
            # Normalize for dot product
            norm_target = np.array(target_embedding) / np.linalg.norm(target_embedding)
            norm_article = np.array(article_embedding) / np.linalg.norm(article_embedding)
            similarity = np.dot(norm_target, norm_article)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        similarities.append((article, float(similarity)))
    
    # Sort by similarity (highest first) and return top k
    similarities.sort(key=lambda x: x[1], reverse=True)
    result = similarities[:top_k]
    
    logger.info(f"Found {len(result)} similar articles")
    if result:
        logger.info(f"Similarity range: [{result[-1][1]:.3f}, {result[0][1]:.3f}]")
    
    return result

def get_cluster_similarity_stats(articles: List[Dict[str, Any]], 
                                cluster_id: int,
                                metric: str = 'cosine') -> Dict[str, Any]:
    """
    Calculate similarity statistics within a specific cluster.
    
    Args:
        articles: List of articles with cluster assignments
        cluster_id: ID of the cluster to analyze
        metric: Similarity metric to use
        
    Returns:
        Dictionary with cluster similarity statistics
    """
    logger.info(f"Calculating similarity stats for cluster {cluster_id}...")
    
    # Filter articles in the cluster
    cluster_articles = [a for a in articles if a.get('cluster_id') == cluster_id]
    
    if len(cluster_articles) < 2:
        logger.warning(f"Cluster {cluster_id} has fewer than 2 articles")
        return {
            'cluster_id': cluster_id,
            'size': len(cluster_articles),
            'error': 'insufficient_articles'
        }
    
    # Calculate similarity matrix for cluster
    similarity_matrix = calculate_similarity_matrix(cluster_articles, metric=metric)
    
    if similarity_matrix.size == 0:
        return {
            'cluster_id': cluster_id,
            'size': len(cluster_articles),
            'error': 'no_embeddings'
        }
    
    # Extract upper triangle (excluding diagonal) for statistics
    mask = np.triu(np.ones_like(similarity_matrix), k=1).astype(bool)
    similarities = similarity_matrix[mask]
    
    stats = {
        'cluster_id': cluster_id,
        'size': len(cluster_articles),
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'median_similarity': float(np.median(similarities)),
        'num_pairs': len(similarities)
    }
    
    logger.info(f"Cluster {cluster_id} similarity stats: mean={stats['mean_similarity']:.3f}, "
               f"std={stats['std_similarity']:.3f}")
    
    return stats

def compare_clusters_similarity(articles: List[Dict[str, Any]], 
                              cluster_id1: int, 
                              cluster_id2: int,
                              metric: str = 'cosine',
                              sample_size: Optional[int] = None) -> Dict[str, Any]:
    """
    Compare similarity between articles from two different clusters.
    
    Args:
        articles: List of articles with cluster assignments
        cluster_id1: First cluster ID
        cluster_id2: Second cluster ID  
        metric: Similarity metric to use
        sample_size: Optional limit on number of articles per cluster for comparison
        
    Returns:
        Dictionary with inter-cluster similarity statistics
    """
    logger.info(f"Comparing similarity between clusters {cluster_id1} and {cluster_id2}...")
    
    # Filter articles by cluster
    cluster1_articles = [a for a in articles if a.get('cluster_id') == cluster_id1]
    cluster2_articles = [a for a in articles if a.get('cluster_id') == cluster_id2]
    
    # Sample if requested
    if sample_size:
        if len(cluster1_articles) > sample_size:
            cluster1_articles = np.random.choice(cluster1_articles, sample_size, replace=False).tolist()
        if len(cluster2_articles) > sample_size:
            cluster2_articles = np.random.choice(cluster2_articles, sample_size, replace=False).tolist()
    
    if not cluster1_articles or not cluster2_articles:
        return {
            'cluster1_id': cluster_id1,
            'cluster2_id': cluster_id2,
            'error': 'empty_clusters'
        }
    
    # Calculate cross-cluster similarities
    similarities = []
    for article1 in cluster1_articles:
        for article2 in cluster2_articles:
            embedding1 = article1.get('embedding')
            embedding2 = article2.get('embedding')
            
            if not embedding1 or not embedding2:
                continue
                
            if metric == 'cosine':
                similarity = cosine_similarity([embedding1], [embedding2])[0][0]
            elif metric == 'euclidean':
                distance = euclidean_distances([embedding1], [embedding2])[0][0]
                similarity = 1 / (1 + distance)
            elif metric == 'dot_product':
                norm1 = np.array(embedding1) / np.linalg.norm(embedding1)
                norm2 = np.array(embedding2) / np.linalg.norm(embedding2)
                similarity = np.dot(norm1, norm2)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
            
            similarities.append(float(similarity))
    
    if not similarities:
        return {
            'cluster1_id': cluster_id1,
            'cluster2_id': cluster_id2,
            'error': 'no_embeddings'
        }
    
    similarities = np.array(similarities)
    
    stats = {
        'cluster1_id': cluster_id1,
        'cluster2_id': cluster_id2,
        'cluster1_size': len(cluster1_articles),
        'cluster2_size': len(cluster2_articles),
        'mean_similarity': float(np.mean(similarities)),
        'std_similarity': float(np.std(similarities)),
        'min_similarity': float(np.min(similarities)),
        'max_similarity': float(np.max(similarities)),
        'median_similarity': float(np.median(similarities)),
        'num_comparisons': len(similarities)
    }
    
    logger.info(f"Inter-cluster similarity ({cluster_id1} vs {cluster_id2}): "
               f"mean={stats['mean_similarity']:.3f}")
    
    return stats

def create_similarity_report(articles: List[Dict[str, Any]], 
                           output_file: Optional[str] = None) -> Dict[str, Any]:
    """
    Create a comprehensive similarity analysis report.
    
    Args:
        articles: List of articles with embeddings and cluster assignments
        output_file: Optional file to save the report
        
    Returns:
        Dictionary with comprehensive similarity analysis
    """
    logger.info("Creating comprehensive similarity report...")
    
    # Get unique cluster IDs
    cluster_ids = list(set(a.get('cluster_id', -1) for a in articles if a.get('cluster_id') != -1))
    cluster_ids.sort()
    
    report = {
        'total_articles': len(articles),
        'total_clusters': len(cluster_ids),
        'cluster_stats': {},
        'inter_cluster_stats': {},
        'overall_stats': {}
    }
    
    # Calculate intra-cluster similarities
    for cluster_id in cluster_ids:
        stats = get_cluster_similarity_stats(articles, cluster_id)
        report['cluster_stats'][cluster_id] = stats
    
    # Calculate inter-cluster similarities (sample for performance)
    max_clusters_to_compare = 10
    if len(cluster_ids) > max_clusters_to_compare:
        compared_clusters = np.random.choice(cluster_ids, max_clusters_to_compare, replace=False)
    else:
        compared_clusters = cluster_ids
    
    for i, cluster1 in enumerate(compared_clusters):
        for cluster2 in compared_clusters[i+1:]:
            key = f"{cluster1}_vs_{cluster2}"
            stats = compare_clusters_similarity(articles, cluster1, cluster2, sample_size=50)
            report['inter_cluster_stats'][key] = stats
    
    # Calculate overall statistics
    if report['cluster_stats']:
        intra_similarities = [stats.get('mean_similarity', 0) 
                            for stats in report['cluster_stats'].values() 
                            if 'mean_similarity' in stats]
        if intra_similarities:
            report['overall_stats']['mean_intra_cluster_similarity'] = float(np.mean(intra_similarities))
            report['overall_stats']['std_intra_cluster_similarity'] = float(np.std(intra_similarities))
    
    if report['inter_cluster_stats']:
        inter_similarities = [stats.get('mean_similarity', 0) 
                            for stats in report['inter_cluster_stats'].values() 
                            if 'mean_similarity' in stats]
        if inter_similarities:
            report['overall_stats']['mean_inter_cluster_similarity'] = float(np.mean(inter_similarities))
            report['overall_stats']['std_inter_cluster_similarity'] = float(np.std(inter_similarities))
    
    # Save report if requested
    if output_file:
        import json
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"Similarity report saved to {output_file}")
    
    logger.info("Similarity report completed")
    return report

if __name__ == "__main__":
    # Example usage
    sample_articles = [
        {'id': 1, 'embedding': np.random.rand(384).tolist(), 'cluster_id': 0},
        {'id': 2, 'embedding': np.random.rand(384).tolist(), 'cluster_id': 0},
        {'id': 3, 'embedding': np.random.rand(384).tolist(), 'cluster_id': 1},
    ]
    
    # Test similarity matrix calculation
    sim_matrix = calculate_similarity_matrix(sample_articles)
    print(f"Similarity matrix shape: {sim_matrix.shape}")
    
    # Test finding similar articles
    similar = find_similar_articles(sample_articles[0], sample_articles, top_k=2)
    print(f"Found {len(similar)} similar articles")