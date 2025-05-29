#!/usr/bin/env python3
"""
Pipeline for clustering articles based on their embeddings.
Uses HDBSCAN to identify topic clusters with improved robustness.
"""
import json
import logging
import os
import time
from typing import List, Dict, Any, Union, Optional
from pathlib import Path
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('clustering_pipeline')

# Try importing clustering components
try:
    import hdbscan
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    HAS_CLUSTERING = True
except ImportError:
    logger.error("Missing dependency: hdbscan or scikit-learn. Install with: pip install hdbscan scikit-learn")
    logger.info("Without clustering libraries, clustering will not be performed.")
    HAS_CLUSTERING = False

def load_articles(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        input_path: Path to the articles JSON file
        
    Returns:
        List of article dictionaries
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return []
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        logger.info(f"Loaded {len(articles)} articles from {input_path}")
        return articles
    except Exception as e:
        logger.error(f"Error loading articles: {str(e)}")
        return []

def save_clustered_articles(articles: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Save clustered articles to a JSON file.
    
    Args:
        articles: List of article dictionaries with cluster assignments
        output_path: Path to save the clustered articles
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    # Calculate file size in MB
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(articles)} clustered articles to {output_path} ({file_size_mb:.2f} MB)")

def cluster_articles(input_path: Union[str, Path] = "data/embedded_articles.json", 
                    output_path: Union[str, Path] = "data/clustered_articles.json",
                    min_cluster_size: int = 5,
                    min_samples: int = 3,
                    cluster_selection_epsilon: float = 0.0,
                    articles: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Cluster articles based on their embeddings using HDBSCAN.
    
    Args:
        input_path: Path to the embedded articles file (ignored if articles is provided)
        output_path: Path to save the clustered articles (set to None to skip saving)
        min_cluster_size: Minimum size of clusters (HDBSCAN parameter)
        min_samples: Minimum samples in a neighborhood for a point to be core (HDBSCAN parameter)
        cluster_selection_epsilon: Distance threshold for cluster selection (0.0 = use HDBSCAN default)
        articles: Optional list of articles with embeddings (if provided, input_path is ignored)
        
    Returns:
        List of articles with cluster assignments
    """
    # Load articles if not provided
    if articles is None:
        articles = load_articles(input_path)
    else:
        logger.info(f"Using {len(articles)} provided articles")
    
    if not articles:
        logger.error("No articles loaded. Exiting.")
        return []
    
    # Check for clustering libraries
    if not HAS_CLUSTERING:
        logger.error("Clustering libraries not available. Cannot perform clustering.")
        return articles
    
    # Extract embeddings from articles
    embeddings = []
    articles_with_embeddings = []
    
    for article in articles:
        if article.get('embedding') and len(article.get('embedding', [])) > 0:
            embeddings.append(article['embedding'])
            articles_with_embeddings.append(article)
    
    if not embeddings:
        logger.error("No articles with embeddings found. Cannot perform clustering.")
        return articles
    
    # Convert to numpy array
    X = np.array(embeddings)
    
    # Scale the embeddings (HDBSCAN works better with normalized data)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Track statistics
    start_time = time.time()
    
    try:
        # Configure HDBSCAN parameters
        hdbscan_params = {
            'min_cluster_size': min_cluster_size,
            'min_samples': min_samples,
            'metric': 'euclidean',
            'cluster_selection_method': 'eom'  # Excess of Mass
        }
        
        # Add cluster selection epsilon if specified
        if cluster_selection_epsilon > 0:
            hdbscan_params['cluster_selection_epsilon'] = cluster_selection_epsilon
        
        # Apply HDBSCAN
        logger.info(f"Clustering {len(X_scaled)} articles using HDBSCAN (min_cluster_size={min_cluster_size}, min_samples={min_samples})")
        clusterer = hdbscan.HDBSCAN(**hdbscan_params)
        cluster_labels = clusterer.fit_predict(X_scaled)
        
        # Count clusters and noise
        unique_labels = set(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        # Calculate cluster statistics
        cluster_sizes = {}
        for label in cluster_labels:
            if label != -1:  # Exclude noise
                cluster_sizes[label] = cluster_sizes.get(label, 0) + 1
        
        # Assign cluster IDs to articles
        for i, article in enumerate(articles_with_embeddings):
            article['cluster_id'] = int(cluster_labels[i])
            
            # Add clustering confidence if available
            if hasattr(clusterer, 'probabilities_'):
                article['cluster_probability'] = float(clusterer.probabilities_[i])
        
        # Calculate clustering quality metrics
        if n_clusters > 1:
            # Filter out noise points for silhouette score
            non_noise_mask = cluster_labels != -1
            if np.sum(non_noise_mask) > 1:
                silhouette_avg = silhouette_score(X_scaled[non_noise_mask], cluster_labels[non_noise_mask])
            else:
                silhouette_avg = 0.0
        else:
            silhouette_avg = 0.0
        
        # Calculate total time
        total_time = time.time() - start_time
        
        # Log clustering results
        logger.info(f"✅ Clustered {len(X_scaled)} articles into {n_clusters} clusters (excluding noise)")
        logger.info(f"📊 Noise points: {n_noise} ({n_noise/len(X_scaled)*100:.1f}%)")
        logger.info(f"📈 Silhouette score: {silhouette_avg:.3f}")
        logger.info(f"⏱️ Total clustering time: {total_time:.2f} seconds")
        
        # Log cluster size distribution
        if cluster_sizes:
            sizes = list(cluster_sizes.values())
            logger.info(f"🔢 Cluster sizes - Min: {min(sizes)}, Max: {max(sizes)}, Avg: {np.mean(sizes):.1f}")
            
            # Show top 5 largest clusters
            sorted_clusters = sorted(cluster_sizes.items(), key=lambda x: x[1], reverse=True)[:5]
            logger.info(f"🏆 Top clusters: {sorted_clusters}")
        
        # Add clustering metadata to articles
        clustering_metadata = {
            'clustering_algorithm': 'HDBSCAN',
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'noise_ratio': n_noise / len(X_scaled),
            'silhouette_score': silhouette_avg,
            'clustering_time': total_time,
            'parameters': hdbscan_params,
            'cluster_sizes': cluster_sizes
        }
        
        # Add metadata to each article
        for article in articles_with_embeddings:
            article['clustering_metadata'] = clustering_metadata
        
        # Save clustered articles
        if output_path:
            save_clustered_articles(articles_with_embeddings, output_path)
        
        return articles_with_embeddings
    
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        return articles

def analyze_clusters(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze clustering results and provide insights.
    
    Args:
        articles: List of clustered articles
        
    Returns:
        Dictionary with cluster analysis
    """
    if not articles:
        return {}
    
    # Group articles by cluster
    clusters = {}
    noise_articles = []
    
    for article in articles:
        cluster_id = article.get('cluster_id', -1)
        if cluster_id == -1:
            noise_articles.append(article)
        else:
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            clusters[cluster_id].append(article)
    
    # Analyze each cluster
    cluster_analysis = {}
    for cluster_id, cluster_articles in clusters.items():
        # Extract common tags
        all_tags = []
        for article in cluster_articles:
            tags = article.get('tags', [])
            all_tags.extend(tags)
        
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Sort tags by frequency
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Calculate average escalation score
        escalation_scores = [article.get('escalation_score', 0) for article in cluster_articles]
        avg_escalation = np.mean(escalation_scores) if escalation_scores else 0
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_articles),
            'top_tags': top_tags,
            'avg_escalation_score': avg_escalation,
            'sample_titles': [article.get('title', '')[:100] for article in cluster_articles[:3]]
        }
    
    analysis = {
        'n_clusters': len(clusters),
        'n_noise': len(noise_articles),
        'cluster_details': cluster_analysis,
        'largest_cluster_size': max([len(articles) for articles in clusters.values()]) if clusters else 0,
        'smallest_cluster_size': min([len(articles) for articles in clusters.values()]) if clusters else 0
    }
    
    return analysis

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cluster articles based on embeddings using HDBSCAN")
    parser.add_argument("--input", default="data/embedded_articles.json", help="Input file path")
    parser.add_argument("--output", default="data/clustered_articles.json", help="Output file path")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="Minimum cluster size")
    parser.add_argument("--min-samples", type=int, default=3, help="Minimum samples parameter")
    parser.add_argument("--cluster-epsilon", type=float, default=0.0, help="Cluster selection epsilon")
    parser.add_argument("--analyze", action="store_true", help="Perform cluster analysis")
    
    args = parser.parse_args()
    
    # Cluster articles
    clustered_articles = cluster_articles(
        input_path=args.input,
        output_path=args.output,
        min_cluster_size=args.min_cluster_size,
        min_samples=args.min_samples,
        cluster_selection_epsilon=args.cluster_epsilon
    )
    
    # Print summary
    if clustered_articles:
        cluster_ids = [article.get('cluster_id', -1) for article in clustered_articles]
        n_clusters = len(set(cluster_ids) - {-1})
        n_noise = cluster_ids.count(-1)
        
        print(f"✅ Clustered {len(clustered_articles)} articles into {n_clusters} clusters (excluding noise)")
        print(f"📊 Noise points: {n_noise} ({n_noise/len(clustered_articles)*100:.1f}%)")
        
        # Perform detailed analysis if requested
        if args.analyze:
            analysis = analyze_clusters(clustered_articles)
            print(f"\n📈 Cluster Analysis:")
            print(f"  Largest cluster: {analysis['largest_cluster_size']} articles")
            print(f"  Smallest cluster: {analysis['smallest_cluster_size']} articles")
            
            # Show top 3 clusters
            sorted_clusters = sorted(analysis['cluster_details'].items(), 
                                   key=lambda x: x[1]['size'], reverse=True)[:3]
            for cluster_id, details in sorted_clusters:
                print(f"\n  🔸 Cluster {cluster_id} ({details['size']} articles):")
                print(f"    Top tags: {[tag for tag, count in details['top_tags']]}")
                print(f"    Avg escalation: {details['avg_escalation_score']:.3f}")
                if details['sample_titles']:
                    print(f"    Sample: {details['sample_titles'][0]}")