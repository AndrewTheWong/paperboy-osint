#!/usr/bin/env python3
"""
Pipeline for clustering articles based on their embeddings.
Uses DBSCAN to identify topic clusters.
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

# Try importing sklearn components, but continue with a stub if not available
try:
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    logger.error("Missing dependency: scikit-learn. Install with: pip install scikit-learn")
    logger.info("Without scikit-learn, clustering will not be performed.")
    HAS_SKLEARN = False

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
                    eps: float = 0.4,
                    min_samples: int = 3,
                    articles: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """
    Cluster articles based on their embeddings using DBSCAN.
    
    Args:
        input_path: Path to the embedded articles file (ignored if articles is provided)
        output_path: Path to save the clustered articles (set to None to skip saving)
        eps: DBSCAN epsilon parameter (max distance between points in same cluster)
        min_samples: DBSCAN min_samples parameter (min points to form a cluster)
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
    
    # Check for sklearn
    if not HAS_SKLEARN:
        logger.error("Scikit-learn not available. Cannot perform clustering.")
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
    
    # Scale the embeddings
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Track statistics
    start_time = time.time()
    
    try:
        # Apply DBSCAN
        logger.info(f"Clustering {len(X_scaled)} articles using DBSCAN (eps={eps}, min_samples={min_samples})")
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        labels = db.labels_
        
        # Count clusters
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        
        # Assign cluster IDs to articles
        for i, article in enumerate(articles_with_embeddings):
            article['cluster_id'] = int(labels[i])
        
        # Calculate stats
        total_time = time.time() - start_time
        
        logger.info(f"✅ Clustered {len(X_scaled)} articles into {n_clusters} clusters (excluding noise)")
        logger.info(f"Noise points: {n_noise} ({n_noise/len(X_scaled)*100:.1f}%)")
        logger.info(f"Total clustering time: {total_time:.2f} seconds")
        
        # Save clustered articles
        if output_path:
            save_clustered_articles(articles_with_embeddings, output_path)
        
        return articles_with_embeddings
    
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        return articles

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Cluster articles based on embeddings")
    parser.add_argument("--input", default="data/embedded_articles.json", help="Input file path")
    parser.add_argument("--output", default="data/clustered_articles.json", help="Output file path")
    parser.add_argument("--eps", type=float, default=0.4, help="DBSCAN epsilon parameter")
    parser.add_argument("--min-samples", type=int, default=3, help="DBSCAN min_samples parameter")
    
    args = parser.parse_args()
    
    # Cluster articles
    clustered_articles = cluster_articles(
        input_path=args.input,
        output_path=args.output,
        eps=args.eps,
        min_samples=args.min_samples
    )
    
    # Print summary
    if clustered_articles:
        cluster_ids = [article.get('cluster_id', -1) for article in clustered_articles]
        n_clusters = len(set(cluster_ids) - {-1})
        n_noise = cluster_ids.count(-1)
        
        print(f"✅ Clustered {len(clustered_articles)} articles into {n_clusters} clusters (excluding noise)")
        print(f"📊 Noise points: {n_noise} ({n_noise/len(clustered_articles)*100:.1f}%)") 