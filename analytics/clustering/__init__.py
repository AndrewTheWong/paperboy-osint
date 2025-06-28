"""
Clustering package for article analysis.
Provides unified clustering with NER integration and comprehensive analysis.
"""

from .unified_clustering import cluster_articles
from .similarity_scoring import calculate_similarity_matrix, find_similar_articles
from .visualization import create_cluster_visualization, plot_cluster_metrics
from .metadata_extraction import extract_cluster_metadata, get_cluster_keywords

__all__ = [
    'cluster_articles',
    'calculate_similarity_matrix',
    'find_similar_articles',
    'create_cluster_visualization',
    'plot_cluster_metrics',
    'extract_cluster_metadata',
    'get_cluster_keywords'
] 