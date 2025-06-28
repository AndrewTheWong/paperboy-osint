#!/usr/bin/env python3
"""
Visualization module for clustering analysis.
Creates cluster visualizations, plots, and interactive charts.
"""
import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('visualization')

# Set up matplotlib style
plt.style.use('default')
sns.set_palette("husl")

def create_cluster_visualization(articles: List[Dict[str, Any]], 
                               method: str = 'tsne',
                               output_file: Optional[str] = None,
                               interactive: bool = True,
                               figsize: Tuple[int, int] = (12, 8)) -> None:
    """
    Create a 2D visualization of article clusters.
    
    Args:
        articles: List of articles with embeddings and cluster assignments
        method: Dimensionality reduction method ('tsne', 'pca', 'umap')
        output_file: Optional file to save the plot
        interactive: Whether to create interactive plot (Plotly) or static (Matplotlib)
        figsize: Figure size for matplotlib plots
    """
    logger.info(f"Creating {method.upper()} cluster visualization...")
    
    # Extract embeddings and cluster IDs
    embeddings = []
    cluster_ids = []
    titles = []
    
    for article in articles:
        embedding = article.get('embedding')
        if embedding and len(embedding) > 0:
            embeddings.append(embedding)
            cluster_ids.append(article.get('cluster_id', -1))
            titles.append(article.get('title', f"Article {article.get('id', 'unknown')}"))
    
    if not embeddings:
        logger.error("No articles with embeddings found")
        return
    
    X = np.array(embeddings)
    cluster_ids = np.array(cluster_ids)
    
    # Apply dimensionality reduction
    if method.lower() == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
        X_2d = reducer.fit_transform(X)
    elif method.lower() == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
    elif method.lower() == 'umap':
        try:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_2d = reducer.fit_transform(X)
        except ImportError:
            logger.warning("UMAP not available, falling back to t-SNE")
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)-1))
            X_2d = reducer.fit_transform(X)
    else:
        raise ValueError(f"Unsupported method: {method}. Use 'tsne', 'pca', or 'umap'")
    
    # Create visualization
    if interactive:
        _create_interactive_plot(X_2d, cluster_ids, titles, method, output_file)
    else:
        _create_static_plot(X_2d, cluster_ids, method, figsize, output_file)
    
    logger.info(f"Cluster visualization completed using {method.upper()}")

def _create_interactive_plot(X_2d: np.ndarray, 
                           cluster_ids: np.ndarray, 
                           titles: List[str],
                           method: str,
                           output_file: Optional[str] = None) -> None:
    """Create interactive Plotly visualization."""
    # Create DataFrame for easier handling
    df = pd.DataFrame({
        'x': X_2d[:, 0],
        'y': X_2d[:, 1],
        'cluster': cluster_ids,
        'title': titles
    })
    
    # Create color map for clusters
    unique_clusters = sorted(df['cluster'].unique())
    colors = px.colors.qualitative.Set3[:len(unique_clusters)]
    
    # Create scatter plot
    fig = px.scatter(
        df, 
        x='x', 
        y='y', 
        color='cluster',
        hover_data=['title'],
        title=f'Article Clusters - {method.upper()} Visualization',
        labels={'x': f'{method.upper()} Component 1', 'y': f'{method.upper()} Component 2'},
        width=800,
        height=600
    )
    
    # Update layout
    fig.update_layout(
        title_font_size=16,
        showlegend=True,
        legend=dict(
            title="Cluster ID",
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.01
        )
    )
    
    # Update markers
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        hovertemplate='<b>%{hovertext}</b><br>' +
                     f'{method.upper()} 1: %{{x:.2f}}<br>' +
                     f'{method.upper()} 2: %{{y:.2f}}<br>' +
                     'Cluster: %{marker.color}<br>' +
                     '<extra></extra>',
        hovertext=df['title']
    )
    
    # Save if requested
    if output_file:
        if output_file.endswith('.html'):
            fig.write_html(output_file)
        else:
            fig.write_image(output_file)
        logger.info(f"Interactive plot saved to {output_file}")
    else:
        fig.show()

def _create_static_plot(X_2d: np.ndarray, 
                       cluster_ids: np.ndarray,
                       method: str,
                       figsize: Tuple[int, int],
                       output_file: Optional[str] = None) -> None:
    """Create static matplotlib visualization."""
    plt.figure(figsize=figsize)
    
    # Get unique clusters for coloring
    unique_clusters = np.unique(cluster_ids)
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_clusters)))
    
    # Plot each cluster
    for i, cluster_id in enumerate(unique_clusters):
        mask = cluster_ids == cluster_id
        label = f'Cluster {cluster_id}' if cluster_id != -1 else 'Noise'
        plt.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                   c=[colors[i]], label=label, alpha=0.7, s=50)
    
    plt.xlabel(f'{method.upper()} Component 1')
    plt.ylabel(f'{method.upper()} Component 2')
    plt.title(f'Article Clusters - {method.upper()} Visualization')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Static plot saved to {output_file}")
    else:
        plt.show()

def plot_cluster_metrics(articles: List[Dict[str, Any]], 
                        output_file: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> None:
    """
    Plot various cluster quality metrics.
    
    Args:
        articles: List of articles with cluster assignments
        output_file: Optional file to save the plot
        figsize: Figure size
    """
    logger.info("Creating cluster metrics visualization...")
    
    # Extract cluster information
    cluster_data = []
    for article in articles:
        cluster_id = article.get('cluster_id', -1)
        cluster_prob = article.get('cluster_probability', 0.0)
        tags = article.get('tags', [])
        cluster_data.append({
            'cluster_id': cluster_id,
            'cluster_probability': cluster_prob,
            'num_tags': len(tags) if isinstance(tags, list) else 0
        })
    
    df = pd.DataFrame(cluster_data)
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle('Cluster Analysis Metrics', fontsize=16)
    
    # 1. Cluster size distribution
    cluster_sizes = df[df['cluster_id'] != -1]['cluster_id'].value_counts().sort_index()
    axes[0, 0].bar(cluster_sizes.index, cluster_sizes.values)
    axes[0, 0].set_xlabel('Cluster ID')
    axes[0, 0].set_ylabel('Number of Articles')
    axes[0, 0].set_title('Cluster Size Distribution')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Cluster probability distribution
    non_noise = df[df['cluster_id'] != -1]
    if not non_noise.empty and 'cluster_probability' in non_noise.columns:
        axes[0, 1].hist(non_noise['cluster_probability'], bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Cluster Probability')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Cluster Probability Distribution')
        axes[0, 1].grid(True, alpha=0.3)
    else:
        axes[0, 1].text(0.5, 0.5, 'No probability data available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Cluster Probability Distribution')
    
    # 3. Tags per cluster
    if not non_noise.empty:
        tag_stats = non_noise.groupby('cluster_id')['num_tags'].mean()
        axes[1, 0].bar(tag_stats.index, tag_stats.values)
        axes[1, 0].set_xlabel('Cluster ID')
        axes[1, 0].set_ylabel('Average Number of Tags')
        axes[1, 0].set_title('Average Tags per Cluster')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].text(0.5, 0.5, 'No cluster data available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Average Tags per Cluster')
    
    # 4. Noise vs Clustered articles
    noise_count = len(df[df['cluster_id'] == -1])
    clustered_count = len(df[df['cluster_id'] != -1])
    
    axes[1, 1].pie([clustered_count, noise_count], 
                   labels=['Clustered', 'Noise'], 
                   autopct='%1.1f%%',
                   startangle=90)
    axes[1, 1].set_title('Clustered vs Noise Articles')
    
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Cluster metrics plot saved to {output_file}")
    else:
        plt.show()

def create_similarity_heatmap(articles: List[Dict[str, Any]], 
                            max_articles: int = 50,
                            output_file: Optional[str] = None,
                            figsize: Tuple[int, int] = (12, 10)) -> None:
    """
    Create a similarity heatmap for a subset of articles.
    
    Args:
        articles: List of articles with embeddings
        max_articles: Maximum number of articles to include (for performance)
        output_file: Optional file to save the plot
        figsize: Figure size
    """
    logger.info(f"Creating similarity heatmap for up to {max_articles} articles...")
    
    # Sample articles if needed
    if len(articles) > max_articles:
        sampled_articles = np.random.choice(articles, max_articles, replace=False).tolist()
    else:
        sampled_articles = articles
    
    # Calculate similarity matrix
    from .similarity_scoring import calculate_similarity_matrix
    similarity_matrix = calculate_similarity_matrix(sampled_articles)
    
    if similarity_matrix.size == 0:
        logger.error("No similarity matrix could be calculated")
        return
    
    # Create labels
    labels = [f"Art_{i+1}" for i in range(len(sampled_articles))]
    
    # Create heatmap
    plt.figure(figsize=figsize)
    sns.heatmap(similarity_matrix, 
                xticklabels=labels, 
                yticklabels=labels,
                cmap='viridis', 
                center=0,
                square=True,
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title(f'Article Similarity Heatmap ({len(sampled_articles)} articles)')
    plt.xlabel('Articles')
    plt.ylabel('Articles')
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Similarity heatmap saved to {output_file}")
    else:
        plt.show()

def create_cluster_evolution_plot(articles_by_time: Dict[str, List[Dict[str, Any]]], 
                                output_file: Optional[str] = None,
                                figsize: Tuple[int, int] = (15, 8)) -> None:
    """
    Create a plot showing how clusters evolve over time.
    
    Args:
        articles_by_time: Dictionary with timestamps as keys and article lists as values
        output_file: Optional file to save the plot
        figsize: Figure size
    """
    logger.info("Creating cluster evolution plot...")
    
    # Prepare data
    time_points = sorted(articles_by_time.keys())
    cluster_counts = []
    noise_percentages = []
    
    for time_point in time_points:
        articles = articles_by_time[time_point]
        cluster_ids = [a.get('cluster_id', -1) for a in articles]
        
        n_clusters = len(set(cluster_ids) - {-1})
        noise_count = cluster_ids.count(-1)
        noise_pct = (noise_count / len(cluster_ids) * 100) if cluster_ids else 0
        
        cluster_counts.append(n_clusters)
        noise_percentages.append(noise_pct)
    
    # Create plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # Plot number of clusters over time
    ax1.plot(time_points, cluster_counts, marker='o', linewidth=2, markersize=6)
    ax1.set_ylabel('Number of Clusters')
    ax1.set_title('Cluster Evolution Over Time')
    ax1.grid(True, alpha=0.3)
    
    # Plot noise percentage over time
    ax2.plot(time_points, noise_percentages, marker='s', color='red', 
             linewidth=2, markersize=6)
    ax2.set_ylabel('Noise Percentage (%)')
    ax2.set_xlabel('Time')
    ax2.set_title('Noise Level Over Time')
    ax2.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save if requested
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        logger.info(f"Cluster evolution plot saved to {output_file}")
    else:
        plt.show()

def save_visualization_report(articles: List[Dict[str, Any]], 
                            output_dir: str = "analytics/reports") -> None:
    """
    Generate and save a comprehensive visualization report.
    
    Args:
        articles: List of articles with clustering information
        output_dir: Directory to save the report files
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"Generating visualization report in {output_dir}...")
    
    # Generate all visualizations
    create_cluster_visualization(
        articles, 
        method='tsne', 
        output_file=f"{output_dir}/cluster_tsne.html",
        interactive=True
    )
    
    plot_cluster_metrics(
        articles, 
        output_file=f"{output_dir}/cluster_metrics.png"
    )
    
    create_similarity_heatmap(
        articles, 
        max_articles=30,
        output_file=f"{output_dir}/similarity_heatmap.png"
    )
    
    # Create summary statistics
    cluster_ids = [a.get('cluster_id', -1) for a in articles]
    n_clusters = len(set(cluster_ids) - {-1})
    n_noise = cluster_ids.count(-1)
    
    summary = {
        'total_articles': len(articles),
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'noise_percentage': (n_noise / len(articles) * 100) if articles else 0,
        'largest_cluster_size': max([cluster_ids.count(cid) for cid in set(cluster_ids) if cid != -1], default=0),
        'visualization_files': [
            'cluster_tsne.html',
            'cluster_metrics.png',
            'similarity_heatmap.png'
        ]
    }
    
    # Save summary
    with open(f"{output_dir}/visualization_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Visualization report completed. Files saved in {output_dir}/")

if __name__ == "__main__":
    # Example usage with sample data
    sample_articles = [
        {
            'id': i,
            'embedding': np.random.rand(384).tolist(),
            'cluster_id': np.random.randint(0, 3),
            'title': f'Sample Article {i}',
            'tags': ['tag1', 'tag2']
        }
        for i in range(50)
    ]
    
    # Test visualization
    create_cluster_visualization(sample_articles, method='pca', interactive=False)
    plot_cluster_metrics(sample_articles)