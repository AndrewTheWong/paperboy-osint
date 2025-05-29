#!/usr/bin/env python3
"""
Unified Backend Pipeline for StraitWatch Phase 1

Orchestrates the complete ML pipeline:
1. Load new articles from Supabase or data/translated_articles.json
2. Tag articles with tagging_pipeline (keyword + ML tags)
3. Run inference_pipeline on each article → add escalation_score
4. Embed using SBERT → save to data/embedded_articles.json
5. Run clustering on embedded articles
6. Forecast tomorrow's escalation score using XGBoost
7. Save final result to data/clustered_articles.json

Usage:
    python run_backend_pipeline.py
    python run_backend_pipeline.py --input-file data/my_articles.json
    python run_backend_pipeline.py --forecast-only  # Only run forecasting
"""
import os
import json
import logging
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backend_pipeline')

# Create directories
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)


def load_articles(input_path: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Load articles from specified file or default locations.
    
    Args:
        input_path: Optional path to articles file
        
    Returns:
        List of article dictionaries
    """
    # Try input_path first if provided
    if input_path and os.path.exists(input_path):
        logger.info(f"Loading articles from: {input_path}")
        with open(input_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {input_path}")
        return articles
    
    # Try default locations
    default_paths = [
        "data/translated_articles.json",
        "data/articles.json",
        "data/processed_articles.json"
    ]
    
    for path in default_paths:
        if os.path.exists(path):
            logger.info(f"Loading articles from: {path}")
            with open(path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            logger.info(f"Loaded {len(articles)} articles from {path}")
            return articles
    
    logger.warning("No article files found in default locations")
    return []


def run_tagging_pipeline(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run tagging pipeline on articles.
    
    Args:
        articles: List of articles to tag
        
    Returns:
        Tagged articles
    """
    logger.info("Running tagging pipeline...")
    
    try:
        from tagging.tagging_pipeline import tag_articles
        
        tagged_articles = tag_articles(articles)
        
        # Log tagging results
        total_tags = sum(len(article.get('tags', [])) for article in tagged_articles)
        articles_with_tags = sum(1 for article in tagged_articles if article.get('tags'))
        review_needed = sum(1 for article in tagged_articles if article.get('needs_review', False))
        
        logger.info(f"Tagging completed:")
        logger.info(f"  Articles with tags: {articles_with_tags}/{len(tagged_articles)}")
        logger.info(f"  Total tags assigned: {total_tags}")
        logger.info(f"  Articles needing review: {review_needed}")
        
        # Save tagged articles
        with open("data/tagged_articles.json", 'w', encoding='utf-8') as f:
            json.dump(tagged_articles, f, ensure_ascii=False, indent=2)
        
        return tagged_articles
        
    except Exception as e:
        logger.error(f"Error in tagging pipeline: {e}")
        # Return original articles if tagging fails
        return articles


def run_inference_pipeline(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run escalation inference on articles.
    
    Args:
        articles: List of tagged articles
        
    Returns:
        Articles with escalation scores
    """
    logger.info("Running escalation inference pipeline...")
    
    try:
        from pipelines.inference_pipeline import predict_escalation
        
        scored_articles = []
        scores = []
        
        for article in articles:
            # Combine title and text for better context
            text = ""
            if 'translated_text' in article and article['translated_text']:
                text = f"{article.get('title', '')} {article['translated_text']}"
            elif 'title' in article:
                text = article['title']
            
            # Get escalation score
            escalation_score = predict_escalation(text) if text else 0.0
            
            # Add score to article
            article['escalation_score'] = escalation_score
            scores.append(escalation_score)
            scored_articles.append(article)
        
        # Log inference results
        avg_score = np.mean(scores) if scores else 0.0
        high_risk_count = sum(1 for score in scores if score > 0.7)
        
        logger.info(f"Escalation inference completed:")
        logger.info(f"  Average escalation score: {avg_score:.3f}")
        logger.info(f"  High-risk articles (>0.7): {high_risk_count}/{len(scored_articles)}")
        
        # Save scored articles
        with open("data/scored_articles.json", 'w', encoding='utf-8') as f:
            json.dump(scored_articles, f, ensure_ascii=False, indent=2)
        
        return scored_articles
        
    except Exception as e:
        logger.error(f"Error in inference pipeline: {e}")
        # Add default scores if inference fails
        for article in articles:
            article['escalation_score'] = 0.5  # Neutral default
        return articles


def run_embedding_pipeline(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate SBERT embeddings for articles.
    
    Args:
        articles: List of articles to embed
        
    Returns:
        Articles with embeddings
    """
    logger.info("Running SBERT embedding pipeline...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Load SBERT model
        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare texts for embedding
        texts = []
        embedded_articles = []
        
        for article in articles:
            # Use translated text if available, otherwise title
            text = article.get('translated_text', '') or article.get('title', '')
            if text:
                texts.append(text)
                embedded_articles.append(article)
            else:
                # Skip articles without text
                logger.warning(f"Skipping article {article.get('id', 'unknown')} - no text")
        
        if not texts:
            logger.warning("No texts found for embedding")
            return articles
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(texts)} articles...")
        embeddings = encoder.encode(texts, show_progress_bar=True)
        
        # Add embeddings to articles
        for article, embedding in zip(embedded_articles, embeddings):
            article['embedding'] = embedding.tolist()
        
        logger.info(f"Generated {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")
        
        # Save embedded articles
        with open("data/embedded_articles.json", 'w', encoding='utf-8') as f:
            json.dump(embedded_articles, f, ensure_ascii=False, indent=2)
        
        return embedded_articles
        
    except Exception as e:
        logger.error(f"Error in embedding pipeline: {e}")
        return articles


def run_clustering_pipeline(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Run HDBSCAN clustering on embedded articles.
    
    Args:
        articles: List of articles with embeddings
        
    Returns:
        Articles with cluster assignments
    """
    logger.info("Running HDBSCAN clustering pipeline...")
    
    try:
        from pipelines.cluster_articles import cluster_articles
        
        clustered_articles = cluster_articles(
            articles=articles,
            output_path="data/clustered_articles.json",
            min_cluster_size=5,
            min_samples=3
        )
        
        # Log clustering results
        cluster_ids = [article.get('cluster_id', -1) for article in clustered_articles]
        unique_clusters = set(cluster_ids)
        n_clusters = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
        n_noise = cluster_ids.count(-1)
        
        logger.info(f"Clustering completed:")
        logger.info(f"  Clusters found: {n_clusters}")
        logger.info(f"  Noise points: {n_noise}/{len(clustered_articles)}")
        
        return clustered_articles
        
    except Exception as e:
        logger.error(f"Error in clustering pipeline: {e}")
        return articles


def run_forecasting_pipeline() -> Optional[float]:
    """
    Run XGBoost forecasting to predict tomorrow's escalation.
    
    Returns:
        Predicted escalation score for tomorrow
    """
    logger.info("Running XGBoost forecasting pipeline...")
    
    try:
        from train_forecasting_model import predict_next_day_escalation
        
        tomorrow_score = predict_next_day_escalation()
        
        logger.info(f"Forecasting completed:")
        logger.info(f"  Tomorrow's predicted escalation: {tomorrow_score:.3f}")
        
        return tomorrow_score
        
    except Exception as e:
        logger.error(f"Error in forecasting pipeline: {e}")
        return None


def save_pipeline_summary(results: Dict[str, Any]) -> None:
    """
    Save pipeline execution summary.
    
    Args:
        results: Pipeline execution results
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = f"logs/pipeline_summary_{timestamp}.json"
    
    # Also save as latest
    latest_path = "data/pipeline_summary.json"
    
    for path in [summary_path, latest_path]:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
    
    logger.info(f"Pipeline summary saved to: {summary_path}")


def run_full_pipeline(input_path: Optional[str] = None, 
                     articles: Optional[List[Dict[str, Any]]] = None,
                     skip_tagging: bool = False,
                     skip_clustering: bool = False,
                     skip_forecasting: bool = False) -> Dict[str, Any]:
    """
    Run the complete backend pipeline.
    
    Args:
        input_path: Optional path to input articles
        articles: Optional list of articles (overrides input_path)
        skip_tagging: Skip tagging pipeline
        skip_clustering: Skip clustering pipeline
        skip_forecasting: Skip forecasting pipeline
        
    Returns:
        Pipeline execution results
    """
    start_time = datetime.now()
    logger.info("🚀 Starting StraitWatch backend pipeline...")
    
    # Initialize results
    results = {
        'pipeline_version': '1.0.0',
        'start_time': start_time.isoformat(),
        'input_path': input_path,
        'pipeline_stats': {},
        'errors': []
    }
    
    try:
        # Step 1: Load articles
        if articles is None:
            articles = load_articles(input_path)
        
        if not articles:
            error_msg = "No articles loaded - pipeline cannot continue"
            logger.error(error_msg)
            results['errors'].append(error_msg)
            return results
        
        results['pipeline_stats']['articles_loaded'] = len(articles)
        
        # Step 2: Tagging pipeline
        if not skip_tagging:
            articles = run_tagging_pipeline(articles)
            results['pipeline_stats']['articles_tagged'] = len(articles)
        
        # Step 3: Escalation inference
        articles = run_inference_pipeline(articles)
        results['pipeline_stats']['articles_scored'] = len(articles)
        
        # Calculate escalation statistics
        scores = [article.get('escalation_score', 0) for article in articles]
        results['pipeline_stats']['avg_escalation_score'] = float(np.mean(scores))
        results['pipeline_stats']['high_risk_articles'] = sum(1 for s in scores if s > 0.7)
        
        # Step 4: SBERT embeddings
        articles = run_embedding_pipeline(articles)
        embedded_count = sum(1 for article in articles if 'embedding' in article)
        results['pipeline_stats']['articles_embedded'] = embedded_count
        
        # Step 5: Clustering
        if not skip_clustering and embedded_count > 0:
            articles = run_clustering_pipeline(articles)
            cluster_ids = [article.get('cluster_id', -1) for article in articles]
            unique_clusters = set(cluster_ids)
            results['pipeline_stats']['clusters_found'] = len(unique_clusters) - (1 if -1 in unique_clusters else 0)
            results['pipeline_stats']['noise_points'] = cluster_ids.count(-1)
        
        # Step 6: Forecasting
        if not skip_forecasting:
            tomorrow_score = run_forecasting_pipeline()
            if tomorrow_score is not None:
                results['tomorrow_escalation_forecast'] = tomorrow_score
        
        # Final results
        results['processed_articles'] = articles
        results['end_time'] = datetime.now().isoformat()
        results['total_runtime_seconds'] = (datetime.now() - start_time).total_seconds()
        
        # Summary statistics
        results['cluster_summary'] = {}
        if 'cluster_id' in articles[0] if articles else {}:
            cluster_counts = {}
            for article in articles:
                cluster_id = article.get('cluster_id', -1)
                cluster_counts[cluster_id] = cluster_counts.get(cluster_id, 0) + 1
            results['cluster_summary'] = dict(sorted(cluster_counts.items()))
        
        logger.info("✅ Backend pipeline completed successfully!")
        logger.info(f"📊 Pipeline Statistics:")
        for key, value in results['pipeline_stats'].items():
            logger.info(f"   {key}: {value}")
        
        if results.get('tomorrow_escalation_forecast'):
            logger.info(f"📈 Tomorrow's escalation forecast: {results['tomorrow_escalation_forecast']:.3f}")
        
    except Exception as e:
        error_msg = f"Pipeline failed with error: {str(e)}"
        logger.error(error_msg)
        results['errors'].append(error_msg)
        results['end_time'] = datetime.now().isoformat()
        results['total_runtime_seconds'] = (datetime.now() - start_time).total_seconds()
    
    # Save pipeline summary
    save_pipeline_summary(results)
    
    return results


def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="StraitWatch Backend ML Pipeline")
    parser.add_argument("--input-file", help="Path to input articles JSON file")
    parser.add_argument("--skip-tagging", action="store_true", help="Skip tagging pipeline")
    parser.add_argument("--skip-clustering", action="store_true", help="Skip clustering pipeline")
    parser.add_argument("--forecast-only", action="store_true", help="Only run forecasting")
    parser.add_argument("--evaluate", action="store_true", help="Run model evaluation after pipeline")
    
    args = parser.parse_args()
    
    if args.forecast_only:
        # Only run forecasting
        logger.info("Running forecasting pipeline only...")
        tomorrow_score = run_forecasting_pipeline()
        if tomorrow_score:
            print(f"\n📈 Tomorrow's escalation forecast: {tomorrow_score:.3f}")
        return
    
    # Run full pipeline
    results = run_full_pipeline(
        input_path=args.input_file,
        skip_tagging=args.skip_tagging,
        skip_clustering=args.skip_clustering
    )
    
    # Print summary
    print("\n" + "="*60)
    print("🎯 STRAITWATCH BACKEND PIPELINE SUMMARY")
    print("="*60)
    
    if results.get('errors'):
        print("❌ Pipeline completed with errors:")
        for error in results['errors']:
            print(f"   {error}")
    else:
        print("✅ Pipeline completed successfully!")
    
    print(f"\n📊 Pipeline Statistics:")
    for key, value in results.get('pipeline_stats', {}).items():
        print(f"   {key.replace('_', ' ').title()}: {value}")
    
    if results.get('tomorrow_escalation_forecast'):
        print(f"\n📈 Tomorrow's escalation forecast: {results['tomorrow_escalation_forecast']:.3f}")
    
    print(f"\n⏱️ Total runtime: {results.get('total_runtime_seconds', 0):.1f} seconds")
    print("="*60)
    
    # Run evaluation if requested
    if args.evaluate:
        print("\n🔍 Running model evaluation...")
        try:
            from evaluate_models import run_full_evaluation
            run_full_evaluation()
            print("✅ Model evaluation completed!")
        except Exception as e:
            print(f"❌ Model evaluation failed: {e}")


if __name__ == "__main__":
    main() 