#!/usr/bin/env python3
"""
StraitWatch Backend Pipeline Orchestrator

This is the main script that orchestrates the complete StraitWatch Phase 1 backend ML pipeline:
1. Article tagging (keyword + ML)
2. SBERT embedding generation  
3. Escalation prediction
4. HDBSCAN clustering
5. Time series forecasting
6. Model evaluation

Usage:
    python -m pipelines.run_backend_pipeline --input data/translated_articles.json
    python pipelines/run_backend_pipeline.py --full-pipeline
"""

import os
import sys
import json
import logging
import time
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backend_pipeline')

# Import pipeline components
try:
    from .tagging_pipeline import ArticleTaggingPipeline, TagConfig
    from .inference_pipeline import predict_escalation
    from .cluster_articles import cluster_articles
    from .train_forecasting_model import main as train_forecasting
    from .evaluate_models import ModelEvaluationPipeline
except ImportError:
    # Fallback for direct execution
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from tagging_pipeline import ArticleTaggingPipeline, TagConfig
    from inference_pipeline import predict_escalation
    from cluster_articles import cluster_articles
    from train_forecasting_model import main as train_forecasting
    from evaluate_models import ModelEvaluationPipeline

# Try importing SBERT for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    logger.error("Missing dependency: sentence-transformers. Install with: pip install sentence-transformers")
    HAS_SBERT = False

import numpy as np

class BackendPipelineOrchestrator:
    """Main orchestrator for the StraitWatch backend pipeline."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._get_default_config()
        self.pipeline_metrics = {}
        self.start_time = None
        
        # Initialize components
        self.tagging_pipeline = ArticleTaggingPipeline(TagConfig())
        self.model_evaluator = ModelEvaluationPipeline()
        
        # Initialize SBERT encoder
        self.encoder = None
        if HAS_SBERT:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("✅ Loaded Sentence-BERT encoder")
            except Exception as e:
                logger.error(f"Failed to load SBERT encoder: {e}")
        else:
            logger.warning("SBERT not available - embeddings will be skipped")
    
    def _get_default_config(self) -> Dict:
        """Get default pipeline configuration."""
        return {
            'input_file': 'data/translated_articles.json',
            'output_dir': 'data',
            'enable_tagging': True,
            'enable_embedding': True,
            'enable_escalation': True,
            'enable_clustering': True,
            'enable_forecasting': False,  # Requires historical data
            'enable_evaluation': True,
            'batch_size': 100,
            'save_intermediate': True
        }
    
    def load_articles(self, filepath: str) -> List[Dict[str, Any]]:
        """Load articles from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                articles = json.load(f)
            logger.info(f"📖 Loaded {len(articles)} articles from {filepath}")
            return articles
        except FileNotFoundError:
            logger.error(f"Input file not found: {filepath}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error in {filepath}: {e}")
            return []
    
    def save_articles(self, articles: List[Dict[str, Any]], filepath: str, stage: str):
        """Save articles to JSON file with stage information."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        logger.info(f"💾 Saved {len(articles)} {stage} articles to {filepath} ({file_size_mb:.2f} MB)")
    
    def step_1_tagging(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 1: Tag articles using keyword and ML approaches."""
        if not self.config['enable_tagging']:
            logger.info("⏭️ Skipping tagging step")
            return articles
        
        logger.info("🏷️ Step 1: Tagging articles...")
        start_time = time.time()
        
        tagged_articles = self.tagging_pipeline.tag_articles_batch(articles)
        
        self.pipeline_metrics['tagging'] = {
            'duration': time.time() - start_time,
            'articles_processed': len(tagged_articles),
            'avg_tags_per_article': np.mean([len(a.get('tags', [])) for a in tagged_articles]),
            'review_articles': sum(1 for a in tagged_articles if a.get('needs_review', False))
        }
        
        if self.config['save_intermediate']:
            output_path = os.path.join(self.config['output_dir'], 'tagged_articles.json')
            self.save_articles(tagged_articles, output_path, 'tagged')
        
        logger.info(f"✅ Tagging completed in {self.pipeline_metrics['tagging']['duration']:.2f}s")
        return tagged_articles
    
    def step_2_embedding(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 2: Generate SBERT embeddings for articles."""
        if not self.config['enable_embedding'] or not self.encoder:
            logger.info("⏭️ Skipping embedding step")
            return articles
        
        logger.info("🧠 Step 2: Generating embeddings...")
        start_time = time.time()
        
        embedded_articles = []
        batch_size = self.config['batch_size']
        
        for i in range(0, len(articles), batch_size):
            batch = articles[i:i + batch_size]
            
            # Extract text for embedding
            texts = []
            for article in batch:
                text = article.get('translated_text', article.get('text', ''))
                if not text:
                    text = article.get('title', '')
                texts.append(text)
            
            # Generate embeddings
            try:
                embeddings = self.encoder.encode(texts, show_progress_bar=False)
                
                # Add embeddings to articles
                for j, article in enumerate(batch):
                    article_copy = article.copy()
                    article_copy['embedding'] = embeddings[j].tolist()
                    embedded_articles.append(article_copy)
                
                if (i + batch_size) % (batch_size * 5) == 0:
                    logger.info(f"  Embedded {i + batch_size}/{len(articles)} articles")
                    
            except Exception as e:
                logger.error(f"Error generating embeddings for batch {i//batch_size}: {e}")
                # Add articles without embeddings
                embedded_articles.extend(batch)
        
        self.pipeline_metrics['embedding'] = {
            'duration': time.time() - start_time,
            'articles_processed': len(embedded_articles),
            'embedding_dim': len(embedded_articles[0]['embedding']) if embedded_articles and 'embedding' in embedded_articles[0] else 0
        }
        
        if self.config['save_intermediate']:
            output_path = os.path.join(self.config['output_dir'], 'embedded_articles.json')
            self.save_articles(embedded_articles, output_path, 'embedded')
        
        logger.info(f"✅ Embedding completed in {self.pipeline_metrics['embedding']['duration']:.2f}s")
        return embedded_articles
    
    def step_3_escalation(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 3: Predict escalation scores for articles."""
        if not self.config['enable_escalation']:
            logger.info("⏭️ Skipping escalation prediction step")
            return articles
        
        logger.info("🎯 Step 3: Predicting escalation scores...")
        start_time = time.time()
        
        articles_with_scores = []
        prediction_errors = 0
        
        for article in articles:
            article_copy = article.copy()
            
            try:
                text = article.get('translated_text', article.get('text', ''))
                if text:
                    escalation_score = predict_escalation(text)
                    article_copy['escalation_score'] = escalation_score
                else:
                    article_copy['escalation_score'] = 0.0
                    prediction_errors += 1
            except Exception as e:
                logger.warning(f"Error predicting escalation for article {article.get('id', 'unknown')}: {e}")
                article_copy['escalation_score'] = 0.0
                prediction_errors += 1
            
            articles_with_scores.append(article_copy)
        
        self.pipeline_metrics['escalation'] = {
            'duration': time.time() - start_time,
            'articles_processed': len(articles_with_scores),
            'prediction_errors': prediction_errors,
            'avg_escalation_score': np.mean([a['escalation_score'] for a in articles_with_scores]),
            'high_escalation_articles': sum(1 for a in articles_with_scores if a['escalation_score'] > 0.7)
        }
        
        if self.config['save_intermediate']:
            output_path = os.path.join(self.config['output_dir'], 'scored_articles.json')
            self.save_articles(articles_with_scores, output_path, 'escalation-scored')
        
        logger.info(f"✅ Escalation prediction completed in {self.pipeline_metrics['escalation']['duration']:.2f}s")
        logger.info(f"📊 Average escalation score: {self.pipeline_metrics['escalation']['avg_escalation_score']:.4f}")
        logger.info(f"🚨 High escalation articles: {self.pipeline_metrics['escalation']['high_escalation_articles']}")
        
        return articles_with_scores
    
    def step_4_clustering(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Step 4: Cluster articles using HDBSCAN."""
        if not self.config['enable_clustering']:
            logger.info("⏭️ Skipping clustering step")
            return articles
        
        logger.info("🔗 Step 4: Clustering articles...")
        start_time = time.time()
        
        # Check if articles have embeddings
        articles_with_embeddings = [a for a in articles if 'embedding' in a]
        if not articles_with_embeddings:
            logger.warning("No articles with embeddings found - skipping clustering")
            return articles
        
        try:
            output_path = os.path.join(self.config['output_dir'], 'clustered_articles.json') if self.config['save_intermediate'] else None
            
            clustered_articles = cluster_articles(
                articles=articles_with_embeddings,
                output_path=output_path,
                min_cluster_size=5,
                min_samples=3
            )
            
            # Extract clustering metrics
            if clustered_articles and 'clustering_metadata' in clustered_articles[0]:
                metadata = clustered_articles[0]['clustering_metadata']
                self.pipeline_metrics['clustering'] = {
                    'duration': time.time() - start_time,
                    'articles_processed': len(clustered_articles),
                    'n_clusters': metadata.get('n_clusters', 0),
                    'n_noise': metadata.get('n_noise', 0),
                    'noise_ratio': metadata.get('noise_ratio', 0),
                    'silhouette_score': metadata.get('silhouette_score', 0)
                }
            else:
                self.pipeline_metrics['clustering'] = {
                    'duration': time.time() - start_time,
                    'articles_processed': len(clustered_articles),
                    'error': 'no_clustering_metadata'
                }
            
            logger.info(f"✅ Clustering completed in {self.pipeline_metrics['clustering']['duration']:.2f}s")
            if 'n_clusters' in self.pipeline_metrics['clustering']:
                logger.info(f"📊 Found {self.pipeline_metrics['clustering']['n_clusters']} clusters")
                logger.info(f"🔇 Noise ratio: {self.pipeline_metrics['clustering']['noise_ratio']:.1%}")
            
            return clustered_articles
            
        except Exception as e:
            logger.error(f"Clustering failed: {e}")
            self.pipeline_metrics['clustering'] = {
                'duration': time.time() - start_time,
                'articles_processed': len(articles),
                'error': str(e)
            }
            return articles
    
    def step_5_forecasting(self) -> bool:
        """Step 5: Train time series forecasting model."""
        if not self.config['enable_forecasting']:
            logger.info("⏭️ Skipping forecasting model training")
            return True
        
        logger.info("📈 Step 5: Training forecasting model...")
        start_time = time.time()
        
        try:
            train_forecasting()
            
            self.pipeline_metrics['forecasting'] = {
                'duration': time.time() - start_time,
                'status': 'completed'
            }
            
            logger.info(f"✅ Forecasting model training completed in {self.pipeline_metrics['forecasting']['duration']:.2f}s")
            return True
            
        except Exception as e:
            logger.error(f"Forecasting model training failed: {e}")
            self.pipeline_metrics['forecasting'] = {
                'duration': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
            return False
    
    def step_6_evaluation(self) -> Dict[str, Any]:
        """Step 6: Evaluate all models in the pipeline."""
        if not self.config['enable_evaluation']:
            logger.info("⏭️ Skipping model evaluation")
            return {}
        
        logger.info("📊 Step 6: Evaluating models...")
        start_time = time.time()
        
        try:
            evaluation_metrics = self.model_evaluator.run_full_evaluation()
            
            self.pipeline_metrics['evaluation'] = {
                'duration': time.time() - start_time,
                'models_evaluated': evaluation_metrics.get('models_evaluated', []),
                'status': 'completed'
            }
            
            logger.info(f"✅ Model evaluation completed in {self.pipeline_metrics['evaluation']['duration']:.2f}s")
            logger.info(f"📈 Evaluated models: {', '.join(evaluation_metrics.get('models_evaluated', []))}")
            
            return evaluation_metrics
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {e}")
            self.pipeline_metrics['evaluation'] = {
                'duration': time.time() - start_time,
                'status': 'failed',
                'error': str(e)
            }
            return {}
    
    def run_full_pipeline(self, input_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete backend pipeline.
        
        Args:
            input_file: Optional path to input articles file
            
        Returns:
            Pipeline execution summary
        """
        logger.info("🚀 Starting StraitWatch Backend Pipeline...")
        self.start_time = time.time()
        
        # Load input articles
        input_file = input_file or self.config['input_file']
        articles = self.load_articles(input_file)
        
        if not articles:
            logger.error("No articles loaded - pipeline cannot continue")
            return {'status': 'failed', 'error': 'no_input_articles'}
        
        pipeline_summary = {
            'start_time': datetime.now().isoformat(),
            'input_file': input_file,
            'initial_article_count': len(articles),
            'config': self.config,
            'steps_completed': []
        }
        
        try:
            # Step 1: Tagging
            articles = self.step_1_tagging(articles)
            pipeline_summary['steps_completed'].append('tagging')
            
            # Step 2: Embedding
            articles = self.step_2_embedding(articles)
            pipeline_summary['steps_completed'].append('embedding')
            
            # Step 3: Escalation Prediction
            articles = self.step_3_escalation(articles)
            pipeline_summary['steps_completed'].append('escalation')
            
            # Step 4: Clustering
            articles = self.step_4_clustering(articles)
            pipeline_summary['steps_completed'].append('clustering')
            
            # Save final processed articles
            final_output_path = os.path.join(self.config['output_dir'], 'processed_articles.json')
            self.save_articles(articles, final_output_path, 'final processed')
            pipeline_summary['final_output'] = final_output_path
            pipeline_summary['final_article_count'] = len(articles)
            
            # Step 5: Forecasting (optional)
            if self.config['enable_forecasting']:
                forecasting_success = self.step_5_forecasting()
                if forecasting_success:
                    pipeline_summary['steps_completed'].append('forecasting')
            
            # Step 6: Evaluation
            evaluation_metrics = self.step_6_evaluation()
            if evaluation_metrics:
                pipeline_summary['steps_completed'].append('evaluation')
                pipeline_summary['evaluation_metrics'] = evaluation_metrics
            
            # Final statistics
            total_duration = time.time() - self.start_time
            pipeline_summary['total_duration'] = total_duration
            pipeline_summary['step_metrics'] = self.pipeline_metrics
            pipeline_summary['status'] = 'completed'
            
            # Save pipeline summary
            summary_path = os.path.join(self.config['output_dir'], 'pipeline_summary.json')
            with open(summary_path, 'w') as f:
                json.dump(pipeline_summary, f, indent=2)
            
            logger.info(f"🎉 Pipeline completed successfully in {total_duration:.2f}s")
            logger.info(f"📄 Summary saved to: {summary_path}")
            
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            pipeline_summary['status'] = 'failed'
            pipeline_summary['error'] = str(e)
            pipeline_summary['total_duration'] = time.time() - self.start_time
            pipeline_summary['step_metrics'] = self.pipeline_metrics
            
            return pipeline_summary
    
    def run_single_step(self, step: str, input_file: str) -> bool:
        """Run a single pipeline step."""
        articles = self.load_articles(input_file)
        if not articles:
            return False
        
        if step == 'tagging':
            self.step_1_tagging(articles)
        elif step == 'embedding':
            self.step_2_embedding(articles)
        elif step == 'escalation':
            self.step_3_escalation(articles)
        elif step == 'clustering':
            self.step_4_clustering(articles)
        elif step == 'forecasting':
            return self.step_5_forecasting()
        elif step == 'evaluation':
            self.step_6_evaluation()
        else:
            logger.error(f"Unknown step: {step}")
            return False
        
        return True

def main():
    """Main entry point for the backend pipeline."""
    parser = argparse.ArgumentParser(description='StraitWatch Backend Pipeline Orchestrator')
    
    # Input/output options
    parser.add_argument('--input', default='data/translated_articles.json',
                       help='Input articles JSON file')
    parser.add_argument('--output-dir', default='data',
                       help='Output directory for processed files')
    
    # Pipeline control
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run the complete pipeline')
    parser.add_argument('--step', choices=['tagging', 'embedding', 'escalation', 'clustering', 'forecasting', 'evaluation'],
                       help='Run a single pipeline step')
    
    # Step options
    parser.add_argument('--skip-tagging', action='store_true',
                       help='Skip the tagging step')
    parser.add_argument('--skip-embedding', action='store_true',
                       help='Skip the embedding step')
    parser.add_argument('--skip-escalation', action='store_true',
                       help='Skip the escalation prediction step')
    parser.add_argument('--skip-clustering', action='store_true',
                       help='Skip the clustering step')
    parser.add_argument('--enable-forecasting', action='store_true',
                       help='Enable forecasting model training')
    parser.add_argument('--skip-evaluation', action='store_true',
                       help='Skip the evaluation step')
    
    # Performance options
    parser.add_argument('--batch-size', type=int, default=100,
                       help='Batch size for processing')
    parser.add_argument('--no-save-intermediate', action='store_true',
                       help='Skip saving intermediate results')
    
    args = parser.parse_args()
    
    # Build configuration
    config = {
        'input_file': args.input,
        'output_dir': args.output_dir,
        'enable_tagging': not args.skip_tagging,
        'enable_embedding': not args.skip_embedding,
        'enable_escalation': not args.skip_escalation,
        'enable_clustering': not args.skip_clustering,
        'enable_forecasting': args.enable_forecasting,
        'enable_evaluation': not args.skip_evaluation,
        'batch_size': args.batch_size,
        'save_intermediate': not args.no_save_intermediate
    }
    
    # Initialize orchestrator
    orchestrator = BackendPipelineOrchestrator(config)
    
    # Run pipeline
    if args.step:
        # Run single step
        success = orchestrator.run_single_step(args.step, args.input)
        if success:
            print(f"✅ Step '{args.step}' completed successfully")
        else:
            print(f"❌ Step '{args.step}' failed")
            sys.exit(1)
    else:
        # Run full pipeline (default)
        summary = orchestrator.run_full_pipeline()
        
        # Print summary
        print("\n" + "="*60)
        print("🎯 STRAITWATCH BACKEND PIPELINE SUMMARY")
        print("="*60)
        print(f"📊 Status: {summary['status'].upper()}")
        print(f"⏱️ Total Duration: {summary.get('total_duration', 0):.2f}s")
        print(f"📁 Input: {summary['input_file']}")
        print(f"📄 Articles Processed: {summary.get('final_article_count', 0)}")
        print(f"🔧 Steps Completed: {', '.join(summary['steps_completed'])}")
        
        if summary['status'] == 'failed':
            print(f"❌ Error: {summary.get('error', 'Unknown error')}")
            sys.exit(1)
        else:
            print(f"💾 Final Output: {summary.get('final_output', 'N/A')}")
            
        print("="*60)

if __name__ == "__main__":
    main() 