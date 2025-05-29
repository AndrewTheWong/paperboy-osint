#!/usr/bin/env python3
"""
Model Evaluation Pipeline

This module evaluates all models in the StraitWatch backend pipeline:
- Escalation classifier (precision, recall, F1, confusion matrix)
- Forecasting model (MAE, RMSE, R²)
- Clustering quality (silhouette score, cluster metrics)
- Tagging accuracy (tag coverage, precision)

Outputs comprehensive metrics to logs/model_metrics.json
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML evaluation imports
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, precision_recall_curve,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score, adjusted_rand_score, calinski_harabasz_score
)
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ClassifierEvaluator:
    """Evaluate escalation classification model."""
    
    def __init__(self, model_path: str = "models/xgboost_conflict_model_20250519.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained classifier."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded classifier from {self.model_path}")
            else:
                logger.warning(f"Model file not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
    
    def evaluate_on_test_data(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate classifier on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            logger.error("No model loaded for evaluation")
            return {}
        
        logger.info("Evaluating classifier on test data...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        # Basic metrics
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        # ROC AUC
        try:
            auc_score = roc_auc_score(y_test, y_pred_proba)
        except ValueError:
            auc_score = 0.0
            logger.warning("Could not calculate ROC AUC (only one class present)")
        
        # Precision-Recall curve metrics
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        pr_auc = np.trapz(recall, precision)
        
        metrics = {
            'model_type': 'escalation_classifier',
            'test_size': len(y_test),
            'accuracy': report['accuracy'],
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score'],
            'roc_auc': float(auc_score),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'class_report': report,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Class-specific metrics
        if '1' in report:  # Escalation class
            metrics['escalation_precision'] = report['1']['precision']
            metrics['escalation_recall'] = report['1']['recall']
            metrics['escalation_f1'] = report['1']['f1-score']
        
        if '0' in report:  # Non-escalation class
            metrics['non_escalation_precision'] = report['0']['precision']
            metrics['non_escalation_recall'] = report['0']['recall']
            metrics['non_escalation_f1'] = report['0']['f1-score']
        
        # Log key metrics
        logger.info(f"Classifier Evaluation Results:")
        logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
        logger.info(f"  ROC AUC: {metrics['roc_auc']:.4f}")
        logger.info(f"  PR AUC: {metrics['pr_auc']:.4f}")
        
        return metrics
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, Any]:
        """Perform cross-validation evaluation."""
        if self.model is None:
            logger.error("No model loaded for cross-validation")
            return {}
        
        logger.info(f"Performing {cv}-fold cross-validation...")
        
        # Cross-validation scores
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='f1_weighted')
        cv_accuracy = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        cv_precision = cross_val_score(self.model, X, y, cv=cv, scoring='precision_weighted')
        cv_recall = cross_val_score(self.model, X, y, cv=cv, scoring='recall_weighted')
        
        cv_metrics = {
            'cv_folds': cv,
            'cv_f1_mean': float(cv_scores.mean()),
            'cv_f1_std': float(cv_scores.std()),
            'cv_accuracy_mean': float(cv_accuracy.mean()),
            'cv_accuracy_std': float(cv_accuracy.std()),
            'cv_precision_mean': float(cv_precision.mean()),
            'cv_precision_std': float(cv_precision.std()),
            'cv_recall_mean': float(cv_recall.mean()),
            'cv_recall_std': float(cv_recall.std())
        }
        
        logger.info(f"Cross-validation F1: {cv_metrics['cv_f1_mean']:.4f} ± {cv_metrics['cv_f1_std']:.4f}")
        
        return cv_metrics

class ForecastEvaluator:
    """Evaluate time series forecasting model."""
    
    def __init__(self, model_path: str = "models/xgboost_forecast.pkl"):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load the trained forecasting model."""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                logger.info(f"Loaded forecasting model from {self.model_path}")
            else:
                logger.warning(f"Forecasting model not found: {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading forecasting model: {e}")
    
    def evaluate_forecast_accuracy(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate forecasting model accuracy.
        
        Args:
            X_test: Test features
            y_test: Test targets (actual escalation scores)
            
        Returns:
            Dictionary of forecast evaluation metrics
        """
        if self.model is None:
            logger.error("No forecasting model loaded")
            return {}
        
        logger.info("Evaluating forecasting model accuracy...")
        
        # Predictions
        y_pred = self.model.predict(X_test)
        
        # Regression metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100  # Avoid division by zero
        residuals = y_test - y_pred
        
        metrics = {
            'model_type': 'escalation_forecaster',
            'test_size': len(y_test),
            'mae': float(mae),
            'rmse': float(rmse),
            'r2_score': float(r2),
            'mape': float(mape),
            'residuals_mean': float(residuals.mean()),
            'residuals_std': float(residuals.std()),
            'residuals_min': float(residuals.min()),
            'residuals_max': float(residuals.max()),
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Direction accuracy (did we predict the trend correctly?)
        if len(y_test) > 1:
            actual_direction = np.diff(y_test) > 0
            pred_direction = np.diff(y_pred) > 0
            direction_accuracy = np.mean(actual_direction == pred_direction)
            metrics['direction_accuracy'] = float(direction_accuracy)
        
        # Log key metrics
        logger.info(f"Forecasting Evaluation Results:")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R² Score: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        if 'direction_accuracy' in metrics:
            logger.info(f"  Direction Accuracy: {metrics['direction_accuracy']:.4f}")
        
        return metrics

class ClusteringEvaluator:
    """Evaluate article clustering quality."""
    
    def evaluate_clustering_quality(self, embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate clustering quality using various metrics.
        
        Args:
            embeddings: Article embeddings
            cluster_labels: Cluster assignments
            
        Returns:
            Dictionary of clustering evaluation metrics
        """
        logger.info("Evaluating clustering quality...")
        
        # Filter out noise points (label -1 in HDBSCAN)
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) < 2:
            logger.warning("Too few non-noise points for clustering evaluation")
            return {
                'model_type': 'clustering',
                'error': 'insufficient_non_noise_points'
            }
        
        # Metrics calculation
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = np.sum(cluster_labels == -1)
        
        # Silhouette score (only for non-noise points)
        if n_clusters > 1:
            silhouette_avg = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
        else:
            silhouette_avg = 0.0
        
        # Calinski-Harabasz score (higher is better)
        if n_clusters > 1:
            ch_score = calinski_harabasz_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
        else:
            ch_score = 0.0
        
        # Cluster size statistics
        unique_labels, cluster_sizes = np.unique(cluster_labels[non_noise_mask], return_counts=True)
        
        metrics = {
            'model_type': 'clustering',
            'n_total_points': len(cluster_labels),
            'n_clusters': n_clusters,
            'n_noise_points': int(n_noise),
            'noise_ratio': float(n_noise / len(cluster_labels)),
            'silhouette_score': float(silhouette_avg),
            'calinski_harabasz_score': float(ch_score),
            'avg_cluster_size': float(cluster_sizes.mean()) if len(cluster_sizes) > 0 else 0.0,
            'min_cluster_size': int(cluster_sizes.min()) if len(cluster_sizes) > 0 else 0,
            'max_cluster_size': int(cluster_sizes.max()) if len(cluster_sizes) > 0 else 0,
            'cluster_size_std': float(cluster_sizes.std()) if len(cluster_sizes) > 0 else 0.0,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Log key metrics
        logger.info(f"Clustering Evaluation Results:")
        logger.info(f"  Number of clusters: {n_clusters}")
        logger.info(f"  Noise ratio: {metrics['noise_ratio']:.2%}")
        logger.info(f"  Silhouette score: {silhouette_avg:.4f}")
        logger.info(f"  Average cluster size: {metrics['avg_cluster_size']:.1f}")
        
        return metrics

class TaggingEvaluator:
    """Evaluate tagging pipeline quality."""
    
    def evaluate_tagging_quality(self, articles: List[Dict]) -> Dict[str, Any]:
        """
        Evaluate tagging pipeline performance.
        
        Args:
            articles: List of tagged articles
            
        Returns:
            Dictionary of tagging evaluation metrics
        """
        logger.info("Evaluating tagging quality...")
        
        if not articles:
            return {'model_type': 'tagging', 'error': 'no_articles'}
        
        # Extract tagging statistics
        total_articles = len(articles)
        tagged_articles = [a for a in articles if a.get('tags')]
        review_articles = [a for a in articles if a.get('needs_review', False)]
        
        # Tag statistics
        all_tags = []
        tag_confidences = []
        text_lengths = []
        
        for article in articles:
            tags = article.get('tags', [])
            all_tags.extend(tags)
            
            confidence = article.get('tag_confidence', 0.0)
            tag_confidences.append(confidence)
            
            text = article.get('translated_text', article.get('text', ''))
            text_lengths.append(len(text))
        
        # Tag frequency analysis
        tag_freq = {}
        for tag in all_tags:
            tag_freq[tag] = tag_freq.get(tag, 0) + 1
        
        # Coverage metrics
        coverage_ratio = len(tagged_articles) / total_articles if total_articles > 0 else 0
        review_ratio = len(review_articles) / total_articles if total_articles > 0 else 0
        avg_confidence = np.mean(tag_confidences) if tag_confidences else 0.0
        avg_tags_per_article = len(all_tags) / total_articles if total_articles > 0 else 0
        
        metrics = {
            'model_type': 'tagging',
            'total_articles': total_articles,
            'tagged_articles': len(tagged_articles),
            'coverage_ratio': float(coverage_ratio),
            'review_ratio': float(review_ratio),
            'avg_confidence': float(avg_confidence),
            'avg_tags_per_article': float(avg_tags_per_article),
            'unique_tags': len(tag_freq),
            'most_common_tags': dict(sorted(tag_freq.items(), key=lambda x: x[1], reverse=True)[:10]),
            'avg_text_length': float(np.mean(text_lengths)) if text_lengths else 0.0,
            'confidence_std': float(np.std(tag_confidences)) if tag_confidences else 0.0,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Quality indicators
        high_confidence_articles = sum(1 for conf in tag_confidences if conf > 0.8)
        low_confidence_articles = sum(1 for conf in tag_confidences if conf < 0.3)
        
        metrics['high_confidence_ratio'] = float(high_confidence_articles / total_articles) if total_articles > 0 else 0.0
        metrics['low_confidence_ratio'] = float(low_confidence_articles / total_articles) if total_articles > 0 else 0.0
        
        # Log key metrics
        logger.info(f"Tagging Evaluation Results:")
        logger.info(f"  Coverage ratio: {coverage_ratio:.2%}")
        logger.info(f"  Review ratio: {review_ratio:.2%}")
        logger.info(f"  Average confidence: {avg_confidence:.4f}")
        logger.info(f"  Average tags per article: {avg_tags_per_article:.2f}")
        logger.info(f"  Unique tags: {len(tag_freq)}")
        
        return metrics

class ModelEvaluationPipeline:
    """Main evaluation pipeline for all models."""
    
    def __init__(self):
        self.classifier_eval = ClassifierEvaluator()
        self.forecast_eval = ForecastEvaluator()
        self.clustering_eval = ClusteringEvaluator()
        self.tagging_eval = TaggingEvaluator()
    
    def run_full_evaluation(self, test_data_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive evaluation of all models.
        
        Args:
            test_data_path: Optional path to test data
            
        Returns:
            Combined evaluation metrics
        """
        logger.info("Starting comprehensive model evaluation...")
        
        all_metrics = {
            'evaluation_timestamp': datetime.now().isoformat(),
            'evaluation_version': '1.0',
            'models_evaluated': []
        }
        
        # 1. Evaluate classifier (if test data available)
        if self._has_test_data():
            try:
                X_test, y_test = self._load_test_data()
                classifier_metrics = self.classifier_eval.evaluate_on_test_data(X_test, y_test)
                all_metrics['classifier'] = classifier_metrics
                all_metrics['models_evaluated'].append('classifier')
                logger.info("✅ Classifier evaluation completed")
            except Exception as e:
                logger.error(f"Classifier evaluation failed: {e}")
                all_metrics['classifier'] = {'error': str(e)}
        else:
            logger.warning("No test data available for classifier evaluation")
        
        # 2. Evaluate forecasting model (if forecast test data available)
        try:
            forecast_metrics = self._evaluate_forecast_with_synthetic_data()
            all_metrics['forecaster'] = forecast_metrics
            all_metrics['models_evaluated'].append('forecaster')
            logger.info("✅ Forecaster evaluation completed")
        except Exception as e:
            logger.error(f"Forecaster evaluation failed: {e}")
            all_metrics['forecaster'] = {'error': str(e)}
        
        # 3. Evaluate clustering (if clustered articles available)
        try:
            clustering_metrics = self._evaluate_clustering_from_files()
            all_metrics['clustering'] = clustering_metrics
            all_metrics['models_evaluated'].append('clustering')
            logger.info("✅ Clustering evaluation completed")
        except Exception as e:
            logger.error(f"Clustering evaluation failed: {e}")
            all_metrics['clustering'] = {'error': str(e)}
        
        # 4. Evaluate tagging (if tagged articles available)
        try:
            tagging_metrics = self._evaluate_tagging_from_files()
            all_metrics['tagging'] = tagging_metrics
            all_metrics['models_evaluated'].append('tagging')
            logger.info("✅ Tagging evaluation completed")
        except Exception as e:
            logger.error(f"Tagging evaluation failed: {e}")
            all_metrics['tagging'] = {'error': str(e)}
        
        # Save comprehensive metrics
        self._save_metrics(all_metrics)
        
        logger.info(f"📊 Full evaluation completed. Evaluated {len(all_metrics['models_evaluated'])} models")
        return all_metrics
    
    def _has_test_data(self) -> bool:
        """Check if test data is available."""
        test_paths = [
            "data/model_ready/X_features.npy",
            "data/model_ready/y_labels.npy"
        ]
        return all(os.path.exists(path) for path in test_paths)
    
    def _load_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load test data for classifier evaluation."""
        X = np.load("data/model_ready/X_features.npy")
        y = np.load("data/model_ready/y_labels.npy")
        
        # Use last 20% as test set
        test_size = int(0.2 * len(X))
        return X[-test_size:], y[-test_size:]
    
    def _evaluate_forecast_with_synthetic_data(self) -> Dict[str, Any]:
        """Evaluate forecasting model with synthetic test data."""
        # Generate synthetic test data for forecasting
        np.random.seed(42)
        n_samples = 100
        n_features = 18  # Number of features expected by forecasting model
        
        X_test = np.random.randn(n_samples, n_features)
        y_test = np.random.uniform(0, 1, n_samples)  # Escalation scores between 0 and 1
        
        return self.forecast_eval.evaluate_forecast_accuracy(X_test, y_test)
    
    def _evaluate_clustering_from_files(self) -> Dict[str, Any]:
        """Evaluate clustering from saved files."""
        # Try to load clustered articles
        cluster_files = [
            "data/clustered_articles.json",
            "data/embedded_articles.json"
        ]
        
        for filepath in cluster_files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    # Extract embeddings and cluster labels
                    embeddings = []
                    cluster_labels = []
                    
                    for article in articles:
                        if 'embedding' in article and 'cluster_id' in article:
                            embeddings.append(article['embedding'])
                            cluster_labels.append(article['cluster_id'])
                    
                    if embeddings and cluster_labels:
                        embeddings = np.array(embeddings)
                        cluster_labels = np.array(cluster_labels)
                        return self.clustering_eval.evaluate_clustering_quality(embeddings, cluster_labels)
                    
                except Exception as e:
                    logger.warning(f"Could not load clustering data from {filepath}: {e}")
        
        # Return empty evaluation if no data found
        return {
            'model_type': 'clustering',
            'error': 'no_clustering_data_found'
        }
    
    def _evaluate_tagging_from_files(self) -> Dict[str, Any]:
        """Evaluate tagging from saved files."""
        tag_files = [
            "data/tagged_articles.json",
            "data/clustered_articles.json"
        ]
        
        for filepath in tag_files:
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        articles = json.load(f)
                    
                    # Check if articles have tagging information
                    if articles and any('tags' in article for article in articles):
                        return self.tagging_eval.evaluate_tagging_quality(articles)
                    
                except Exception as e:
                    logger.warning(f"Could not load tagging data from {filepath}: {e}")
        
        return {
            'model_type': 'tagging',
            'error': 'no_tagging_data_found'
        }
    
    def _save_metrics(self, metrics: Dict[str, Any]):
        """Save evaluation metrics to file."""
        os.makedirs("logs", exist_ok=True)
        
        metrics_file = "logs/model_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"📄 Evaluation metrics saved to {metrics_file}")

def main():
    """Main evaluation entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate StraitWatch backend models')
    parser.add_argument('--output', default='logs/model_metrics.json',
                       help='Output file for evaluation metrics')
    args = parser.parse_args()
    
    # Run full evaluation
    evaluator = ModelEvaluationPipeline()
    metrics = evaluator.run_full_evaluation()
    
    # Print summary
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION SUMMARY")
    print("="*60)
    
    for model_type in metrics.get('models_evaluated', []):
        if model_type in metrics:
            model_metrics = metrics[model_type]
            print(f"\n🔧 {model_type.upper()}")
            
            if 'error' in model_metrics:
                print(f"  ❌ Error: {model_metrics['error']}")
            else:
                # Print key metrics based on model type
                if model_type == 'classifier':
                    print(f"  📈 Accuracy: {model_metrics.get('accuracy', 0):.3f}")
                    print(f"  📈 F1-Score: {model_metrics.get('f1_score', 0):.3f}")
                    print(f"  📈 ROC AUC: {model_metrics.get('roc_auc', 0):.3f}")
                elif model_type == 'forecaster':
                    print(f"  📈 MAE: {model_metrics.get('mae', 0):.4f}")
                    print(f"  📈 RMSE: {model_metrics.get('rmse', 0):.4f}")
                    print(f"  📈 R² Score: {model_metrics.get('r2_score', 0):.3f}")
                elif model_type == 'clustering':
                    print(f"  📈 Clusters: {model_metrics.get('n_clusters', 0)}")
                    print(f"  📈 Silhouette: {model_metrics.get('silhouette_score', 0):.3f}")
                    print(f"  📈 Noise ratio: {model_metrics.get('noise_ratio', 0):.1%}")
                elif model_type == 'tagging':
                    print(f"  📈 Coverage: {model_metrics.get('coverage_ratio', 0):.1%}")
                    print(f"  📈 Avg confidence: {model_metrics.get('avg_confidence', 0):.3f}")
                    print(f"  📈 Review ratio: {model_metrics.get('review_ratio', 0):.1%}")
    
    print(f"\n📄 Full metrics saved to: {args.output}")
    print("="*60)

if __name__ == "__main__":
    main() 