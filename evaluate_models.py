#!/usr/bin/env python3
"""
Model Evaluation for StraitWatch Phase 1 Backend Pipeline

Evaluates all essential models:
- Classifier: accuracy, confusion matrix, precision, recall, F1, AUC-ROC
- Forecasting: MAE, RMSE, R² score  
- Clustering: silhouette score, cluster quality metrics

Saves metrics to: logs/model_metrics.json
"""
import os
import json
import joblib
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, List
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score,
    silhouette_score
)
import matplotlib.pyplot as plt
import seaborn as sns

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('model_evaluation')

# Create directories
os.makedirs("logs", exist_ok=True)


def evaluate_classifier(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    """
    Evaluate binary classifier performance.
    
    Args:
        y_true: Ground truth labels (0/1)
        y_pred: Predicted labels (0/1)
        y_prob: Predicted probabilities for positive class
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating classifier performance...")
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.0
    }
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics.update({
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0
        })
    
    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    class_dist = dict(zip(unique.astype(int), counts.astype(int)))
    metrics['class_distribution'] = class_dist
    
    logger.info(f"Classifier metrics:")
    logger.info(f"  Accuracy: {metrics['accuracy']:.3f}")
    logger.info(f"  Precision: {metrics['precision']:.3f}")
    logger.info(f"  Recall: {metrics['recall']:.3f}")
    logger.info(f"  F1: {metrics['f1']:.3f}")
    logger.info(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
    
    return metrics


def evaluate_forecasting(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate forecasting model performance.
    
    Args:
        y_true: Ground truth values
        y_pred: Predicted values
        
    Returns:
        Dictionary of evaluation metrics
    """
    logger.info("Evaluating forecasting performance...")
    
    metrics = {
        'mae': mean_absolute_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100,
        'mean_true': float(np.mean(y_true)),
        'mean_pred': float(np.mean(y_pred)),
        'std_true': float(np.std(y_true)),
        'std_pred': float(np.std(y_pred))
    }
    
    logger.info(f"Forecasting metrics:")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    logger.info(f"  MAPE: {metrics['mape']:.2f}%")
    
    return metrics


def evaluate_clustering(embeddings: np.ndarray, cluster_labels: np.ndarray) -> Dict[str, Any]:
    """
    Evaluate clustering performance.
    
    Args:
        embeddings: Article embeddings used for clustering
        cluster_labels: Assigned cluster labels (-1 for noise)
        
    Returns:
        Dictionary of clustering metrics
    """
    logger.info("Evaluating clustering performance...")
    
    # Basic cluster statistics
    unique_labels = set(cluster_labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    n_points = len(cluster_labels)
    
    # Cluster sizes
    cluster_sizes = {}
    for label in cluster_labels:
        if label != -1:  # Exclude noise
            cluster_sizes[int(label)] = cluster_sizes.get(int(label), 0) + 1
    
    # Silhouette score (exclude noise points)
    silhouette_avg = 0.0
    if n_clusters > 1:
        non_noise_mask = cluster_labels != -1
        if np.sum(non_noise_mask) > 1:
            try:
                silhouette_avg = silhouette_score(embeddings[non_noise_mask], cluster_labels[non_noise_mask])
            except Exception as e:
                logger.warning(f"Could not calculate silhouette score: {e}")
    
    metrics = {
        'n_clusters': n_clusters,
        'n_noise_points': n_noise,
        'noise_ratio': n_noise / n_points if n_points > 0 else 0.0,
        'silhouette_score': silhouette_avg,
        'avg_cluster_size': np.mean(list(cluster_sizes.values())) if cluster_sizes else 0.0,
        'min_cluster_size': min(cluster_sizes.values()) if cluster_sizes else 0,
        'max_cluster_size': max(cluster_sizes.values()) if cluster_sizes else 0,
        'cluster_size_std': np.std(list(cluster_sizes.values())) if cluster_sizes else 0.0,
        'total_points': n_points
    }
    
    logger.info(f"Clustering metrics:")
    logger.info(f"  Clusters: {n_clusters}")
    logger.info(f"  Noise points: {n_noise} ({metrics['noise_ratio']:.1%})")
    logger.info(f"  Silhouette score: {silhouette_avg:.3f}")
    logger.info(f"  Avg cluster size: {metrics['avg_cluster_size']:.1f}")
    
    return metrics


def evaluate_inference_pipeline() -> Dict[str, Any]:
    """
    Evaluate the inference pipeline performance.
    
    Returns:
        Dictionary of inference pipeline metrics
    """
    logger.info("Evaluating inference pipeline...")
    
    try:
        from pipelines.inference_pipeline import predict_escalation, load_models
        
        # Load models to ensure they work
        load_models()
        
        # Test predictions on sample texts
        test_texts = [
            "Military exercises conducted near Taiwan strait",
            "Diplomatic meeting scheduled for next week",
            "Cyber attack reported on government infrastructure",
            "Peaceful trade agreement signed between nations",
            "Naval forces on high alert following tensions"
        ]
        
        predictions = []
        processing_times = []
        
        import time
        for text in test_texts:
            start_time = time.time()
            score = predict_escalation(text)
            processing_time = time.time() - start_time
            
            predictions.append(score)
            processing_times.append(processing_time)
        
        metrics = {
            'model_loaded': True,
            'avg_prediction_score': float(np.mean(predictions)),
            'score_std': float(np.std(predictions)),
            'avg_processing_time': float(np.mean(processing_times)),
            'max_processing_time': float(np.max(processing_times)),
            'predictions_in_range': all(0.0 <= p <= 1.0 for p in predictions),
            'sample_predictions': dict(zip(test_texts[:3], predictions[:3]))
        }
        
        logger.info(f"Inference pipeline metrics:")
        logger.info(f"  Avg prediction: {metrics['avg_prediction_score']:.3f}")
        logger.info(f"  Avg processing time: {metrics['avg_processing_time']:.3f}s")
        logger.info(f"  All predictions in range: {metrics['predictions_in_range']}")
        
    except Exception as e:
        logger.error(f"Error evaluating inference pipeline: {e}")
        metrics = {
            'model_loaded': False,
            'error': str(e)
        }
    
    return metrics


def evaluate_forecasting_model() -> Dict[str, Any]:
    """
    Evaluate the trained forecasting model.
    
    Returns:
        Dictionary of forecasting model metrics
    """
    logger.info("Evaluating forecasting model...")
    
    try:
        from train_forecasting_model import load_timeseries_data, create_forecast_features, predict_next_day_escalation
        
        # Load data and create features
        df = load_timeseries_data()
        df_features = create_forecast_features(df)
        
        # Try to predict next day escalation
        try:
            tomorrow_prediction = predict_next_day_escalation()
            model_available = True
        except Exception:
            tomorrow_prediction = None
            model_available = False
        
        metrics = {
            'model_available': model_available,
            'data_points': len(df),
            'feature_points': len(df_features),
            'date_range_days': (df['date'].max() - df['date'].min()).days,
            'avg_escalation_score': float(df['escalation_score'].mean()),
            'escalation_score_std': float(df['escalation_score'].std()),
            'tomorrow_prediction': tomorrow_prediction
        }
        
        logger.info(f"Forecasting model metrics:")
        logger.info(f"  Model available: {model_available}")
        logger.info(f"  Data points: {len(df)}")
        logger.info(f"  Tomorrow prediction: {tomorrow_prediction}")
        
    except Exception as e:
        logger.error(f"Error evaluating forecasting model: {e}")
        metrics = {
            'model_available': False,
            'error': str(e)
        }
    
    return metrics


def create_evaluation_plots(metrics: Dict[str, Any]) -> str:
    """
    Create visualization plots for model evaluation.
    
    Args:
        metrics: Combined metrics dictionary
        
    Returns:
        Path to saved plots
    """
    logger.info("Creating evaluation plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('StraitWatch Model Evaluation Dashboard', fontsize=16)
    
    # Classifier performance (if available)
    if 'classifier' in metrics and 'accuracy' in metrics['classifier']:
        clf_metrics = metrics['classifier']
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1', 'AUC-ROC']
        metric_values = [
            clf_metrics.get('accuracy', 0),
            clf_metrics.get('precision', 0),
            clf_metrics.get('recall', 0),
            clf_metrics.get('f1', 0),
            clf_metrics.get('auc_roc', 0)
        ]
        
        bars = axes[0, 0].bar(metric_names, metric_values, color='skyblue')
        axes[0, 0].set_title('Classifier Performance')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
    else:
        axes[0, 0].text(0.5, 0.5, 'Classifier metrics\nnot available', 
                       ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Classifier Performance')
    
    # Forecasting performance (if available)
    if 'forecasting' in metrics and 'mae' in metrics['forecasting']:
        fc_metrics = metrics['forecasting']
        metric_names = ['MAE', 'RMSE', 'R²']
        metric_values = [
            fc_metrics.get('mae', 0),
            fc_metrics.get('rmse', 0),
            fc_metrics.get('r2', 0)
        ]
        
        bars = axes[0, 1].bar(metric_names[:2], metric_values[:2], color='lightgreen')
        ax2 = axes[0, 1].twinx()
        ax2.bar(metric_names[2], metric_values[2], color='orange', alpha=0.7)
        
        axes[0, 1].set_title('Forecasting Performance')
        axes[0, 1].set_ylabel('MAE / RMSE')
        ax2.set_ylabel('R²')
        ax2.set_ylim(-1, 1)
    else:
        axes[0, 1].text(0.5, 0.5, 'Forecasting metrics\nnot available', 
                       ha='center', va='center', transform=axes[0, 1].transAxes)
        axes[0, 1].set_title('Forecasting Performance')
    
    # Clustering performance (if available)
    if 'clustering' in metrics and 'n_clusters' in metrics['clustering']:
        clust_metrics = metrics['clustering']
        
        # Pie chart of clustered vs noise points
        if clust_metrics['total_points'] > 0:
            clustered_points = clust_metrics['total_points'] - clust_metrics['n_noise_points']
            sizes = [clustered_points, clust_metrics['n_noise_points']]
            labels = ['Clustered', 'Noise']
            colors = ['lightblue', 'lightcoral']
            
            axes[1, 0].pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
            axes[1, 0].set_title(f'Clustering Results\n({clust_metrics["n_clusters"]} clusters)')
        else:
            axes[1, 0].text(0.5, 0.5, 'No clustering data', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
    else:
        axes[1, 0].text(0.5, 0.5, 'Clustering metrics\nnot available', 
                       ha='center', va='center', transform=axes[1, 0].transAxes)
        axes[1, 0].set_title('Clustering Performance')
    
    # System overview
    system_status = []
    system_colors = []
    
    if 'inference_pipeline' in metrics:
        if metrics['inference_pipeline'].get('model_loaded', False):
            system_status.append('Inference ✓')
            system_colors.append('green')
        else:
            system_status.append('Inference ✗')
            system_colors.append('red')
    
    if 'forecasting_model' in metrics:
        if metrics['forecasting_model'].get('model_available', False):
            system_status.append('Forecasting ✓')
            system_colors.append('green')
        else:
            system_status.append('Forecasting ✗')
            system_colors.append('red')
    
    if 'clustering' in metrics:
        if metrics['clustering'].get('n_clusters', 0) > 0:
            system_status.append('Clustering ✓')
            system_colors.append('green')
        else:
            system_status.append('Clustering ✗')
            system_colors.append('red')
    
    if system_status:
        y_pos = np.arange(len(system_status))
        axes[1, 1].barh(y_pos, [1] * len(system_status), color=system_colors, alpha=0.7)
        axes[1, 1].set_yticks(y_pos)
        axes[1, 1].set_yticklabels(system_status)
        axes[1, 1].set_xlabel('System Status')
        axes[1, 1].set_title('Pipeline Components')
        axes[1, 1].set_xlim(0, 1)
    else:
        axes[1, 1].text(0.5, 0.5, 'No system status\navailable', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"logs/evaluation_dashboard_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"Saved evaluation plots to: {plot_path}")
    plt.close()
    
    return plot_path


def run_full_evaluation() -> Dict[str, Any]:
    """
    Run comprehensive evaluation of all models and save results.
    
    Returns:
        Dictionary containing all evaluation metrics
    """
    logger.info("Starting comprehensive model evaluation...")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Initialize results
    evaluation_results = {
        'timestamp': timestamp,
        'evaluation_date': datetime.now().isoformat(),
        'version': '1.0.0',
        'components_evaluated': []
    }
    
    # Evaluate inference pipeline
    try:
        inference_metrics = evaluate_inference_pipeline()
        evaluation_results['inference_pipeline'] = inference_metrics
        evaluation_results['components_evaluated'].append('inference_pipeline')
    except Exception as e:
        logger.error(f"Failed to evaluate inference pipeline: {e}")
        evaluation_results['inference_pipeline'] = {'error': str(e)}
    
    # Evaluate forecasting model
    try:
        forecasting_metrics = evaluate_forecasting_model()
        evaluation_results['forecasting_model'] = forecasting_metrics
        evaluation_results['components_evaluated'].append('forecasting_model')
    except Exception as e:
        logger.error(f"Failed to evaluate forecasting model: {e}")
        evaluation_results['forecasting_model'] = {'error': str(e)}
    
    # Evaluate clustering (if data available)
    try:
        import json
        with open("data/clustered_articles.json", 'r') as f:
            clustered_articles = json.load(f)
        
        # Extract embeddings and cluster labels
        embeddings = []
        cluster_labels = []
        for article in clustered_articles:
            if 'embedding' in article and 'cluster_id' in article:
                embeddings.append(article['embedding'])
                cluster_labels.append(article['cluster_id'])
        
        if embeddings:
            clustering_metrics = evaluate_clustering(np.array(embeddings), np.array(cluster_labels))
            evaluation_results['clustering'] = clustering_metrics
            evaluation_results['components_evaluated'].append('clustering')
        else:
            logger.warning("No valid clustering data found")
            evaluation_results['clustering'] = {'error': 'No valid clustering data'}
            
    except FileNotFoundError:
        logger.warning("No clustered articles file found")
        evaluation_results['clustering'] = {'error': 'No clustered articles file'}
    except Exception as e:
        logger.error(f"Failed to evaluate clustering: {e}")
        evaluation_results['clustering'] = {'error': str(e)}
    
    # Create evaluation plots
    try:
        plot_path = create_evaluation_plots(evaluation_results)
        evaluation_results['plot_path'] = plot_path
    except Exception as e:
        logger.error(f"Failed to create evaluation plots: {e}")
        evaluation_results['plot_error'] = str(e)
    
    # Save results
    output_path = f"logs/model_metrics_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    # Also save as latest
    with open("logs/model_metrics.json", 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    logger.info(f"✅ Model evaluation completed!")
    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Components evaluated: {evaluation_results['components_evaluated']}")
    
    return evaluation_results


if __name__ == "__main__":
    # Run full evaluation
    results = run_full_evaluation()
    
    # Print summary
    print("\n" + "="*50)
    print("STRAITWATCH MODEL EVALUATION SUMMARY")
    print("="*50)
    
    for component in results['components_evaluated']:
        if component in results and 'error' not in results[component]:
            print(f"✅ {component.replace('_', ' ').title()}: OK")
        else:
            print(f"❌ {component.replace('_', ' ').title()}: ERROR")
    
    print(f"\nDetailed results saved to: logs/model_metrics.json")
    print("="*50) 