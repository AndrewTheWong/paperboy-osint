#!/usr/bin/env python3
"""
Comprehensive test suite for StraitWatch Phase 1 backend ML pipeline.
Tests all essential models: classifier, clustering, forecasting.
"""
import pytest
import numpy as np
import pandas as pd
import json
import os
import tempfile
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

# Import modules to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.inference_pipeline import predict_escalation, encode_tags, extract_tags_from_text
from pipelines.cluster_articles import cluster_articles
from tagging.tagging_pipeline import tag_articles


class TestInferencePipeline:
    """Test escalation prediction inference pipeline."""
    
    def test_extract_tags_from_text(self):
        """Test tag extraction from text."""
        test_text = "Military forces conduct naval exercises near Taiwan strait"
        tags = extract_tags_from_text(test_text)
        assert "military movement" in tags
        assert len(tags) >= 1
    
    def test_encode_tags(self):
        """Test tag encoding to feature vector."""
        test_text = "Diplomatic meeting scheduled for peace talks"
        features = encode_tags(test_text, confidence=0.8)
        assert features.shape == (1, 8)  # 7 tags + confidence
        assert features[0][-1] == 0.8  # confidence score
    
    @patch('pipelines.inference_pipeline.model')
    @patch('pipelines.inference_pipeline.load_models')
    def test_predict_escalation(self, mock_load, mock_model):
        """Test escalation prediction."""
        # Mock model prediction
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        score = predict_escalation("Military conflict escalates in region")
        assert 0.0 <= score <= 1.0
        assert score == 0.7
    
    def test_predict_escalation_invalid_input(self):
        """Test escalation prediction with invalid input."""
        assert predict_escalation("") == 0.0
        assert predict_escalation(None) == 0.0
        assert predict_escalation("short") == 0.0


class TestClusteringPipeline:
    """Test HDBSCAN clustering pipeline."""
    
    def create_sample_articles_with_embeddings(self, n_articles=10):
        """Create sample articles with embeddings for testing."""
        articles = []
        for i in range(n_articles):
            # Create random embeddings (384-dim for SBERT)
            embedding = np.random.rand(384).tolist()
            articles.append({
                'id': f'article_{i}',
                'title': f'Test Article {i}',
                'text': f'Content of test article {i}',
                'embedding': embedding
            })
        return articles
    
    def test_cluster_articles_with_embeddings(self):
        """Test clustering with valid embeddings."""
        articles = self.create_sample_articles_with_embeddings(20)
        
        clustered = cluster_articles(
            articles=articles,
            output_path=None,  # Don't save
            min_cluster_size=3
        )
        
        assert len(clustered) == 20
        # Check that cluster IDs are assigned
        for article in clustered:
            assert 'cluster_id' in article
            assert isinstance(article['cluster_id'], int)
    
    def test_cluster_articles_no_embeddings(self):
        """Test clustering with articles without embeddings."""
        articles = [{'id': 'test', 'title': 'Test', 'text': 'Test content'}]
        
        clustered = cluster_articles(
            articles=articles,
            output_path=None
        )
        
        assert len(clustered) == 1
        # Should return original articles unchanged
        assert clustered[0]['id'] == 'test'


class TestTaggingPipeline:
    """Test keyword + ML tagging pipeline."""
    
    def create_sample_articles(self):
        """Create sample articles for testing."""
        return [
            {
                'id': 'article_1',
                'title': 'Military Exercise in Taiwan Strait',
                'translated_text': 'Taiwan military conducts naval exercises in response to tensions'
            },
            {
                'id': 'article_2', 
                'title': 'Diplomatic Summit',
                'translated_text': 'Leaders meet for diplomatic talks on regional security'
            },
            {
                'id': 'article_3',
                'title': 'Cyber Security Alert',
                'translated_text': 'Government warns of increased cyber attacks on infrastructure'
            }
        ]
    
    def test_tag_articles(self):
        """Test tagging pipeline with sample articles."""
        articles = self.create_sample_articles()
        
        tagged = tag_articles(articles)
        
        assert len(tagged) == 3
        for article in tagged:
            assert 'tags' in article
            assert 'ml_tags' in article
            assert 'needs_review' in article
            assert isinstance(article['tags'], list)


class TestForecastingModel:
    """Test XGBoost time-series forecasting model."""
    
    def create_sample_timeseries_data(self):
        """Create sample time series data for testing."""
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        data = {
            'date': dates,
            'escalation_score': np.random.rand(100) * 0.5 + 0.25,  # Values between 0.25-0.75
            'gdelt_events': np.random.poisson(50, 100),
            'acled_events': np.random.poisson(10, 100)
        }
        return pd.DataFrame(data)
    
    def test_create_forecast_features(self):
        """Test feature engineering for forecasting."""
        from train_forecasting_model import create_forecast_features
        
        df = self.create_sample_timeseries_data()
        features_df = create_forecast_features(df)
        
        # Check lag features
        assert 'escalation_lag_1' in features_df.columns
        assert 'escalation_lag_2' in features_df.columns
        assert 'escalation_lag_3' in features_df.columns
        
        # Check rolling features
        assert 'escalation_7d_mean' in features_df.columns
        assert 'escalation_14d_mean' in features_df.columns
        
        # Check day of week
        assert 'day_of_week' in features_df.columns
    
    @patch('train_forecasting_model.xgb.XGBRegressor')
    def test_train_forecasting_model(self, mock_xgb):
        """Test forecasting model training."""
        from train_forecasting_model import train_forecasting_model, create_forecast_features
        from sklearn.model_selection import TimeSeriesSplit
        
        df = self.create_sample_timeseries_data()
        df_features = create_forecast_features(df)
        
        # Simulate the TimeSeriesSplit logic used in the actual function
        X = df_features[['escalation_lag_1', 'escalation_lag_2', 'escalation_lag_3',
                        'escalation_7d_mean', 'escalation_14d_mean',
                        'gdelt_events', 'acled_events', 'gdelt_lag_1', 'acled_lag_1',
                        'gdelt_7d_mean', 'acled_7d_mean',
                        'day_of_week', 'month', 'day_of_year']].fillna(0)
        
        tscv = TimeSeriesSplit(n_splits=3)
        train_idx, test_idx = list(tscv.split(X))[-1]  # Use last split like the real function
        expected_test_size = len(test_idx)
        
        # Mock XGBoost model with correct prediction size
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.rand(expected_test_size)
        # Mock feature importances (need 14 features based on the actual function)
        mock_model.feature_importances_ = np.random.rand(14)
        mock_xgb.return_value = mock_model
        
        model, metrics = train_forecasting_model(df_features)
        
        assert model is not None
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics


class TestModelEvaluation:
    """Test model evaluation metrics."""
    
    def test_evaluate_classifier(self):
        """Test classifier evaluation metrics."""
        from evaluate_models import evaluate_classifier
        
        # Sample predictions
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 1, 1, 1, 0, 0])
        y_prob = np.array([0.1, 0.6, 0.8, 0.9, 0.2, 0.4])
        
        metrics = evaluate_classifier(y_true, y_pred, y_prob)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc_roc' in metrics
        assert 0 <= metrics['accuracy'] <= 1
    
    def test_evaluate_forecasting(self):
        """Test forecasting model evaluation."""
        from evaluate_models import evaluate_forecasting
        
        y_true = np.array([0.5, 0.6, 0.4, 0.7, 0.3])
        y_pred = np.array([0.48, 0.62, 0.38, 0.72, 0.28])
        
        metrics = evaluate_forecasting(y_true, y_pred)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'r2' in metrics
        assert metrics['mae'] >= 0
        assert metrics['rmse'] >= 0


class TestBackendPipeline:
    """Test unified backend pipeline."""
    
    def create_sample_translated_articles(self):
        """Create sample translated articles."""
        return [
            {
                'id': 'article_1',
                'title': 'Taiwan Military Exercise',
                'translated_text': 'Taiwan conducts large-scale military exercise',
                'source': 'test_source'
            },
            {
                'id': 'article_2',
                'title': 'Regional Diplomacy',
                'translated_text': 'Diplomatic meeting between regional powers',
                'source': 'test_source'
            }
        ]
    
    @patch('tagging.tagging_pipeline.tag_articles')
    @patch('pipelines.inference_pipeline.predict_escalation')
    @patch('sentence_transformers.SentenceTransformer')
    @patch('pipelines.cluster_articles.cluster_articles')
    def test_run_full_pipeline(self, mock_cluster, mock_sbert, mock_predict, mock_tag):
        """Test complete backend pipeline execution."""
        # Setup mocks
        mock_tag.return_value = self.create_sample_translated_articles()
        mock_predict.return_value = 0.7
        mock_encoder = Mock()
        mock_encoder.encode.return_value = np.random.rand(2, 384)
        mock_sbert.return_value = mock_encoder
        mock_cluster.return_value = self.create_sample_translated_articles()
        
        from run_backend_pipeline import run_full_pipeline
        
        # Run pipeline
        results = run_full_pipeline(
            input_path=None,
            articles=self.create_sample_translated_articles()
        )
        
        assert 'processed_articles' in results
        assert 'cluster_summary' in results
        assert 'pipeline_stats' in results
        assert len(results['processed_articles']) == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 