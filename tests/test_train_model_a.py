#!/usr/bin/env python3
"""
Test suite for Model A - XGBoost Escalation Classifier

This test suite validates all components of Model A including:
- Data loading and validation
- Feature engineering
- Model training and evaluation
- Prediction functionality
- Model persistence

Author: AI Assistant
Date: 2025
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from train_model_a import ModelA, main, run_tests


class TestModelA(unittest.TestCase):
    """Test cases for Model A"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        self.model_a = ModelA(test_size=0.3, random_state=42)
        
        # Create sample test data
        self.sample_conflict_data = pd.DataFrame({
            'text': [
                'Armed conflict between government forces and rebels',
                'Military operation in the northern region',
                'Violence against civilians reported',
                'Terrorist attack in major city',
                'Cross-border clashes continue'
            ],
            'label': [1, 1, 1, 1, 1],
            'source': ['test_conflict'] * 5
        })
        
        self.sample_peaceful_data = pd.DataFrame({
            'text': [
                'Diplomatic talks between nations',
                'Peace treaty signed',
                'Cultural exchange program launched',
                'Trade agreement reached',
                'International cooperation on climate'
            ],
            'label': [0, 0, 0, 0, 0],
            'source': ['test_peaceful'] * 5
        })
        
        self.combined_data = pd.concat([self.sample_conflict_data, self.sample_peaceful_data], 
                                      ignore_index=True)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_model_initialization(self):
        """Test ModelA initialization"""
        model = ModelA(model_name="test-model", test_size=0.25, random_state=123)
        
        self.assertEqual(model.model_name, "test-model")
        self.assertEqual(model.test_size, 0.25)
        self.assertEqual(model.random_state, 123)
        self.assertIsNone(model.model)
        self.assertIsNone(model.encoder_model)
        self.assertEqual(len(model.label_encoders), 0)
        self.assertEqual(len(model.metrics), 0)
    
    @patch('train_model_a.pd.read_csv')
    @patch('train_model_a.os.path.exists')
    def test_load_data_success(self, mock_exists, mock_read_csv):
        """Test successful data loading"""
        # Mock file existence - need to handle multiple file checks
        mock_exists.return_value = True
        
        # Mock CSV reading - need to handle all possible file reads
        def read_csv_side_effect(filepath):
            if 'all_conflict_data.csv' in filepath:
                return self.sample_conflict_data
            elif 'all_peaceful_events.csv' in filepath:
                return self.sample_peaceful_data
            else:
                # Return empty dataframe for other files
                return pd.DataFrame({'text': [], 'label': []})
        
        mock_read_csv.side_effect = read_csv_side_effect
        
        df = self.model_a.load_data()
        
        self.assertEqual(len(df), 10)
        self.assertIn('text', df.columns)
        self.assertIn('label', df.columns)
        self.assertIn('source', df.columns)
        self.assertEqual(df['label'].sum(), 5)  # 5 conflict, 5 peaceful
    
    @patch('train_model_a.os.path.exists')
    def test_load_data_no_files(self, mock_exists):
        """Test data loading when no files exist"""
        mock_exists.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            self.model_a.load_data()
    
    @patch('train_model_a.SentenceTransformer')
    def test_create_features(self, mock_transformer):
        """Test feature creation"""
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(10, 384)  # Mock embeddings
        mock_transformer.return_value = mock_encoder
        
        X, y = self.model_a.create_features(self.combined_data)
        
        # Check shapes
        self.assertEqual(X.shape[0], 10)
        self.assertEqual(len(y), 10)
        self.assertTrue(X.shape[1] > 384)  # Embeddings + additional features
        
        # Check labels
        self.assertEqual(y.sum(), 5)  # 5 conflict events
        
        # Check that encoder model is set
        self.assertIsNotNone(self.model_a.encoder_model)
    
    @patch('train_model_a.SentenceTransformer')
    def test_train_model(self, mock_transformer):
        """Test model training"""
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(10, 384)
        mock_transformer.return_value = mock_encoder
        
        # Create features
        X, y = self.model_a.create_features(self.combined_data)
        
        # Train model
        self.model_a.train_model(X, y)
        
        # Check that model is trained
        self.assertIsNotNone(self.model_a.model)
        self.assertIsNotNone(self.model_a.X_test)
        self.assertIsNotNone(self.model_a.y_test)
        
        # Check test set size
        expected_test_size = int(len(X) * self.model_a.test_size)
        self.assertEqual(len(self.model_a.X_test), expected_test_size)
    
    @patch('train_model_a.SentenceTransformer')
    def test_evaluate_model(self, mock_transformer):
        """Test model evaluation"""
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(10, 384)
        mock_transformer.return_value = mock_encoder
        
        # Create features and train model
        X, y = self.model_a.create_features(self.combined_data)
        self.model_a.train_model(X, y)
        
        # Evaluate model
        metrics = self.model_a.evaluate_model()
        
        # Check metrics
        expected_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
        for metric in expected_metrics:
            self.assertIn(metric, metrics)
            self.assertIsInstance(metrics[metric], (int, float))
            self.assertGreaterEqual(metrics[metric], 0)
            self.assertLessEqual(metrics[metric], 1)
    
    def test_evaluate_model_not_trained(self):
        """Test evaluation without training"""
        with self.assertRaises(ValueError):
            self.model_a.evaluate_model()
    
    @patch('train_model_a.SentenceTransformer')
    def test_save_model(self, mock_transformer):
        """Test model saving"""
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(10, 384)
        mock_transformer.return_value = mock_encoder
        
        # Create features, train and evaluate model
        X, y = self.model_a.create_features(self.combined_data)
        self.model_a.train_model(X, y)
        self.model_a.evaluate_model()
        
        # Save model
        self.model_a.save_model(self.test_dir)
        
        # Check files exist
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "xgb_model_a.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "model_a_complete.pkl")))
        self.assertTrue(os.path.exists(os.path.join(self.test_dir, "model_a_metrics.txt")))
        
        # Check metrics file content
        with open(os.path.join(self.test_dir, "model_a_metrics.txt"), 'r') as f:
            content = f.read()
            self.assertIn("Model A - XGBoost Escalation Classifier", content)
            self.assertIn("Accuracy:", content)
    
    @patch('train_model_a.SentenceTransformer')
    def test_predict(self, mock_transformer):
        """Test prediction functionality"""
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(10, 384)
        mock_transformer.return_value = mock_encoder
        
        # Create features and train model
        X, y = self.model_a.create_features(self.combined_data)
        self.model_a.train_model(X, y)
        
        # Test prediction
        test_texts = ["Peaceful diplomatic meeting", "Armed conflict reported"]
        
        # Mock encoder for prediction
        mock_encoder.encode.return_value = np.random.rand(2, 384)
        
        predictions, probabilities = self.model_a.predict(test_texts)
        
        # Check outputs
        self.assertEqual(len(predictions), 2)
        self.assertEqual(len(probabilities), 2)
        self.assertEqual(probabilities.shape, (2, 2))  # 2 samples, 2 classes
        
        # Check prediction values
        self.assertTrue(all(p in [0, 1] for p in predictions))
        self.assertTrue(all(0 <= prob <= 1 for prob_row in probabilities for prob in prob_row))
    
    def test_predict_not_trained(self):
        """Test prediction without training"""
        test_texts = ["Some text"]
        
        with self.assertRaises(ValueError):
            self.model_a.predict(test_texts)
    
    def test_binary_label_conversion(self):
        """Test conversion of non-binary labels to binary"""
        # Create data with non-binary labels
        data_with_multiclass = pd.DataFrame({
            'text': ['text1', 'text2', 'text3', 'text4'],
            'label': [0, 1, 2, 3],  # Non-binary labels
            'source': ['test'] * 4
        })
        
        def read_csv_side_effect(filepath):
            if 'all_conflict_data.csv' in filepath:
                return data_with_multiclass
            else:
                # Return empty dataframe for other files
                return pd.DataFrame({'text': [], 'label': []})
        
        with patch('train_model_a.os.path.exists', return_value=True), \
             patch('train_model_a.pd.read_csv', side_effect=read_csv_side_effect):
            
            df = self.model_a.load_data()
            
            # Check that labels are converted to binary
            unique_labels = df['label'].unique()
            self.assertTrue(all(label in [0, 1] for label in unique_labels))
            # Since only one file has data, we should get 4 samples
            self.assertEqual(len(df), 4)
            self.assertEqual(df['label'].tolist(), [0, 1, 1, 1])  # 0 stays 0, >0 becomes 1


class TestModelAPipeline(unittest.TestCase):
    """Test the complete Model A pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_dir = tempfile.mkdtemp()
        
        # Create test data files
        self.create_test_data_files()
    
    def create_test_data_files(self):
        """Create test data files"""
        os.makedirs(os.path.join(self.test_dir, 'data'), exist_ok=True)
        
        # Create conflict data
        conflict_data = pd.DataFrame({
            'text': [
                'Armed conflict between government and rebels',
                'Military operation launched in northern region',
                'Violence against civilians reported in city',
                'Terrorist attack hits government building',
                'Cross-border clashes between forces'
            ] * 20,  # Repeat for more data
            'label': [1] * 100,
            'source': ['test_conflict'] * 100
        })
        conflict_data.to_csv(os.path.join(self.test_dir, 'data', 'all_conflict_data.csv'), index=False)
        
        # Create peaceful data
        peaceful_data = pd.DataFrame({
            'text': [
                'Diplomatic talks between nations successful',
                'Peace treaty signed by both parties',
                'Cultural exchange program launched',
                'Trade agreement benefits both countries',
                'International cooperation on climate change'
            ] * 20,  # Repeat for more data
            'label': [0] * 100,
            'source': ['test_peaceful'] * 100
        })
        peaceful_data.to_csv(os.path.join(self.test_dir, 'data', 'all_peaceful_events.csv'), index=False)
    
    def tearDown(self):
        """Clean up test fixtures"""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    @patch('train_model_a.SentenceTransformer')
    def test_full_pipeline(self, mock_transformer):
        """Test the complete training pipeline"""
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(200, 384)  # 200 samples
        mock_transformer.return_value = mock_encoder
        
        # Change to test directory
        original_cwd = os.getcwd()
        os.chdir(self.test_dir)
        
        try:
            # Initialize and run pipeline
            model_a = ModelA(test_size=0.2, random_state=42)
            
            # Load and prepare data
            df = model_a.load_data()
            self.assertEqual(len(df), 200)  # 100 conflict + 100 peaceful
            
            # Create features
            X, y = model_a.create_features(df)
            self.assertEqual(X.shape[0], 200)
            
            # Train model
            model_a.train_model(X, y)
            self.assertIsNotNone(model_a.model)
            
            # Evaluate model
            metrics = model_a.evaluate_model()
            self.assertIn('accuracy', metrics)
            
            # Save model
            model_a.save_model()
            self.assertTrue(os.path.exists('models/xgb_model_a.pkl'))
            
        finally:
            os.chdir(original_cwd)


class TestModelAEdgeCases(unittest.TestCase):
    """Test edge cases and error conditions"""
    
    def test_empty_data(self):
        """Test handling of empty data"""
        model_a = ModelA()
        
        empty_df = pd.DataFrame({'text': [], 'label': [], 'source': []})
        
        with patch('train_model_a.os.path.exists', return_value=True), \
             patch('train_model_a.pd.read_csv', return_value=empty_df):
            
            with self.assertRaises(FileNotFoundError):  # Should fail when no valid data found
                model_a.load_data()
    
    def test_missing_columns(self):
        """Test handling of data with missing columns"""
        model_a = ModelA()
        
        # Data missing 'label' column
        bad_df = pd.DataFrame({'text': ['some text'], 'other_col': [1]})
        
        with patch('train_model_a.os.path.exists', return_value=True), \
             patch('train_model_a.pd.read_csv', return_value=bad_df):
            
            with self.assertRaises(FileNotFoundError):  # No valid files
                model_a.load_data()
    
    @patch('train_model_a.SentenceTransformer')
    def test_single_class_data(self, mock_transformer):
        """Test handling of data with only one class"""
        model_a = ModelA()
        
        # Data with only one class
        single_class_data = pd.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': [1, 1, 1],  # All same class
            'source': ['test'] * 3
        })
        
        # Mock sentence transformer
        mock_encoder = MagicMock()
        mock_encoder.encode.return_value = np.random.rand(3, 384)
        mock_transformer.return_value = mock_encoder
        
        X, y = model_a.create_features(single_class_data)
        
        # Training should handle single class gracefully
        with self.assertRaises(ValueError):  # Stratified split fails with single class
            model_a.train_model(X, y)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2) 