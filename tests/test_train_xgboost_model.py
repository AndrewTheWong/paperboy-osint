import pytest
import pandas as pd
import numpy as np
import os
import tempfile
import shutil
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pipelines.train_xgboost_model import (
    load_training_data,
    encode_texts_with_sbert,
    train_xgboost_model,
    evaluate_model,
    save_model_and_metrics,
    generate_robust_acled_data,
    generate_robust_gdelt_data,
    generate_robust_peaceful_data,
    create_training_datasets,
    main
)


class TestTrainXGBoostModel:
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test files."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        return {
            'acled': pd.DataFrame({
                'text': ['Conflict escalated in region A', 'Violence increased dramatically'],
                'label': [1, 1]
            }),
            'gdelt': pd.DataFrame({
                'text': ['Protests turned violent', 'Armed conflict erupted'],
                'label': [1, 1]
            }),
            'peaceful': pd.DataFrame({
                'text': ['Peaceful negotiations continued', 'Diplomatic talks progressed'],
                'label': [0, 0]
            })
        }
    
    @pytest.fixture
    def sample_csv_files(self, temp_dir, sample_data):
        """Create sample CSV files."""
        data_dir = os.path.join(temp_dir, 'data', 'training')
        os.makedirs(data_dir, exist_ok=True)
        
        files = {}
        for name, df in sample_data.items():
            file_path = os.path.join(data_dir, f'{name}_labeled.csv')
            df.to_csv(file_path, index=False)
            files[name] = file_path
        
        return files, data_dir
    
    def test_generate_robust_acled_data(self):
        """Test generation of robust ACLED data."""
        acled_df = generate_robust_acled_data(100)  # Small test sample
        
        # Check basic structure
        assert len(acled_df) == 100
        assert 'text' in acled_df.columns
        assert 'label' in acled_df.columns
        assert all(acled_df['label'] == 1)  # All should be conflict
        assert all(acled_df['source'] == 'generated_acled')
        
        # Check text quality
        assert all(isinstance(text, str) and len(text) > 20 for text in acled_df['text'])
        
        # Check that most texts contain conflict indicators (at least 75%)
        conflict_indicators = ['casualties', 'killed', 'fighting', 'attack', 'violence', 'conflict', 
                              'forces', 'military', 'bombing', 'explosion', 'clash', 'dead', 'wounded']
        texts_with_indicators = sum(1 for text in acled_df['text'].head(20) 
                                   if any(indicator in text.lower() for indicator in conflict_indicators))
        assert texts_with_indicators >= 15  # At least 75% of 20 samples
        
        print(f"✅ Generated {len(acled_df)} ACLED samples")
    
    def test_generate_robust_gdelt_data(self):
        """Test generation of robust GDELT data."""
        gdelt_df = generate_robust_gdelt_data(100)  # Small test sample
        
        # Check basic structure
        assert len(gdelt_df) == 100
        assert 'text' in gdelt_df.columns
        assert 'label' in gdelt_df.columns
        assert all(gdelt_df['label'] == 1)  # All should be conflict
        assert all(gdelt_df['source'] == 'generated_gdelt')
        
        # Check text quality
        assert all(isinstance(text, str) and len(text) > 20 for text in gdelt_df['text'])
        
        # Check that most texts contain international conflict indicators
        conflict_indicators = ['tensions', 'crisis', 'conflict', 'forces', 'military', 'attack', 
                              'between', 'clashes', 'violence', 'against', 'operations']
        texts_with_indicators = sum(1 for text in gdelt_df['text'].head(20) 
                                   if any(indicator in text.lower() for indicator in conflict_indicators))
        assert texts_with_indicators >= 15  # At least 75% of 20 samples
        
        print(f"✅ Generated {len(gdelt_df)} GDELT samples")
    
    def test_generate_robust_peaceful_data(self):
        """Test generation of robust peaceful data."""
        peaceful_df = generate_robust_peaceful_data(100)  # Small test sample
        
        # Check basic structure
        assert len(peaceful_df) == 100
        assert 'text' in peaceful_df.columns
        assert 'label' in peaceful_df.columns
        assert all(peaceful_df['label'] == 0)  # All should be peaceful
        assert all(peaceful_df['source'] == 'generated_peaceful')
        
        # Check text quality
        assert all(isinstance(text, str) and len(text) > 20 for text in peaceful_df['text'])
        
        # Check that most texts contain peaceful indicators
        peaceful_indicators = ['cooperation', 'agreement', 'partnership', 'aid', 'development', 
                              'peace', 'diplomatic', 'trade', 'cultural', 'scientific', 'humanitarian']
        texts_with_indicators = sum(1 for text in peaceful_df['text'].head(20) 
                                   if any(indicator in text.lower() for indicator in peaceful_indicators))
        assert texts_with_indicators >= 15  # At least 75% of 20 samples
        
        print(f"✅ Generated {len(peaceful_df)} peaceful samples")
    
    def test_create_training_datasets(self, temp_dir):
        """Test creation of training datasets."""
        # Change to temp directory to avoid creating files in real project
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Patch the functions to generate smaller datasets for testing
            with patch('pipelines.train_xgboost_model.generate_robust_acled_data') as mock_acled, \
                 patch('pipelines.train_xgboost_model.generate_robust_gdelt_data') as mock_gdelt, \
                 patch('pipelines.train_xgboost_model.generate_robust_peaceful_data') as mock_peaceful:
                
                # Setup mock returns
                mock_acled.return_value = pd.DataFrame({
                    'text': ['Test ACLED conflict'] * 10,
                    'label': [1] * 10
                })
                mock_gdelt.return_value = pd.DataFrame({
                    'text': ['Test GDELT conflict'] * 10,
                    'label': [1] * 10
                })
                mock_peaceful.return_value = pd.DataFrame({
                    'text': ['Test peaceful event'] * 10,
                    'label': [0] * 10
                })
                
                # Run function
                create_training_datasets()
                
                # Check files were created
                assert os.path.exists('data/training/acled_labeled.csv')
                assert os.path.exists('data/training/gdelt_labeled.csv')
                assert os.path.exists('data/training/peaceful_labeled.csv')
                
                # Verify content
                acled_df = pd.read_csv('data/training/acled_labeled.csv')
                assert len(acled_df) == 10
                assert all(acled_df['label'] == 1)
                
                print("✅ Training datasets created successfully")
        
        finally:
            os.chdir(original_cwd)
    
    def test_load_training_data(self, sample_csv_files):
        """Test loading training data from CSV files."""
        files, data_dir = sample_csv_files
        
        combined_df = load_training_data(data_dir)
        
        assert len(combined_df) == 6  # 2 + 2 + 2
        assert 'text' in combined_df.columns
        assert 'label' in combined_df.columns
        assert combined_df['label'].nunique() == 2  # 0 and 1
        assert combined_df['label'].sum() == 4  # 4 conflict labels
        
        print("✅ Training data loaded successfully")
    
    def test_load_training_data_with_generation(self, temp_dir):
        """Test loading training data with automatic generation."""
        # Change to temp directory
        original_cwd = os.getcwd()
        os.chdir(temp_dir)
        
        try:
            # Patch generation functions to create small test datasets
            with patch('pipelines.train_xgboost_model.generate_robust_acled_data') as mock_acled, \
                 patch('pipelines.train_xgboost_model.generate_robust_gdelt_data') as mock_gdelt, \
                 patch('pipelines.train_xgboost_model.generate_robust_peaceful_data') as mock_peaceful:
                
                mock_acled.return_value = pd.DataFrame({
                    'text': ['ACLED conflict event'] * 5,
                    'label': [1] * 5
                })
                mock_gdelt.return_value = pd.DataFrame({
                    'text': ['GDELT conflict event'] * 5,
                    'label': [1] * 5
                })
                mock_peaceful.return_value = pd.DataFrame({
                    'text': ['Peaceful cooperation event'] * 5,
                    'label': [0] * 5
                })
                
                # Load data (should trigger generation)
                combined_df = load_training_data()
                
                # Verify combined dataset
                assert len(combined_df) == 15  # 5 + 5 + 5
                assert combined_df['label'].sum() == 10  # 10 conflict events
                assert (combined_df['label'] == 0).sum() == 5  # 5 peaceful events
                
                print("✅ Training data loaded with auto-generation")
        
        finally:
            os.chdir(original_cwd)
    
    @patch('pipelines.train_xgboost_model.SentenceTransformer')
    def test_encode_texts_with_sbert(self, mock_transformer, sample_data):
        """Test text encoding with SBERT."""
        # Mock the SentenceTransformer
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(6, 384)  # 6 texts, 384 dimensions
        mock_transformer.return_value = mock_model
        
        combined_df = pd.concat(list(sample_data.values()), ignore_index=True)
        texts = combined_df['text'].tolist()
        
        embeddings = encode_texts_with_sbert(texts)
        
        assert embeddings.shape[0] == len(texts)
        assert embeddings.shape[1] == 384  # Expected dimension
        mock_transformer.assert_called_once_with('all-MiniLM-L6-v2')
        mock_model.encode.assert_called_once()
        
        print("✅ Text encoding test passed")
    
    def test_train_xgboost_model(self):
        """Test XGBoost model training."""
        # Create sample embeddings and labels
        X = np.random.rand(100, 384)
        y = np.random.randint(0, 2, 100)
        
        model, X_test, y_test, y_pred = train_xgboost_model(X, y)
        
        assert model is not None
        assert len(X_test) == int(0.2 * len(X))  # 20% test split
        assert len(y_test) == len(X_test)
        assert len(y_pred) == len(y_test)
        assert all(pred in [0, 1] for pred in y_pred)
        
        print("✅ XGBoost training test passed")
    
    def test_evaluate_model(self):
        """Test model evaluation metrics calculation."""
        y_test = np.array([0, 1, 0, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = evaluate_model(y_test, y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1_score' in metrics
        assert all(0 <= v <= 1 for v in metrics.values())
        
        print("✅ Model evaluation test passed")
    
    def test_save_model_and_metrics(self, temp_dir):
        """Test saving model and metrics to files."""
        # Create mock model
        from xgboost import XGBClassifier
        model = XGBClassifier()
        
        metrics = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1_score': 0.85
        }
        
        models_dir = os.path.join(temp_dir, 'models')
        logs_dir = os.path.join(temp_dir, 'logs')
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(logs_dir, exist_ok=True)
        
        save_model_and_metrics(model, metrics, models_dir, logs_dir)
        
        # Check model file exists
        model_path = os.path.join(models_dir, 'escalation_model.pkl')
        assert os.path.exists(model_path)
        
        # Check metrics file exists and contains expected content
        metrics_path = os.path.join(logs_dir, 'escalation_classifier_metrics.txt')
        assert os.path.exists(metrics_path)
        
        with open(metrics_path, 'r') as f:
            content = f.read()
            assert 'Accuracy: 0.85' in content
            assert 'Precision: 0.82' in content
            assert 'Recall: 0.88' in content
            assert 'F1 Score: 0.85' in content
            assert '15,000' in content  # Total training samples
        
        print("✅ Model and metrics saving test passed")


if __name__ == '__main__':
    pytest.main([__file__, '-v']) 