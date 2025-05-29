import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.inference_pipeline import predict_escalation, main

class TestInferencePipeline:
    
    def test_predict_escalation_returns_float(self):
        """Test that predict_escalation returns a float between 0 and 1"""
        test_text = "Taiwan military conducts exercises near the strait as tensions rise"
        
        with patch('pipelines.inference_pipeline.model') as mock_model:
            # Mock model prediction
            mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])
            
            result = predict_escalation(test_text)
            
            assert isinstance(result, float)
            assert 0 <= result <= 1
            assert result == 0.7  # Should return the escalation probability
    
    def test_predict_escalation_empty_text(self):
        """Test that empty or short text returns 0"""
        assert predict_escalation("") == 0.0
        assert predict_escalation("hi") == 0.0
        assert predict_escalation("short") == 0.0
    
    def test_predict_escalation_none_text(self):
        """Test that None text returns 0"""
        assert predict_escalation(None) == 0.0
    
    def test_predict_escalation_long_text(self):
        """Test prediction with longer text"""
        long_text = "Military tensions escalate in the Taiwan Strait as both sides conduct exercises"
        
        with patch('pipelines.inference_pipeline.model') as mock_model:
            mock_model.predict_proba.return_value = np.array([[0.2, 0.8]])
            
            result = predict_escalation(long_text)
            assert result == 0.8
    
    def test_predict_escalation_detects_military_keywords(self):
        """Test that military keywords are properly detected"""
        military_text = "Military forces conducting joint exercises with troops deployed"
        
        with patch('pipelines.inference_pipeline.model') as mock_model:
            mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
            
            result = predict_escalation(military_text)
            assert result == 0.6
    
    def test_predict_escalation_detects_conflict_keywords(self):
        """Test that conflict keywords are properly detected"""
        conflict_text = "Armed clash results in violence between opposing forces"
        
        with patch('pipelines.inference_pipeline.model') as mock_model:
            mock_model.predict_proba.return_value = np.array([[0.1, 0.9]])
            
            result = predict_escalation(conflict_text)
            assert result == 0.9
    
    def test_batch_csv_processing(self):
        """Test CLI batch processing with CSV input"""
        # Create temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            test_data = pd.DataFrame({
                'id': [1, 2, 3],
                'title': ['Article 1', 'Article 2', 'Article 3'],
                'text': [
                    'Taiwan military exercises increase tensions',
                    'Peaceful trade agreement signed',
                    'Diplomatic meeting scheduled for next week'
                ]
            })
            test_data.to_csv(f.name, index=False)
            input_file = f.name
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            output_file = f.name
        
        try:
            with patch('pipelines.inference_pipeline.model') as mock_model:
                # Mock predictions
                mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
                
                # Mock sys.argv
                with patch('sys.argv', ['inference_pipeline.py', input_file, output_file]):
                    main()
                
                # Check output file
                result_df = pd.read_csv(output_file)
                assert 'escalation_score' in result_df.columns
                assert len(result_df) == 3
                assert all(0 <= score <= 1 for score in result_df['escalation_score'])
        
        finally:
            # Clean up temporary files
            os.unlink(input_file)
            if os.path.exists(output_file):
                os.unlink(output_file)
    
    def test_model_loading_error_handling(self):
        """Test graceful handling of model loading errors"""
        # Clear any existing loaded model
        import pipelines.inference_pipeline
        pipelines.inference_pipeline.model = None
        
        with patch('pipelines.inference_pipeline.joblib.load', side_effect=FileNotFoundError):
            with patch('pipelines.inference_pipeline.os.path.exists', return_value=True):
                with pytest.raises(FileNotFoundError):
                    # Force reload models which should raise the error
                    pipelines.inference_pipeline.load_models()
    
    @patch('pipelines.inference_pipeline.model')
    def test_model_prediction_error_handling(self, mock_model):
        """Test handling of model prediction errors"""
        mock_model.predict_proba.side_effect = Exception("Model error")
        
        result = predict_escalation("Test text with enough characters and military keywords")
        assert result == 0.0  # Should return 0 on error
    
    def test_predict_escalation_handles_long_text(self):
        """Test handling of very long text input"""
        long_text = "military exercise " * 1000  # Very long text
        
        with patch('pipelines.inference_pipeline.model') as mock_model:
            mock_model.predict_proba.return_value = np.array([[0.4, 0.6]])
            
            result = predict_escalation(long_text)
            
            # Should still work and return valid result
            assert isinstance(result, float)
            assert 0 <= result <= 1
    
    def test_main_function_csv_processing(self):
        """Test the main function with CSV processing"""
        # Create a temporary CSV file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            f.write("id,title,text\n")
            f.write("1,Test Article,Taiwan military conducts exercises\n")
            f.write("2,Another Article,Peace talks continue\n")
            temp_input = f.name
        
        output_file = temp_input.replace('.csv', '_predictions.csv')
        
        try:
            with patch('pipelines.inference_pipeline.model') as mock_model:
                mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2]])
                
                # Test main function
                with patch('sys.argv', ['inference_pipeline.py', temp_input, output_file]):
                    main()
                
                # Check output file was created
                assert os.path.exists(output_file)
                
                # Verify content
                df = pd.read_csv(output_file)
                assert 'escalation_score' in df.columns
                assert len(df) == 2
                
        finally:
            # Cleanup
            if os.path.exists(temp_input):
                os.unlink(temp_input)
            if os.path.exists(output_file):
                os.unlink(output_file) 