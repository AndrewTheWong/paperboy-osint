import os
import unittest
import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path
from fetch_peaceful_events import (
    extract_peaceful_gdelt_events,
    create_synthetic_peaceful_events,
    create_peaceful_dataset,
    combine_with_conflict_data,
    prepare_balanced_model_data
)
from fetch_and_clean_data import TAG_VOCAB

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestPeacefulEvents(unittest.TestCase):
    """Test the peaceful events data processing."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = Path("test_data_peaceful")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample data for testing
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample GDELT data for testing."""
        # Sample GDELT data with both conflict and non-conflict events
        self.gdelt_data = pd.DataFrame({
            "EventCode": ["190", "010", "195", "020", "080", "040"],
            "GoldsteinScale": [5.0, -2.0, 8.0, 0.0, 3.0, 1.0],
            "actor1": ["USA", "RUS", "CHN", "JPN", "FRA", "DEU"],
            "actor2": ["IRN", "UKR", "TWN", "KOR", "ITA", "ESP"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"]
        })
        
        # Save to test file
        os.makedirs(self.test_dir / "data", exist_ok=True)
        self.gdelt_data.to_csv(self.test_dir / "data/gdelt_events.csv", index=False)
    
    def test_extract_peaceful_gdelt_events(self):
        """Test extraction of peaceful events from GDELT data."""
        # Mock the data loading to use our test data
        from unittest.mock import patch
        with patch('fetch_peaceful_events.pd.read_csv', return_value=self.gdelt_data):
            peaceful_df = extract_peaceful_gdelt_events(force=True)
        
        # Check that only non-conflict events are extracted
        self.assertEqual(len(peaceful_df), 4)  # Should have 4 non-conflict events (EventCode not starting with "19")
        
        # Check label and source
        self.assertTrue(all(peaceful_df["label"] == 0))
        self.assertTrue(all(peaceful_df["source"] == "gdelt_peaceful"))
        
        # Check text generation
        self.assertTrue(all("Interaction between" in text for text in peaceful_df["text"]))
        
        logger.info("[PASS] Peaceful GDELT event extraction test passed")
    
    def test_create_synthetic_peaceful_events(self):
        """Test creation of synthetic peaceful events."""
        synthetic_df = create_synthetic_peaceful_events()
        
        # Check data
        self.assertGreater(len(synthetic_df), 0)
        self.assertTrue(all(synthetic_df["label"] == 0))
        self.assertTrue(all(synthetic_df["source"] == "synthetic_peaceful"))
        
        # Check confidence values
        self.assertTrue(all(0 <= conf <= 1 for conf in synthetic_df["confidence"]))
        
        logger.info("[PASS] Synthetic peaceful events creation test passed")
    
    def test_create_peaceful_dataset(self):
        """Test creation of combined peaceful dataset with tag encoding."""
        # Mock the GDELT extraction to use our test data
        from unittest.mock import patch
        
        # Prepare mock return values
        mock_gdelt_peaceful = pd.DataFrame({
            "text": ["Interaction between USA and GBR with EventCode 010", 
                     "Interaction between FRA and DEU with EventCode 040"],
            "label": [0, 0],
            "confidence": [0.4, 0.6],
            "source": ["gdelt_peaceful", "gdelt_peaceful"],
            "date": ["2023-01-01", "2023-01-02"]
        })
        
        # Apply mocks
        with patch('fetch_peaceful_events.extract_peaceful_gdelt_events', return_value=mock_gdelt_peaceful):
            peaceful_df = create_peaceful_dataset()
        
        # Check tag encoding
        self.assertIn("tags", peaceful_df.columns)
        self.assertTrue(all(isinstance(tags, list) for tags in peaceful_df["tags"]))
        self.assertTrue(all(len(tags) == len(TAG_VOCAB) for tags in peaceful_df["tags"]))
        
        # Check tag stats
        self.assertIn("tag_count", peaceful_df.columns)
        self.assertIn("no_tags", peaceful_df.columns)
        
        logger.info("[PASS] Peaceful dataset creation test passed")
    
    def test_combine_with_conflict_data(self):
        """Test combining peaceful and conflict data."""
        # Create mock data
        peaceful_df = pd.DataFrame({
            "text": ["Peaceful event 1", "Peaceful event 2"],
            "label": [0, 0],
            "confidence": [0.5, 0.6],
            "source": ["synthetic_peaceful", "gdelt_peaceful"],
            "tags": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "tag_count": [0, 0],
            "no_tags": [True, True]
        })
        
        conflict_df = pd.DataFrame({
            "text": ["Conflict event 1", "Conflict event 2"],
            "label": [1, 1],
            "confidence": [0.7, 0.8],
            "source": ["gdelt", "acled"],
            "tags": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]],
            "tag_count": [1, 1],
            "no_tags": [False, False]
        })
        
        # Mock data loading
        from unittest.mock import patch, mock_open
        with patch('fetch_peaceful_events.pd.read_csv', return_value=conflict_df), \
             patch('builtins.open', mock_open()), \
             patch('fetch_peaceful_events.os.path.exists', return_value=False):
            
            combined_df = combine_with_conflict_data(peaceful_df, force=True)
        
        # Check combined data
        self.assertEqual(len(combined_df), 4)  # 2 peaceful + 2 conflict
        self.assertEqual(combined_df["label"].sum(), 2)  # 2 conflict events
        self.assertEqual(len(combined_df) - combined_df["label"].sum(), 2)  # 2 peaceful events
        
        logger.info("[PASS] Data combination test passed")
    
    def test_balanced_model_data_preparation(self):
        """Test end-to-end preparation of balanced model data."""
        # This test requires mocking multiple components and full filesystem access
        # For simplicity, we'll verify that the function calls other functions that we've already tested
        from unittest.mock import patch, MagicMock, mock_open
        import pickle
        
        # Create mock data and returns
        mock_peaceful_df = pd.DataFrame({
            "text": ["Peaceful event 1", "Peaceful event 2"],
            "label": [0, 0],
            "confidence": [0.5, 0.6],
            "source": ["synthetic_peaceful", "gdelt_peaceful"],
            "tags": [[0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "tag_count": [0, 0],
            "no_tags": [True, True]
        })
        
        mock_combined_df = pd.DataFrame({
            "text": ["Conflict event 1", "Conflict event 2", "Peaceful event 1", "Peaceful event 2"],
            "label": [1, 1, 0, 0],
            "confidence": [0.7, 0.8, 0.5, 0.6],
            "source": ["gdelt", "acled", "synthetic_peaceful", "gdelt_peaceful"],
            "tags": [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]],
            "tag_count": [1, 1, 0, 0],
            "no_tags": [False, False, True, True]
        })
        
        mock_X = np.array([[1, 0, 0, 0, 0, 0.7], 
                           [0, 1, 0, 0, 0, 0.8], 
                           [0, 0, 0, 0, 0, 0.5], 
                           [0, 0, 0, 0, 0, 0.6]])
        mock_y = np.array([1, 1, 0, 0])
        mock_texts = ["Conflict event 1", "Conflict event 2", "Peaceful event 1", "Peaceful event 2"]
        
        # Apply mocks
        with patch('fetch_peaceful_events.create_peaceful_dataset', return_value=mock_peaceful_df), \
             patch('fetch_peaceful_events.combine_with_conflict_data', return_value=mock_combined_df), \
             patch('fetch_and_clean_data.prepare_model_data', return_value=(mock_X, mock_y, mock_texts)), \
             patch('numpy.save'), \
             patch('pickle.dump'), \
             patch('os.makedirs', return_value=None), \
             patch('builtins.open', mock_open()):
            
            X, y, texts = prepare_balanced_model_data(force=True)
        
        # Check return values
        self.assertEqual(X.shape, (4, 6))  # 4 samples, 5 tags + 1 confidence
        self.assertEqual(y.shape, (4,))  # 4 labels
        self.assertEqual(len(texts), 4)  # 4 texts
        
        logger.info("[PASS] Balanced model data preparation test passed")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main() 