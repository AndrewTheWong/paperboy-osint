import os
import unittest
import pandas as pd
import numpy as np
import logging
import sys
import json
from pathlib import Path
from fetch_and_clean_data import (
    tag_encoder, 
    preprocess_gdelt, 
    preprocess_acled, 
    preprocess_ucdp, 
    create_unified_dataset,
    prepare_model_data,
    TAG_VOCAB
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestPreprocessingPipeline(unittest.TestCase):
    """Test the preprocessing pipeline for conflict data."""
    
    def setUp(self):
        """Set up test environment."""
        # Create test directories
        self.test_dir = Path("test_data")
        self.test_dir.mkdir(exist_ok=True)
        
        # Create sample data for testing
        self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample data for each source."""
        # Sample GDELT data
        self.gdelt_data = pd.DataFrame({
            "EventCode": ["190", "010", "195", "020"],
            "GoldsteinScale": [5.0, -2.0, 8.0, 0.0],
            "actor1": ["USA", "RUS", "CHN", "JPN"],
            "actor2": ["IRN", "UKR", "TWN", "KOR"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        })
        
        # Sample ACLED data
        self.acled_data = pd.DataFrame({
            "event_type": ["Battles", "Protests", "Violence against civilians", "Strategic developments"],
            "fatalities": [5, 0, 10, 0],
            "actor1": ["Group A", "Protesters", "Military", "Government"],
            "actor2": ["Group B", "", "Civilians", ""],
            "notes": ["armed clash in city center", "peaceful protest", "civilian deaths reported", "diplomatic meeting"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        })
        
        # Sample UCDP data
        self.ucdp_data = pd.DataFrame({
            "type_of_violence": [1, 2, 3, 1],
            "best": [25, 15, 5, 40],
            "side_a": ["Government", "Rebel Group", "Militia", "Military"],
            "side_b": ["Rebels", "Government", "Civilians", "Insurgents"],
            "country": ["Syria", "Yemen", "Sudan", "Myanmar"],
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04"]
        })
    
    def test_tag_encoder(self):
        """Test the tag encoder function."""
        # Test with text containing tags
        text1 = "Military movement and armed clash between forces"
        tags1 = tag_encoder(text1)
        
        # Test with text not containing tags
        text2 = "Regular diplomatic relations established"
        tags2 = tag_encoder(text2)
        
        # Test with empty text
        text3 = ""
        tags3 = tag_encoder(text3)
        
        # Assertions
        self.assertEqual(len(tags1), len(TAG_VOCAB))
        self.assertEqual(len(tags2), len(TAG_VOCAB))
        self.assertEqual(len(tags3), len(TAG_VOCAB))
        
        # Check that tags are correctly identified
        self.assertEqual(tags1[TAG_VOCAB.index("military movement")], 1)
        self.assertEqual(tags1[TAG_VOCAB.index("armed clash")], 1)
        
        # Check that tags are correctly not identified
        self.assertEqual(sum(tags2), 0)  # No tags should be found
        self.assertEqual(sum(tags3), 0)  # No tags in empty text
        
        logger.info("[PASS] Tag encoder test passed")
    
    def test_gdelt_preprocessing(self):
        """Test GDELT preprocessing function."""
        processed = preprocess_gdelt(self.gdelt_data)
        
        # Check column presence
        self.assertIn("text", processed.columns)
        self.assertIn("label", processed.columns)
        self.assertIn("confidence", processed.columns)
        self.assertIn("source", processed.columns)
        
        # Check data validity
        self.assertEqual(len(processed), len(self.gdelt_data))
        self.assertEqual(processed["source"].unique()[0], "gdelt")
        
        # Check label generation (EventCode starting with 19 should be 1)
        expected_labels = [1, 0, 1, 0]
        actual_labels = processed["label"].tolist()
        self.assertEqual(actual_labels, expected_labels)
        
        # Check confidence score conversion
        expected_confidence = [(5.0 + 10) / 20, (-2.0 + 10) / 20, (8.0 + 10) / 20, (0.0 + 10) / 20]
        actual_confidence = processed["confidence"].tolist()
        np.testing.assert_almost_equal(actual_confidence, expected_confidence)
        
        logger.info("[PASS] GDELT preprocessing test passed")
    
    def test_acled_preprocessing(self):
        """Test ACLED preprocessing function."""
        processed = preprocess_acled(self.acled_data)
        
        # Check column presence
        self.assertIn("text", processed.columns)
        self.assertIn("label", processed.columns)
        self.assertIn("confidence", processed.columns)
        self.assertIn("source", processed.columns)
        
        # Check data validity
        self.assertEqual(len(processed), len(self.acled_data))
        self.assertEqual(processed["source"].unique()[0], "acled")
        
        # Check label generation (Battles and Violence against civilians should be 1)
        expected_labels = [1, 0, 1, 0]
        actual_labels = processed["label"].tolist()
        self.assertEqual(actual_labels, expected_labels)
        
        # Check confidence score conversion (based on fatalities)
        expected_confidence = [0.5, 0.0, 1.0, 0.0]  # 5/10, 0/10, 10/10 (capped at 1), 0/10
        actual_confidence = processed["confidence"].tolist()
        np.testing.assert_almost_equal(actual_confidence, expected_confidence)
        
        logger.info("[PASS] ACLED preprocessing test passed")
    
    def test_ucdp_preprocessing(self):
        """Test UCDP preprocessing function."""
        processed = preprocess_ucdp(self.ucdp_data)
        
        # Check column presence
        self.assertIn("text", processed.columns)
        self.assertIn("label", processed.columns)
        self.assertIn("confidence", processed.columns)
        self.assertIn("source", processed.columns)
        
        # Check data validity
        self.assertEqual(len(processed), len(self.ucdp_data))
        self.assertEqual(processed["source"].unique()[0], "ucdp")
        
        # Check label generation (all UCDP should be 1)
        expected_labels = [1, 1, 1, 1]
        actual_labels = processed["label"].tolist()
        self.assertEqual(actual_labels, expected_labels)
        
        # Check confidence score conversion (based on best)
        expected_confidence = [0.25, 0.15, 0.05, 0.4]  # 25/100, 15/100, 5/100, 40/100
        actual_confidence = processed["confidence"].tolist()
        np.testing.assert_almost_equal(actual_confidence, expected_confidence)
        
        logger.info("[PASS] UCDP preprocessing test passed")
    
    def test_create_unified_dataset(self):
        """Test creating a unified dataset from multiple sources."""
        # Preprocess each dataset
        gdelt_processed = preprocess_gdelt(self.gdelt_data)
        acled_processed = preprocess_acled(self.acled_data)
        ucdp_processed = preprocess_ucdp(self.ucdp_data)
        
        # Create unified dataset
        unified = create_unified_dataset(gdelt_processed, acled_processed, ucdp_processed)
        
        # Check for required columns
        required_columns = ["text", "label", "confidence", "source", "tags", "tag_count", "no_tags"]
        for col in required_columns:
            self.assertIn(col, unified.columns)
        
        # Check data validity
        self.assertEqual(len(unified), len(gdelt_processed) + len(acled_processed) + len(ucdp_processed))
        
        # Check tag encoding
        self.assertTrue(all(isinstance(tags, list) for tags in unified["tags"]))
        self.assertEqual(len(unified["tags"][0]), len(TAG_VOCAB))
        
        # Check tag statistics
        self.assertTrue(all(count >= 0 for count in unified["tag_count"]))
        
        logger.info("[PASS] Unified dataset creation test passed")
    
    def test_prepare_model_data(self):
        """Test preparing model-ready data for training."""
        # Create unified dataset
        gdelt_processed = preprocess_gdelt(self.gdelt_data)
        acled_processed = preprocess_acled(self.acled_data)
        ucdp_processed = preprocess_ucdp(self.ucdp_data)
        unified = create_unified_dataset(gdelt_processed, acled_processed, ucdp_processed)
        
        # Prepare model data
        X, y, texts = prepare_model_data(unified)
        
        # Check dimensions
        self.assertEqual(X.shape[0], len(unified))
        self.assertEqual(X.shape[1], len(TAG_VOCAB) + 1)  # +1 for confidence
        self.assertEqual(y.shape[0], len(unified))
        self.assertEqual(len(texts), len(unified))
        
        # Check data types
        self.assertIsInstance(X, np.ndarray)
        self.assertIsInstance(y, np.ndarray)
        self.assertIsInstance(texts, list)
        
        # Check content
        self.assertTrue(all(val in [0, 1] for val in np.unique(X[:, :-1])))  # Tags should be binary
        self.assertTrue(all(0 <= val <= 1 for val in X[:, -1]))  # Confidence should be between 0 and 1
        self.assertTrue(all(val in [0, 1] for val in np.unique(y)))  # Labels should be binary
        
        logger.info("[PASS] Model data preparation test passed")
    
    def test_end_to_end_pipeline(self):
        """Test the complete preprocessing pipeline from raw data to model input."""
        # Process data through the entire pipeline
        gdelt_processed = preprocess_gdelt(self.gdelt_data)
        acled_processed = preprocess_acled(self.acled_data)
        ucdp_processed = preprocess_ucdp(self.ucdp_data)
        unified = create_unified_dataset(gdelt_processed, acled_processed, ucdp_processed)
        X, y, texts = prepare_model_data(unified)
        
        # Save to test directory
        unified.to_csv(self.test_dir / "test_unified_data.csv", index=False)
        np.save(self.test_dir / "test_X_features.npy", X)
        np.save(self.test_dir / "test_y_labels.npy", y)
        
        # Check that files were created
        self.assertTrue((self.test_dir / "test_unified_data.csv").exists())
        self.assertTrue((self.test_dir / "test_X_features.npy").exists())
        self.assertTrue((self.test_dir / "test_y_labels.npy").exists())
        
        # Check file contents by reloading
        df_reloaded = pd.read_csv(self.test_dir / "test_unified_data.csv")
        X_reloaded = np.load(self.test_dir / "test_X_features.npy")
        y_reloaded = np.load(self.test_dir / "test_y_labels.npy")
        
        self.assertEqual(len(df_reloaded), len(unified))
        self.assertEqual(X_reloaded.shape, X.shape)
        self.assertEqual(y_reloaded.shape, y.shape)
        
        logger.info("[PASS] End-to-end pipeline test passed")
    
    def tearDown(self):
        """Clean up test files."""
        import shutil
        if self.test_dir.exists():
            shutil.rmtree(self.test_dir)

if __name__ == "__main__":
    unittest.main() 