import os
import unittest
import pandas as pd
import logging
import tempfile
import shutil
import sys
from io import StringIO
from unittest.mock import patch, MagicMock
import fetch_and_clean_data as fcd
import subprocess
import json
import time
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TestDataPipeline(unittest.TestCase):
    """Test the entire data pipeline and model training workflow."""
    
    def setUp(self):
        """Set up test environment."""
        self.data_dir = Path("data")
        self.logs_dir = Path("logs")
        self.models_dir = Path("models")
        
        # Ensure directories exist
        self.data_dir.mkdir(exist_ok=True)
        self.logs_dir.mkdir(exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        # Set test parameters
        self.min_data_points = 800  # Minimum data points for testing (lowered for practical testing)
        self.ucdp_limit = 5000  # Limit UCDP API fetching for faster testing
        self.gdelt_limit = 5000  # Limit GDELT fetching for faster testing
        
        # Define files that should be created
        self.expected_files = [
            self.data_dir / "all_conflict_data.csv",
            self.data_dir / "gdelt_events.csv",
        ]
        
        # Track test runs
        self.test_timestamp = time.strftime("%Y%m%d_%H%M%S")
        logger.info(f"Starting test run at {self.test_timestamp}")
    
    def test_01_data_fetching(self):
        """Test data fetching from all sources with sample limits."""
        logger.info("Testing data fetching with sample limits...")
        
        # Run the data fetching script with limited samples
        cmd = [
            "python", "fetch_and_clean_data.py",
            "--force",  # Force re-download
            f"--gdelt-samples={self.gdelt_limit}",
            "--gdelt-days=5",  # Use fewer days to speed up testing
            f"--ucdp-limit={self.ucdp_limit}",
            f"--min-total-samples={self.min_data_points}"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output for debugging
        logger.info(f"Command output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command errors: {result.stderr}")
        
        # Check if the command succeeded
        self.assertEqual(result.returncode, 0, f"Data fetching failed with return code {result.returncode}")
        
        # Check if expected files were created
        for file_path in self.expected_files:
            self.assertTrue(file_path.exists(), f"Expected file {file_path} was not created")
        
        # Check if the all_conflict_data.csv file contains data
        all_data_file = self.data_dir / "all_conflict_data.csv"
        self.assertTrue(all_data_file.exists(), f"Combined data file {all_data_file} does not exist")
        
        # Read the combined data
        df = pd.read_csv(all_data_file)
        logger.info(f"Combined data has {len(df)} rows")
        
        # Check if we have the minimum required data
        self.assertGreaterEqual(len(df), self.min_data_points,
                               f"Expected at least {self.min_data_points} data points, got {len(df)}")
        
        # Check if we have both positive and negative examples (conflict and non-conflict)
        conflict_count = df["label"].sum()
        non_conflict_count = len(df) - conflict_count
        
        logger.info(f"Data has {conflict_count} conflict events and {non_conflict_count} non-conflict events")
        self.assertGreater(conflict_count, 0, "Expected at least some conflict events")
        
        # Store data counts for later tests
        self.data_counts = {
            "total": len(df),
            "conflict": int(conflict_count),
            "non_conflict": int(non_conflict_count)
        }
    
    def test_02_model_training(self):
        """Test XGBoost model training with the fetched data."""
        logger.info("Testing XGBoost model training...")
        
        # Ensure we have data from the previous test
        try:
            data_counts = getattr(self, "data_counts", None)
            if not data_counts:
                # If test_01 wasn't run or failed, try to get counts directly
                df = pd.read_csv(self.data_dir / "all_conflict_data.csv")
                data_counts = {
                    "total": len(df),
                    "conflict": int(df["label"].sum()),
                    "non_conflict": int(len(df) - df["label"].sum())
                }
                
            logger.info(f"Using dataset with {data_counts['total']} samples for model training")
        except Exception as e:
            self.fail(f"Could not get data counts from previous test or directly from file: {e}")
        
        # Run the model training script
        cmd = ["python", "train_xgboost_model.py"]
        logger.info(f"Running command: {' '.join(cmd)}")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output for debugging
        logger.info(f"Command output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command errors: {result.stderr}")
        
        # Check if the command succeeded
        self.assertEqual(result.returncode, 0, f"Model training failed with return code {result.returncode}")
        
        # Check if model files were created
        model_files = list(self.models_dir.glob("xgboost_conflict_model_*.pkl"))
        self.assertGreater(len(model_files), 0, "No model file was created")
        
        # Check if metrics file was created
        metadata_files = list(self.models_dir.glob("metadata_xgboost_*.json"))
        self.assertGreater(len(metadata_files), 0, "No metadata file was created")
        
        # Check the most recent metadata file
        latest_metadata = max(metadata_files, key=lambda f: f.stat().st_mtime)
        with open(latest_metadata, 'r') as f:
            metadata = json.load(f)
        
        # Verify metadata contains expected fields
        self.assertIn("metrics", metadata, "Metadata does not contain metrics")
        self.assertIn("roc_auc", metadata["metrics"], "Metrics do not contain ROC AUC")
        self.assertIn("accuracy", metadata["metrics"], "Metrics do not contain accuracy")
        
        # Store metrics for the next test
        self.model_metrics = metadata["metrics"]
        logger.info(f"Model metrics: {self.model_metrics}")
    
    def test_03_model_evaluation(self):
        """Test the model evaluation metrics to ensure they meet quality standards."""
        logger.info("Testing model evaluation metrics...")
        
        # Get metrics from previous test
        metrics = getattr(self, "model_metrics", None)
        if not metrics:
            # If test_02 wasn't run or failed, try to get metrics directly
            metadata_files = list(self.models_dir.glob("metadata_xgboost_*.json"))
            if not metadata_files:
                self.fail("No metadata files found, can't evaluate model")
                
            latest_metadata = max(metadata_files, key=lambda f: f.stat().st_mtime)
            with open(latest_metadata, 'r') as f:
                metadata = json.load(f)
            metrics = metadata["metrics"]
        
        # Check key metrics against minimum standards
        min_roc_auc = 0.5  # Better than random
        min_accuracy = 0.4  # Reasonable baseline for imbalanced data
        
        logger.info(f"Checking if ROC AUC {metrics['roc_auc']} >= {min_roc_auc}")
        self.assertGreaterEqual(metrics["roc_auc"], min_roc_auc, 
                              f"ROC AUC {metrics['roc_auc']} is below minimum standard {min_roc_auc}")
        
        logger.info(f"Checking if accuracy {metrics['accuracy']} >= {min_accuracy}")
        self.assertGreaterEqual(metrics["accuracy"], min_accuracy,
                              f"Accuracy {metrics['accuracy']} is below minimum standard {min_accuracy}")
        
        # Check for overfitting
        logger.info("Checking for overfitting (extreme metrics may indicate problems)")
        self.assertLess(metrics["accuracy"], 0.99, "Accuracy suspiciously high, might indicate data leakage")
        
        logger.info("All model metrics meet quality standards")
        
    def test_04_ucdp_sample_limit(self):
        """Test that UCDP API fetching respects the sample limit."""
        logger.info("Testing UCDP API sample limit functionality...")
        
        # Run the data fetching script with a very small UCDP limit
        small_limit = 500  # A small limit that should be reached quickly
        
        cmd = [
            "python", "fetch_and_clean_data.py",
            "--force",
            "--skip-gdelt",
            "--skip-acled",
            "--skip-ucdp-csv",
            f"--ucdp-limit={small_limit}"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        # Log output for debugging
        logger.info(f"Command output: {result.stdout}")
        if result.stderr:
            logger.warning(f"Command errors: {result.stderr}")
        
        # Check if the command succeeded
        self.assertEqual(result.returncode, 0, f"Data fetching failed with return code {result.returncode}")
        
        # Check if the UCDP API data file was created
        ucdp_file = self.data_dir / "ucdp_api_clean.csv"
        self.assertTrue(ucdp_file.exists(), f"UCDP API data file {ucdp_file} does not exist")
        
        # Read the UCDP data
        df = pd.read_csv(ucdp_file)
        logger.info(f"UCDP API data has {len(df)} rows")
        
        # Check if we have approximately the right number of rows (may be slightly different due to processing)
        # Allow for some tolerance (e.g., +/- 10%)
        tolerance = 0.1
        min_expected = small_limit * (1 - tolerance)
        max_expected = small_limit * (1 + tolerance)
        
        self.assertGreaterEqual(len(df), min_expected, 
                              f"Expected at least {min_expected} UCDP records, got {len(df)}")
        self.assertLessEqual(len(df), max_expected, 
                           f"Expected at most {max_expected} UCDP records, got {len(df)}")
        
        logger.info(f"UCDP API sample limit of {small_limit} was respected ({len(df)} records fetched)")
        
        # Check that we have valid data in the file
        self.assertIn("text", df.columns, "UCDP data should contain a text column")
        self.assertIn("label", df.columns, "UCDP data should contain a label column")
        
        # All UCDP data should be labeled as conflict (label=1)
        self.assertTrue((df["label"] == 1).all(), "All UCDP data should be labeled as conflict (label=1)")
        
        logger.info("UCDP API sample limit test passed")
    
    def tearDown(self):
        """Clean up after tests if needed."""
        logger.info(f"Completed test run at {time.strftime('%Y%m%d_%H%M%S')}")

def inspect_function(func):
    """Return the source code of a function as a string."""
    import inspect
    return inspect.getsource(func)

if __name__ == "__main__":
    unittest.main() 