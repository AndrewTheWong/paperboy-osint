#!/usr/bin/env python3
"""
Test script for run_pipeline.py
"""
import os
import json
import unittest
import logging
from unittest.mock import patch, MagicMock, mock_open
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Disable logging for tests
logging.disable(logging.CRITICAL)

class TestRunPipeline(unittest.TestCase):
    """Test cases for the run_pipeline.py script"""
    
    def setUp(self):
        """Set up test environment"""
        # Create test data directory
        os.makedirs("data", exist_ok=True)
        
        # Sample article data
        self.sample_articles = [
            {
                "title": "Test Article 1",
                "url": "https://example.com/1",
                "source": "test_source",
                "text": "This is a test article",
                "scraped_at": "2025-05-20T12:00:00"
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/2",
                "source": "test_source",
                "text": "This is another test article",
                "scraped_at": "2025-05-20T12:05:00"
            }
        ]
        
        # Sample translated articles
        self.sample_translated = [
            {
                "title": "Test Article 1",
                "url": "https://example.com/1",
                "source": "test_source",
                "text": "This is a test article",
                "translated_text": "This is a translated test article",
                "language": "en",
                "scraped_at": "2025-05-20T12:00:00"
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/2",
                "source": "test_source",
                "text": "This is another test article",
                "translated_text": "This is another translated test article",
                "language": "en",
                "scraped_at": "2025-05-20T12:05:00"
            }
        ]
        
        # Sample translated articles with non-English
        self.sample_translated_mixed = [
            {
                "title": "Test Article 1",
                "url": "https://example.com/1",
                "source": "test_source",
                "text": "This is a test article",
                "translated_text": "This is a translated test article",
                "language": "en",
                "scraped_at": "2025-05-20T12:00:00"
            },
            {
                "title": "Article de test 2",
                "url": "https://example.com/2",
                "source": "test_source",
                "text": "Ceci est un article de test",
                "translated_text": "This is a test article",
                "language": "fr",
                "scraped_at": "2025-05-20T12:05:00"
            },
            {
                "title": "テスト記事 3",
                "url": "https://example.com/3",
                "source": "test_source",
                "text": "これはテスト記事です",
                "translated_text": "This is a test article",
                "language": "ja",
                "scraped_at": "2025-05-20T12:10:00"
            }
        ]
        
        # Sample tagged articles
        self.sample_tagged = [
            {
                "title": "Test Article 1",
                "url": "https://example.com/1",
                "source": "test_source",
                "text": "This is a test article",
                "translated_text": "This is a translated test article",
                "language": "en",
                "tags": ["military", "diplomacy"],
                "needs_review": False,
                "scraped_at": "2025-05-20T12:00:00"
            },
            {
                "title": "Test Article 2",
                "url": "https://example.com/2",
                "source": "test_source",
                "text": "This is another test article",
                "translated_text": "This is another translated test article",
                "language": "en",
                "tags": ["unknown"],
                "needs_review": True,
                "scraped_at": "2025-05-20T12:05:00"
            }
        ]
    
    @patch("run_pipeline.check_dependencies")
    @patch("builtins.open", new_callable=mock_open)
    def test_full_pipeline(self, mock_file, mock_check_dependencies):
        """Test the full pipeline runs correctly"""
        # Import the module to test
        from run_pipeline import run_pipeline
        
        # Mock the dependencies check to return True
        mock_check_dependencies.return_value = True
        
        # Create mock modules and functions
        with patch("run_pipeline.scrape_all_dynamic") as mock_scrape:
            with patch("run_pipeline.save_articles_to_file") as mock_save:
                with patch("run_pipeline.translate_articles") as mock_translate:
                    with patch("run_pipeline.tag_articles") as mock_tag:
                        with patch("run_pipeline.upload_articles_to_supabase") as mock_upload:
                            with patch("json.load") as mock_load:
                                with patch("json.dump") as mock_dump:
                                    # Set up return values
                                    mock_scrape.return_value = self.sample_articles
                                    mock_translate.return_value = self.sample_translated
                                    mock_tag.return_value = self.sample_tagged
                                    mock_load.side_effect = [
                                        self.sample_articles,  # For translation
                                        self.sample_translated,  # For tagging
                                        self.sample_tagged,  # For storage
                                    ]
                                    mock_upload.return_value = 2
                                    
                                    # Run the pipeline
                                    result = run_pipeline()
                                    
                                    # Assertions
                                    self.assertTrue(result)
                                    mock_scrape.assert_called_once()
                                    mock_save.assert_called_once()
                                    mock_translate.assert_called_once()
                                    mock_tag.assert_called_once()
                                    mock_upload.assert_called_once()
                                    
                                    # Check function arguments
                                    mock_save.assert_called_with(self.sample_articles, "data/articles.json")
                                    mock_translate.assert_called_with(self.sample_articles)
                                    mock_tag.assert_called_with(self.sample_translated)
                                    mock_upload.assert_called_with(self.sample_tagged)

    @patch("run_pipeline.check_dependencies")
    @patch("builtins.open", new_callable=mock_open)
    def test_non_english_translation(self, mock_file, mock_check_dependencies):
        """Test the pipeline correctly identifies and counts non-English articles"""
        # Import the module to test
        from run_pipeline import run_pipeline
        
        # Mock the dependencies check to return True
        mock_check_dependencies.return_value = True
        
        # Create mock modules and functions
        with patch("run_pipeline.scrape_all_dynamic") as mock_scrape:
            with patch("run_pipeline.save_articles_to_file") as mock_save:
                with patch("run_pipeline.translate_articles") as mock_translate:
                    with patch("run_pipeline.tag_articles") as mock_tag:
                        with patch("run_pipeline.upload_articles_to_supabase") as mock_upload:
                            with patch("json.load") as mock_load:
                                with patch("json.dump") as mock_dump:
                                    with patch("logging.Logger.info") as mock_log_info:
                                        # Set up return values
                                        mock_scrape.return_value = self.sample_articles
                                        mock_translate.return_value = self.sample_translated_mixed
                                        mock_tag.return_value = self.sample_tagged
                                        mock_load.side_effect = [
                                            self.sample_articles,  # For translation
                                            self.sample_translated_mixed,  # For tagging
                                            self.sample_tagged,  # For storage
                                        ]
                                        mock_upload.return_value = 2
                                        
                                        # Run the pipeline
                                        result = run_pipeline()
                                        
                                        # Assertions
                                        self.assertTrue(result)
                                        
                                        # Verify that non-English articles were logged
                                        # Find the specific log message about non-English articles
                                        non_english_log_message = None
                                        for call in mock_log_info.call_args_list:
                                            args, _ = call
                                            if args and isinstance(args[0], str) and "non-English articles were translated" in args[0]:
                                                non_english_log_message = args[0]
                                                break
                                        
                                        self.assertIsNotNone(non_english_log_message, "Log message about non-English articles not found")
                                        self.assertIn("2 non-English articles were translated", non_english_log_message)

if __name__ == "__main__":
    unittest.main() 