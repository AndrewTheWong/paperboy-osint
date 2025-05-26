#!/usr/bin/env python3
"""
run_pipeline_with_mocks.py - Execute the full Paperboy backend pipeline
using mock data for testing purposes.

This script simulates the full pipeline flow without making actual
external requests to scrapers, translation services, or Supabase.
"""
import os
import json
import logging
import argparse
import subprocess
from pathlib import Path

# Try importing dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    logger = logging.getLogger('paperboy_pipeline_mock')
    logger.info("Successfully imported dotenv")
    has_dotenv = True
except ImportError:
    logger = logging.getLogger('paperboy_pipeline_mock')
    logger.warning("python-dotenv not installed. Environment variables will not be loaded from .env file.")
    logger.info("Install with: pip install python-dotenv")
    has_dotenv = False

# Import the mock data generator
try:
    from pipelines.mock_data_generator import (
        generate_mock_articles,
        generate_translated_articles,
        generate_tagged_articles,
        generate_embedded_articles
    )
    has_mock_generator = True
except ImportError:
    logger.warning("pipelines.mock_data_generator not found. Using simple mock data generation.")
    has_mock_generator = False
    
    # Simple mock data generators
    def generate_mock_articles(count):
        """Generate simple mock articles"""
        return [{"id": i, "title": f"Mock Article {i}", "text": f"This is mock article {i}", "url": f"https://example.com/{i}"} for i in range(count)]
        
    def generate_translated_articles(articles):
        """Generate simple mock translated articles"""
        for article in articles:
            article["translated_text"] = article["text"]
            article["source_language"] = "en"
        return articles
        
    def generate_tagged_articles(articles):
        """Generate simple mock tagged articles"""
        for article in articles:
            article["tags"] = ["mock", "test"]
            article["needs_review"] = i % 3 == 0  # Every third article needs review
        return articles
        
    def generate_embedded_articles(articles):
        """Generate simple mock embeddings"""
        for article in articles:
            article["embedding"] = [0.1, 0.2, 0.3, 0.4, 0.5]
        return articles

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('paperboy_pipeline_mock')

def run_pipeline_with_mocks(article_count: int = 10, use_existing: bool = False):
    """
    Execute the full Paperboy backend pipeline using mock data:
    1. Generate/use mock articles
    2. Generate/use mock translations
    3. Generate/use mock tags
    4. Generate/use mock embeddings
    5. Simulate Supabase storage
    6. Conditionally launch Streamlit UI for human review
    
    Args:
        article_count: Number of articles to generate
        use_existing: Whether to use existing mock data files
    """
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Load environment variables if dotenv is available
    if has_dotenv:
        load_dotenv()
    else:
        logger.warning("⚠️ Skipping loading .env file due to missing python-dotenv")
    
    # Define file paths
    articles_path = "data/articles.json"
    translated_path = "data/translated_articles.json"
    tagged_path = "data/tagged_articles.json"
    embedded_path = "data/embedded_articles.json"
    
    # 1. SCRAPE (or use mock articles)
    logger.info("Step 1: Generating mock articles")
    try:
        if use_existing and os.path.exists(articles_path):
            logger.info(f"Using existing articles from {articles_path}")
            with open(articles_path, 'r', encoding='utf-8') as f:
                articles = json.load(f)
        else:
            articles = generate_mock_articles(article_count)
            # Save articles to file
            with open(articles_path, 'w', encoding='utf-8') as f:
                json.dump(articles, f, ensure_ascii=False, indent=2)
            
        logger.info(f"Using {len(articles)} articles")
    except Exception as e:
        logger.error(f"Error in mock article generation: {str(e)}")
        return False
    
    # 2. TRANSLATE (or use mock translations)
    logger.info("Step 2: Generating mock translations")
    try:
        if use_existing and os.path.exists(translated_path):
            logger.info(f"Using existing translations from {translated_path}")
            with open(translated_path, 'r', encoding='utf-8') as f:
                translated_articles = json.load(f)
        else:
            # Generate mock translations
            translated_articles = generate_translated_articles(articles)
            # Save translated articles
            with open(translated_path, 'w', encoding='utf-8') as f:
                json.dump(translated_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Using {len(translated_articles)} translated articles")
    except Exception as e:
        logger.error(f"Error in mock translation: {str(e)}")
        return False
    
    # 3. TAG (or use mock tags)
    logger.info("Step 3: Generating mock tags")
    try:
        if use_existing and os.path.exists(tagged_path):
            logger.info(f"Using existing tags from {tagged_path}")
            with open(tagged_path, 'r', encoding='utf-8') as f:
                tagged_articles = json.load(f)
        else:
            # Generate mock tags
            tagged_articles = generate_tagged_articles(translated_articles)
            # Save tagged articles
            with open(tagged_path, 'w', encoding='utf-8') as f:
                json.dump(tagged_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Using {len(tagged_articles)} tagged articles")
    except Exception as e:
        logger.error(f"Error in mock tagging: {str(e)}")
        return False
    
    # 4. EMBED (or use mock embeddings)
    logger.info("Step 4: Generating mock embeddings")
    try:
        if use_existing and os.path.exists(embedded_path):
            logger.info(f"Using existing embeddings from {embedded_path}")
            with open(embedded_path, 'r', encoding='utf-8') as f:
                embedded_articles = json.load(f)
        else:
            # Generate mock embeddings
            embedded_articles = generate_embedded_articles(tagged_articles)
            # Save embedded articles
            with open(embedded_path, 'w', encoding='utf-8') as f:
                json.dump(embedded_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Using {len(embedded_articles)} embedded articles")
    except Exception as e:
        logger.error(f"Error in mock embedding: {str(e)}")
        return False
    
    # 5. STORE (mock Supabase storage)
    logger.info("Step 5: Simulating Supabase storage")
    try:
        # Just simulate storage by logging
        upload_count = len(embedded_articles)
        logger.info(f"Simulated upload of {upload_count} articles to Supabase")
    except Exception as e:
        logger.error(f"Error in mock storage: {str(e)}")
        # Continue even if storage fails
    
    # 6. CHECK FOR ARTICLES NEEDING REVIEW
    logger.info("Step 6: Checking for articles needing human review")
    try:
        # Check if any articles need review
        need_review = any(
            article.get("needs_review", False) or article.get("tags") == ["unknown"]
            for article in tagged_articles
        )
        
        review_count = sum(1 for a in tagged_articles if a.get("needs_review", False))
        unknown_count = sum(1 for a in tagged_articles if a.get("tags") == ["unknown"])
        
        logger.info(f"Found {review_count} articles needing review and {unknown_count} with unknown tags")
        
        if need_review:
            logger.info("🚨 Articles need human review")
            
            # Launch Streamlit UI
            try:
                streamlit_ui_path = "ui/human_tag_review.py"
                if not Path(streamlit_ui_path).exists():
                    logger.error(f"❌ Streamlit UI file not found: {streamlit_ui_path}")
                    logger.info("Skipping Streamlit launch")
                else:
                    logger.info(f"Launching Streamlit UI: {streamlit_ui_path}")
                    # Launch Streamlit in a subprocess
                    subprocess.run(["streamlit", "run", streamlit_ui_path], check=True)
            except FileNotFoundError:
                logger.error("❌ Failed to launch Streamlit. Make sure it is installed and available in PATH.")
                logger.error("Try: pip install streamlit")
            except subprocess.CalledProcessError as e:
                logger.error(f"❌ Streamlit process failed with exit code {e.returncode}")
                logger.error("Try: pip install streamlit")
        else:
            logger.info("✅ No articles need human review")
    except Exception as e:
        logger.error(f"Error checking for articles needing review: {str(e)}")
    
    logger.info("Mock pipeline execution completed successfully")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run the Paperboy pipeline with mock data")
    parser.add_argument("--count", "-c", type=int, default=10, help="Number of mock articles to generate")
    parser.add_argument("--use-existing", "-e", action="store_true", help="Use existing mock data files if available")
    
    args = parser.parse_args()
    
    logger.info("Starting Paperboy backend pipeline with mock data")
    success = run_pipeline_with_mocks(args.count, args.use_existing)
    exit(0 if success else 1) 