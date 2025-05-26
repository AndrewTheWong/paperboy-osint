#!/usr/bin/env python3
"""
Module for storing and retrieving articles from Supabase.
"""
import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import json

from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('supabase_storage')

# Maximum length for text fields to avoid issues with database constraints
MAX_TEXT_LENGTH = 5000

def get_supabase_client() -> Client:
    """
    Initialize and return a Supabase client using environment variables.
    
    Returns:
        Supabase client
    """
    # Load environment variables if not already loaded
    load_dotenv()
    
    # Get Supabase credentials from environment
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    
    if not supabase_url or not supabase_key:
        raise ValueError("Missing Supabase credentials in environment variables. "
                         "Make sure SUPABASE_URL and SUPABASE_KEY are set in .env file.")
    
    # Create and return Supabase client
    return create_client(supabase_url, supabase_key)

def _prepare_article_for_upload(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Prepare an article for upload by ensuring it meets database constraints.
    
    Args:
        article: The article to prepare
        
    Returns:
        Article with fields properly formatted for database storage
    """
    # Create a copy to avoid modifying the original
    db_article = {}
    
    # Copy relevant fields, truncating text fields if necessary
    for key, value in article.items():
        if key == 'translated_text' and isinstance(value, str) and len(value) > MAX_TEXT_LENGTH:
            db_article[key] = value[:MAX_TEXT_LENGTH]
        elif key == 'embedding':
            # Store embedding as is - Postgres with pgvector can handle arrays
            db_article[key] = value
        elif key == 'tags' and isinstance(value, list):
            # Convert tags to array format for PostgreSQL
            db_article[key] = value
        else:
            db_article[key] = value
    
    return db_article

def article_exists(supabase: Client, url: str) -> bool:
    """
    Check if an article with the given URL already exists in the database.
    
    Args:
        supabase: Supabase client
        url: Article URL to check
        
    Returns:
        True if article exists, False otherwise
    """
    try:
        result = supabase.table('osint_articles').select("id").eq("url", url).execute()
        return len(result.data) > 0
    except Exception as e:
        logger.error(f"Error checking if article exists: {str(e)}")
        # In case of error, assume it doesn't exist
        return False

def upload_articles_to_supabase(articles: List[Dict[str, Any]]) -> int:
    """
    Upload articles to Supabase, skipping duplicates.
    
    Args:
        articles: List of article dictionaries to upload
        
    Returns:
        Number of articles successfully uploaded
    """
    try:
        # Get Supabase client
        supabase = get_supabase_client()
        
        # Track success count
        success_count = 0
        duplicate_count = 0
        error_count = 0
        
        for article in articles:
            try:
                # Skip if article has no URL
                if 'url' not in article:
                    logger.warning("Skipping article without URL")
                    continue
                
                # Check if article already exists
                if article_exists(supabase, article['url']):
                    logger.debug(f"Article already exists: {article['url']}")
                    duplicate_count += 1
                    continue
                
                # Prepare article for upload
                db_article = _prepare_article_for_upload(article)
                
                # Upload to Supabase
                supabase.table('osint_articles').insert(db_article).execute()
                success_count += 1
                logger.debug(f"Uploaded article: {article['url']}")
                
            except Exception as e:
                logger.error(f"Error uploading article {article.get('url', 'unknown')}: {str(e)}")
                error_count += 1
        
        # Log summary
        logger.info(f"Upload complete: {success_count} new articles uploaded, "
                   f"{duplicate_count} duplicates skipped, {error_count} errors")
        
        return success_count
        
    except Exception as e:
        logger.error(f"Error in upload process: {str(e)}")
        return 0

def load_articles_from_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        List of article dictionaries
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {file_path}")
        return articles
    except Exception as e:
        logger.error(f"Error loading articles from file: {str(e)}")
        return []

if __name__ == "__main__":
    # CLI test block
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload articles to Supabase")
    parser.add_argument("--input", "-i", default="data/embedded_articles.json", 
                        help="Path to the input JSON file containing articles")
    
    args = parser.parse_args()
    
    # Load articles
    articles = load_articles_from_file(args.input)
    
    if not articles:
        logger.error("No articles to upload")
        exit(1)
    
    # Upload to Supabase
    uploaded_count = upload_articles_to_supabase(articles)
    
    # Print summary
    print(f"Upload complete: {uploaded_count} new articles uploaded to Supabase") 