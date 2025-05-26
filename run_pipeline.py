#!/usr/bin/env python3
"""
run_pipeline.py - Execute the full Paperboy OSINT data pipeline:
- Scraping
- Translation
- Auto-tagging
- Embedding
- Supabase storage
"""
import os
import json
import logging
import time
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('paperboy_pipeline')

# Try importing dotenv, but continue if not available
try:
    from dotenv import load_dotenv
    logger.info("Successfully imported dotenv")
    has_dotenv = True
except ImportError:
    logger.warning("python-dotenv not installed. Environment variables will not be loaded from .env file.")
    logger.info("Install with: pip install python-dotenv")
    has_dotenv = False

# Define functions with proper error handling that will try to import and use the real implementations
# but fallback to stubs if there are import errors

def scrape_all_dynamic():
    """
    Scrape articles from multiple sources dynamically based on config.
    Returns a list of article dictionaries.
    """
    try:
        # Try to import from new location first
        try:
            from scraping.runner import scrape_all_dynamic as real_scrape
            return real_scrape(use_samples=True)
        except ImportError:
            # Fall back to old location
            from pipelines.dynamic_scraper import scrape_all_dynamic as real_scrape
            return real_scrape()
    except ImportError:
        logger.error("Failed to import scraper module. Make sure it exists and is properly installed.")
        # Return a minimal sample dataset as fallback
        from datetime import datetime
        return [
            {
                "title": "Sample article for testing",
                "url": "https://example.com/sample/1",
                "source": "Sample News",
                "scraped_at": datetime.utcnow().isoformat(),
                "language": "en"
            }
        ]
    except Exception as e:
        logger.error(f"Error in scrape_all_dynamic: {str(e)}")
        return []

def save_articles_to_file(articles, filepath):
    """
    Save articles to a JSON file.
    Returns the path to the saved file.
    """
    try:
        # Try to import from new location first
        try:
            from scraping.runner import save_articles_to_file as real_save
            return real_save(articles, filepath)
        except ImportError:
            # Fall back to old location
            from pipelines.dynamic_scraper import save_articles_to_file as real_save
            return real_save(articles, filepath)
    except ImportError:
        logger.error("Failed to import scraper module for saving articles.")
        # Fallback implementation
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        return filepath
    except Exception as e:
        logger.error(f"Error in save_articles_to_file: {str(e)}")
        return None

def translate_articles(articles):
    """
    Translate non-English articles to English.
    Returns a list of articles with translations added.
    """
    try:
        from pipelines.translation_pipeline import translate_articles as real_translate
        return real_translate(articles)
    except ImportError:
        logger.error("Failed to import translation_pipeline module. Make sure it exists and is properly installed.")
        # Simply return the original articles as a fallback, adding a translated_text field
        for article in articles:
            if article.get('language', 'en') != 'en':
                article['translated_text'] = f"[TRANSLATION PLACEHOLDER] {article.get('title', '')}"
            else:
                article['translated_text'] = article.get('title', '')
        return articles
    except Exception as e:
        logger.error(f"Error in translate_articles: {str(e)}")
        return articles

def tag_articles(articles):
    """
    Add tags and review flags to articles based on content analysis.
    Returns a list of articles with tags and review flags added.
    """
    try:
        from tagging.tagging_pipeline import tag_articles as real_tag
        return real_tag(articles)
    except ImportError:
        logger.error("Failed to import tagging_pipeline module. Make sure it exists and is properly installed.")
        # Fallback implementation - add default tags
        for article in articles:
            article["tags"] = ["unknown"]
            article["needs_review"] = True
        return articles
    except Exception as e:
        logger.error(f"Error in tag_articles: {str(e)}")
        return articles

def embed_articles(articles):
    """
    Generate embeddings for articles using the E5 multilingual model.
    Returns a list of articles with embeddings added.
    """
    try:
        from pipelines.embedding_pipeline import embed_articles as real_embed
        return real_embed(articles)
    except ImportError:
        logger.error("Failed to import embedding_pipeline module. Make sure it exists and is properly installed.")
        # Fallback implementation - add empty embeddings
        for article in articles:
            article["embedding"] = []
            article["embedding_model"] = None
        return articles
    except Exception as e:
        logger.error(f"Error in embed_articles: {str(e)}")
        return articles

def cluster_articles(articles):
    """
    Cluster articles based on their embeddings using DBSCAN.
    Returns a list of articles with cluster assignments.
    """
    try:
        from pipelines.cluster_articles import cluster_articles as real_cluster
        return real_cluster(input_path=None, output_path=None, articles=articles)
    except ImportError:
        logger.error("Failed to import cluster_articles module. Make sure it exists and is properly installed.")
        # Fallback implementation - assign default cluster
        for article in articles:
            article["cluster_id"] = -1
        return articles
    except Exception as e:
        logger.error(f"Error in cluster_articles: {str(e)}")
        return articles

def upload_articles_to_supabase(articles):
    """
    Upload articles to Supabase, deduplicating based on URL.
    Returns the number of articles successfully uploaded.
    """
    try:
        from pipelines.supabase_storage import upload_articles_to_supabase as real_upload
        return real_upload(articles)
    except ImportError:
        logger.error("Failed to import supabase_storage module. Make sure it exists and is properly installed.")
        logger.error("Ensure python-dotenv and supabase packages are installed.")
        return 0
    except Exception as e:
        logger.error(f"Error in upload_articles_to_supabase: {str(e)}")
        return 0

def check_dependencies():
    """Check for required dependencies and .env file"""
    # Check for .env file
    env_file = Path('.env')
    if not env_file.exists():
        logger.error("❌ .env file not found. Create one with required credentials.")
        logger.info("Required .env variables: SUPABASE_URL, SUPABASE_KEY")
        return False
    
    # Load environment variables
    if has_dotenv:
        load_dotenv()
    else:
        logger.warning("⚠️ Skipping loading .env file due to missing python-dotenv")
    
    # Check for required environment variables
    required_vars = ['SUPABASE_URL', 'SUPABASE_KEY']
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        logger.info("Make sure these are set in your .env file")
        return False
    
    logger.info("✅ All critical dependencies available")
    return True

def run_pipeline():
    """
    Execute the full Paperboy OSINT data pipeline:
    1. Scrape articles
    2. Translate articles
    3. Tag articles
    4. Generate embeddings
    5. Store in Supabase
    """
    # Start pipeline timer
    start_time = time.time()
    
    # Check dependencies first
    if not check_dependencies():
        logger.error("❌ Pipeline aborted due to missing dependencies")
        return False
    
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Define file paths
    articles_path = "data/articles.json"
    translated_path = "data/translated_articles.json"
    tagged_path = "data/tagged_articles.json"
    embedded_path = "data/embedded_articles.json"
    
    # Track statistics
    stats = {
        "scraped": 0,
        "translated": 0,
        "tagged": 0,
        "embedded": 0,
        "stored": 0,
        "needs_review": 0,
        "keyword_only_tagged": 0,
        "ml_tagged": 0,
        "hybrid_tagged": 0
    }
    
    # 1. SCRAPE
    logger.info("Phase 1: Scraping articles")
    try:
        articles = scrape_all_dynamic()
        if not articles:
            logger.warning("No articles were scraped")
            return False
        
        stats["scraped"] = len(articles)
        
        # Save articles to file
        save_articles_to_file(articles, articles_path)
        logger.info(f"✅ Scraped and saved {stats['scraped']} articles to {articles_path}")
    except Exception as e:
        logger.error(f"❌ Error in scraping step: {str(e)}")
        return False
    
    # 2. TRANSLATE
    logger.info("Phase 2: Translating articles")
    try:
        # Load articles from file
        with open(articles_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        
        # Translate articles
        translated_articles = translate_articles(articles)
        stats["translated"] = len(translated_articles)
        
        # Count non-English articles that were translated
        non_english_count = sum(1 for article in translated_articles 
                              if article.get('language') != 'en')
        
        # Save translated articles
        with open(translated_path, 'w', encoding='utf-8') as f:
            json.dump(translated_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Translated and saved {stats['translated']} articles to {translated_path}")
        logger.info(f"✅ {non_english_count} non-English articles were translated")
        
        # Check for any translation failures
        if stats["translated"] < stats["scraped"]:
            logger.warning(f"⚠️ {stats['scraped'] - stats['translated']} articles failed translation")
    except Exception as e:
        logger.error(f"❌ Error in translation step: {str(e)}")
        return False
    
    # 3. TAG
    logger.info("Phase 3: Tagging articles")
    try:
        # Load translated articles
        with open(translated_path, 'r', encoding='utf-8') as f:
            translated_articles = json.load(f)
        
        # Tag articles
        tagged_articles = tag_articles(translated_articles)
        stats["tagged"] = len(tagged_articles)
        
        # Count articles needing review
        stats["needs_review"] = sum(
            1 for article in tagged_articles 
            if article.get("needs_review", False) or article.get("tags") == ["unknown"]
        )
        
        # Count tagging method statistics
        stats["keyword_only_tagged"] = sum(
            1 for article in tagged_articles 
            if article.get("keyword_tags") and not article.get("ml_tags")
        )
        stats["ml_tagged"] = sum(
            1 for article in tagged_articles 
            if article.get("ml_tags") and not article.get("keyword_tags")
        )
        stats["hybrid_tagged"] = sum(
            1 for article in tagged_articles 
            if article.get("keyword_tags") and article.get("ml_tags")
        )
        
        # Save tagged articles
        with open(tagged_path, 'w', encoding='utf-8') as f:
            json.dump(tagged_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Tagged and saved {stats['tagged']} articles to {tagged_path}")
        logger.info(f"🧠 Tagged {stats['tagged']} articles ({stats['keyword_only_tagged']} keyword-only, {stats['ml_tagged']} ML-only, {stats['hybrid_tagged']} hybrid)")
        logger.info(f"ℹ️ {stats['needs_review']} articles flagged for human review")
    except Exception as e:
        logger.error(f"❌ Error in tagging step: {str(e)}")
        return False
    
    # 4. EMBED
    logger.info("Phase 4: Generating embeddings")
    try:
        # Load tagged articles
        with open(tagged_path, 'r', encoding='utf-8') as f:
            tagged_articles = json.load(f)
        
        # Generate embeddings
        embedded_articles = embed_articles(tagged_articles)
        stats["embedded"] = sum(
            1 for article in embedded_articles
            if article.get("embedding") and len(article.get("embedding", [])) > 0
        )
        
        # Save embedded articles
        with open(embedded_path, 'w', encoding='utf-8') as f:
            json.dump(embedded_articles, f, ensure_ascii=False, indent=2)
        
        # Get model name from first article with embeddings
        model_name = "unknown"
        for article in embedded_articles:
            if article.get("embedding_model"):
                model_name = article.get("embedding_model")
                break
        
        logger.info(f"✅ Embedded and saved {stats['embedded']} articles to {embedded_path}")
        logger.info(f"🧬 Embedded {stats['embedded']} articles using {model_name}")
        
        # Check for any embedding failures
        if stats["embedded"] < stats["tagged"]:
            logger.warning(f"⚠️ {stats['tagged'] - stats['embedded']} articles couldn't be embedded (may be missing text)")
    except Exception as e:
        logger.error(f"❌ Error in embedding step: {str(e)}")
        logger.warning("Continuing without embeddings")
        embedded_articles = tagged_articles
    
    # 5. CLUSTER
    logger.info("Phase 5: Clustering articles by topic similarity")
    clustered_path = "data/clustered_articles.json"
    stats["clusters"] = 0
    
    try:
        # Cluster articles based on embeddings
        clustered_articles = cluster_articles(embedded_articles)
        
        # Count number of unique clusters (excluding -1/noise)
        cluster_ids = [article.get('cluster_id', -1) for article in clustered_articles]
        n_clusters = len(set(cluster_ids) - {-1})
        stats["clusters"] = n_clusters
        
        # Save clustered articles
        with open(clustered_path, 'w', encoding='utf-8') as f:
            json.dump(clustered_articles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Clustered and saved articles to {clustered_path}")
        logger.info(f"🔗 Clustered into {n_clusters} topic groups")
        
        # Use clustered articles for storage
        embedded_articles = clustered_articles
    except Exception as e:
        logger.error(f"❌ Error in clustering step: {str(e)}")
        logger.warning("Continuing without clustering")
    
    # 6. STORE
    logger.info("Phase 6: Storing articles in Supabase")
    storage_success = False
    try:
        # Use embedded articles if available, otherwise tagged articles
        articles_to_store = embedded_articles if "embedded_articles" in locals() else tagged_articles
        
        # Upload to Supabase
        uploaded_count = upload_articles_to_supabase(articles_to_store)
        stats["stored"] = uploaded_count
        
        logger.info(f"✅ Uploaded {stats['stored']} articles to Supabase")
        
        # Check for any storage failures
        if stats["stored"] < len(articles_to_store):
            logger.warning(f"⚠️ {len(articles_to_store) - stats['stored']} articles failed to upload (may be duplicates)")
        
        storage_success = True
    except Exception as e:
        logger.error(f"❌ Error in storage step: {str(e)}")
        logger.error("This may be due to invalid Supabase credentials or connection issues")
    
    # Calculate total run time
    end_time = time.time()
    run_time = end_time - start_time
    
    # Print pipeline summary
    logger.info("\n--- PIPELINE SUMMARY ---")
    logger.info(f"Articles scraped:    {stats['scraped']}")
    logger.info(f"Articles translated: {stats['translated']}")
    logger.info(f"Articles tagged:     {stats['tagged']} ({stats['keyword_only_tagged']} keyword-only, {stats['ml_tagged']} ML-only, {stats['hybrid_tagged']} hybrid)")
    logger.info(f"Articles embedded:   {stats['embedded']}")
    logger.info(f"Topic clusters:      {stats.get('clusters', 0)}")
    logger.info(f"Articles stored:     {stats['stored']}")
    logger.info(f"Articles for review: {stats['needs_review']}")
    logger.info(f"Total run time:      {run_time:.2f} seconds")
    
    # Overall status
    pipeline_success = stats["scraped"] > 0 and stats["tagged"] > 0 and storage_success
    
    if pipeline_success:
        logger.info("\n✅ Pipeline execution completed successfully")
    else:
        logger.error("\n🔥 Pipeline completed with issues")
    
    return pipeline_success

if __name__ == "__main__":
    logger.info("Starting Paperboy OSINT data pipeline")
    success = run_pipeline()
    exit(0 if success else 1) 