#!/usr/bin/env python3
"""
🚀 Entry Script: pipeline_main.py
Demonstrates programmatic use of the upgraded StraitWatch pipeline
"""

import logging
from typing import Dict, Any
from app.tasks.pipeline_tasks import run_article_pipeline, run_batch_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_single_pipeline(article: Dict[str, Any]) -> str:
    """
    Run the complete pipeline for a single article
    
    Args:
        article: Article data dictionary
        
    Returns:
        str: Task ID for tracking
    """
    logger.info(f"🚀 Starting pipeline for article: {article.get('title', 'Unknown')}")
    
    # Execute the pipeline chain: Preprocess → NER Tag → Embed+Cluster → Store
    task = run_article_pipeline.delay(article)
    
    logger.info(f"✅ Pipeline task started with ID: {task.id}")
    return task.id

def run_batch_pipeline_example(articles: list) -> str:
    """
    Run the pipeline for multiple articles in batch
    
    Args:
        articles: List of article data dictionaries
        
    Returns:
        str: Batch task ID for tracking
    """
    logger.info(f"🔄 Starting batch pipeline for {len(articles)} articles")
    
    # Execute batch processing
    task = run_batch_pipeline.delay(articles)
    
    logger.info(f"✅ Batch pipeline task started with ID: {task.id}")
    return task.id

def main():
    """Main demonstration of the pipeline"""
    print("🚀 StraitWatch Upgraded Pipeline Demo")
    print("=" * 50)
    
    # Example article data
    sample_article = {
        "article_id": "demo-001",
        "title": "South China Sea Maritime Security Update",
        "body": """
        Recent intelligence reports indicate increased naval activity in the South China Sea. 
        China has been conducting military exercises near Taiwan, raising concerns about 
        regional maritime security. Cybersecurity experts have also noted increased 
        hacking attempts targeting naval command systems in the region.
        """,
        "region": "East Asia", 
        "topic": "Maritime Security",
        "source_url": "https://example.com/demo-article"
    }
    
    # Run single article pipeline
    print("📝 Processing single article...")
    task_id = run_single_pipeline(sample_article)
    print(f"   Task ID: {task_id}")
    
    # Example batch data
    batch_articles = [
        {
            "article_id": "batch-001",
            "title": "Strait of Malacca Security Alert",
            "body": "Singapore maritime authorities report increased piracy activity in the Strait of Malacca.",
            "region": "Southeast Asia",
            "topic": "Maritime Security", 
            "source_url": "https://example.com/batch-1"
        },
        {
            "article_id": "batch-002", 
            "title": "Regional Cybersecurity Cooperation",
            "body": "ASEAN nations announce new cybersecurity partnership to combat regional cyber threats.",
            "region": "Southeast Asia",
            "topic": "Cybersecurity",
            "source_url": "https://example.com/batch-2"
        }
    ]
    
    # Run batch pipeline  
    print("\n📦 Processing batch articles...")
    batch_task_id = run_batch_pipeline_example(batch_articles)
    print(f"   Batch Task ID: {batch_task_id}")
    
    print("\n✅ Pipeline tasks submitted!")
    print("\n💡 Monitor progress with:")
    print("   - Celery worker logs: celery -A app.celery_worker worker --loglevel=info --pool=solo")
    print("   - Supabase dashboard: Check articles table for processed results")
    print("   - API status: GET /ingest/status")

if __name__ == "__main__":
    main() 