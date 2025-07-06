#!/usr/bin/env python3
"""
Test script for Celery workers
Demonstrates the pipeline: Scraping → Translation → Tagging → Embedding → Clustering
"""

import time
import logging
from workers.orchestrator import run_full_pipeline
from workers.translator import translate_single_article
from workers.tagger import tag_single_article
from workers.embedder import embed_single_article
from workers.cluster import cluster_articles_batch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_single_worker():
    """Test a single worker with a sample article"""
    logger.info("🧪 Testing single worker pipeline")
    
    # Sample article
    test_article = {
        'article_id': 'test_001',
        'title': '测试标题 - Test Title',
        'body': '这是测试内容。This is test content.',
        'url': 'https://example.com/test',
        'source_name': 'Test Source'
    }
    
    logger.info(f"📰 Original article: {test_article['title']}")
    
    # Step 1: Translation
    logger.info("🌐 Step 1: Translation")
    translated_article = translate_single_article.delay(test_article).get()
    logger.info(f"✅ Translated: {translated_article.get('title_translated', 'N/A')}")
    
    # Step 2: Tagging
    logger.info("🏷️ Step 2: Tagging")
    tagged_article = tag_single_article.delay(translated_article).get()
    tags = tagged_article.get('tags', [])
    entities = tagged_article.get('entities', [])
    logger.info(f"✅ Tagged: {len(tags)} tags, {len(entities)} entities")
    
    # Step 3: Embedding
    logger.info("🔢 Step 3: Embedding")
    embedded_article = embed_single_article.delay(tagged_article).get()
    embedding = embedded_article.get('embedding', [])
    logger.info(f"✅ Embedded: {len(embedding)} dimensions")
    
    logger.info("✅ Single worker test completed")
    return embedded_article

def test_batch_workers():
    """Test batch processing with multiple articles"""
    logger.info("🧪 Testing batch worker pipeline")
    
    # Sample articles
    test_articles = [
        {
            'article_id': 'test_001',
            'title': '测试标题 - Test Title',
            'body': '这是测试内容。This is test content.',
            'url': 'https://example.com/test1',
            'source_name': 'Test Source 1'
        },
        {
            'article_id': 'test_002', 
            'title': 'Another Test Article',
            'body': 'This is another test article with different content.',
            'url': 'https://example.com/test2',
            'source_name': 'Test Source 2'
        }
    ]
    
    logger.info(f"📰 Processing {len(test_articles)} articles")
    
    # Step 1: Batch Translation
    logger.info("🌐 Step 1: Batch Translation")
    from workers.translator import translate_articles_batch
    translated_articles = translate_articles_batch.delay(test_articles).get()
    logger.info(f"✅ Translated {len(translated_articles)} articles")
    
    # Step 2: Batch Tagging
    logger.info("🏷️ Step 2: Batch Tagging")
    from workers.tagger import tag_articles_batch
    tagged_articles = tag_articles_batch.delay(translated_articles).get()
    total_tags = sum(len(article.get('tags', [])) for article in tagged_articles)
    total_entities = sum(len(article.get('entities', [])) for article in tagged_articles)
    logger.info(f"✅ Tagged: {total_tags} total tags, {total_entities} total entities")
    
    # Step 3: Batch Embedding
    logger.info("🔢 Step 3: Batch Embedding")
    from workers.embedder import embed_articles_batch
    embedded_articles = embed_articles_batch.delay(tagged_articles).get()
    successful_embeddings = sum(1 for article in embedded_articles if article.get('embedding'))
    logger.info(f"✅ Embedded: {successful_embeddings}/{len(embedded_articles)} articles")
    
    # Step 4: Batch Clustering
    logger.info("🔗 Step 4: Batch Clustering")
    from workers.cluster import cluster_articles_batch
    clustering_result = cluster_articles_batch.delay(embedded_articles).get()
    clusters_created = clustering_result.get('clusters_created', 0)
    logger.info(f"✅ Clustering: {clusters_created} clusters created")
    
    logger.info("✅ Batch worker test completed")
    return embedded_articles, clustering_result

def test_full_pipeline():
    """Test the complete pipeline orchestration"""
    logger.info("🧪 Testing full pipeline orchestration")
    
    # Run the full pipeline
    start_time = time.time()
    pipeline_result = run_full_pipeline.delay().get()
    pipeline_time = time.time() - start_time
    
    logger.info(f"✅ Full pipeline completed in {pipeline_time:.2f}s")
    logger.info(f"📊 Results: {pipeline_result}")
    
    return pipeline_result

def main():
    """Run all tests"""
    logger.info("🚀 Starting Celery worker tests")
    
    try:
        # Test 1: Single worker
        logger.info("\n" + "="*50)
        logger.info("TEST 1: Single Worker")
        logger.info("="*50)
        test_single_worker()
        
        # Test 2: Batch workers
        logger.info("\n" + "="*50)
        logger.info("TEST 2: Batch Workers")
        logger.info("="*50)
        test_batch_workers()
        
        # Test 3: Full pipeline (optional - takes longer)
        logger.info("\n" + "="*50)
        logger.info("TEST 3: Full Pipeline")
        logger.info("="*50)
        logger.info("⚠️ This test will run the complete pipeline with scraping")
        logger.info("   It may take several minutes and requires Redis/Supabase")
        
        response = input("Run full pipeline test? (y/N): ")
        if response.lower() == 'y':
            test_full_pipeline()
        else:
            logger.info("Skipping full pipeline test")
        
        logger.info("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        raise

if __name__ == "__main__":
    main() 