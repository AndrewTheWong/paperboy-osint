#!/usr/bin/env python3
"""
Test the complete Paperboy pipeline
"""

import requests
import time
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_full_pipeline():
    """Test the complete pipeline from scraping to clustering"""
    
    base_url = "http://localhost:8000"
    
    # Wait for server to be ready
    logger.info("🔄 Waiting for server to be ready...")
    for i in range(30):
        try:
            response = requests.get(f"{base_url}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Server is ready")
                break
        except:
            time.sleep(1)
    else:
        logger.error("❌ Server not ready after 30 seconds")
        return
    
    # Step 1: Test scraper endpoint
    logger.info("📰 Testing scraper endpoint...")
    try:
        response = requests.post(
            f"{base_url}/scraper/run",
            json={
                "use_default_sources": True,
                "max_articles_per_source": 5
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Scraper queued tasks: {result}")
        else:
            logger.error(f"❌ Scraper failed: {response.status_code} - {response.text}")
            return
            
    except Exception as e:
        logger.error(f"❌ Scraper error: {e}")
        return
    
    # Wait for preprocessing
    logger.info("⏳ Waiting for preprocessing to complete...")
    time.sleep(10)
    
    # Step 2: Test ingest endpoint
    logger.info("📥 Testing ingest endpoint...")
    try:
        test_article = {
            "title": "Test Article - Pipeline Test",
            "body": "This is a test article to verify the pipeline is working correctly. It contains sample content for testing purposes.",
            "source_url": "https://example.com/test-article",
            "region": "Test",
            "topic": "Technology"
        }
        
        response = requests.post(
            f"{base_url}/ingest",
            json=test_article,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Article ingested: {result}")
        else:
            logger.error(f"❌ Ingest failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Ingest error: {e}")
    
    # Wait for processing
    logger.info("⏳ Waiting for processing to complete...")
    time.sleep(15)
    
    # Step 3: Test clustering endpoint
    logger.info("🔍 Testing clustering endpoint...")
    try:
        response = requests.post(
            f"{base_url}/cluster/run",
            json={},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Clustering result: {result}")
        else:
            logger.error(f"❌ Clustering failed: {response.status_code} - {response.text}")
            
    except Exception as e:
        logger.error(f"❌ Clustering error: {e}")
    
    # Step 4: Check Redis queue status
    logger.info("📊 Checking Redis queue status...")
    try:
        response = requests.get(f"{base_url}/ingest/status", timeout=10)
        if response.status_code == 200:
            status = response.json()
            logger.info(f"📊 Queue status: {status}")
        else:
            logger.error(f"❌ Status check failed: {response.status_code}")
            
    except Exception as e:
        logger.error(f"❌ Status check error: {e}")
    
    logger.info("🎉 Pipeline test complete!")

if __name__ == "__main__":
    test_full_pipeline() 