#!/usr/bin/env python3
"""
Test script for the upgraded StraitWatch pipeline
Tests: Preprocess → NER Tag → Embed+Cluster → Store to Local Supabase
"""

import requests
import json
import time
from typing import List, Dict

# API Configuration
API_BASE = "http://localhost:8000"
INGEST_ENDPOINT_V2 = f"{API_BASE}/ingest/v2/"
BATCH_ENDPOINT_V2 = f"{API_BASE}/ingest/v2/batch/"
STATUS_ENDPOINT = f"{API_BASE}/ingest/status"

def test_single_article():
    """Test single article processing with upgraded pipeline"""
    print("🧪 Testing Single Article - Upgraded Pipeline")
    print("=" * 50)
    
    article = {
        "title": "China Naval Exercises in South China Sea",
        "body": """
        <p>The Chinese People's Liberation Army Navy (PLAN) has announced large-scale 
        military exercises in the South China Sea this week. The exercises include naval 
        vessels, submarines, and cybersecurity operations targeting regional intelligence 
        networks.</p>
        
        <p>Taiwan has expressed concerns over these military activities near the Taiwan Strait. 
        The exercises are seen as a demonstration of China's growing naval capabilities and 
        surveillance technology in the region.</p>
        """,
        "region": "East Asia",
        "topic": "Maritime Security",
        "source_url": "https://example.com/china-naval-exercises-2025"
    }
    
    try:
        response = requests.post(INGEST_ENDPOINT_V2, json=article)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Article ingested successfully!")
        print(f"   ID: {result['id']}")
        print(f"   Status: {result['status']}")
        print(f"   Message: {result['message']}")
        
        return result['id']
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to ingest article: {e}")
        return None

def test_batch_articles():
    """Test batch article processing"""
    print("\n🧪 Testing Batch Articles - Upgraded Pipeline")
    print("=" * 50)
    
    articles = [
        {
            "title": "Strait of Malacca Cyber Security Alert",
            "body": """
            Singapore and Malaysia have issued a joint cybersecurity alert regarding 
            increased hacking attempts targeting maritime shipping systems in the 
            Strait of Malacca. The cyber attacks appear to be targeting navigation 
            systems and port operations.
            """,
            "region": "Southeast Asia",
            "topic": "Cybersecurity",
            "source_url": "https://example.com/malacca-cyber-alert"
        },
        {
            "title": "Philippines Naval Intelligence Report",
            "body": """
            The Philippine Navy has released an intelligence assessment covering 
            maritime surveillance operations in the West Philippine Sea. The report 
            highlights increased naval activity and potential security threats in 
            the region.
            """,
            "region": "Southeast Asia", 
            "topic": "Security Intelligence",
            "source_url": "https://example.com/philippines-naval-intel"
        },
        {
            "title": "Vietnam Maritime Piracy Prevention",
            "body": """
            Vietnamese authorities have announced new measures to combat maritime 
            piracy in the South China Sea. The initiative includes enhanced naval 
            patrols and international cooperation with regional partners.
            """,
            "region": "Southeast Asia",
            "topic": "Maritime Security", 
            "source_url": "https://example.com/vietnam-piracy-prevention"
        }
    ]
    
    try:
        response = requests.post(BATCH_ENDPOINT_V2, json=articles)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ Batch ingested successfully!")
        print(f"   Article Count: {result['article_count']}")
        print(f"   Status: {result['status']}")
        print(f"   Task ID: {result['task_id']}")
        print(f"   Pipeline: {result['pipeline']}")
        
        print("\n📋 Articles in batch:")
        for article in result['articles']:
            print(f"   - {article['id']}: {article['title']}")
            
        return result['task_id']
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to ingest batch: {e}")
        return None

def check_pipeline_status():
    """Check the overall pipeline status"""
    print("\n📊 Checking Pipeline Status")
    print("=" * 30)
    
    try:
        response = requests.get(STATUS_ENDPOINT)
        response.raise_for_status()
        
        status = response.json()
        print(f"Status: {status.get('status', 'unknown')}")
        print(f"Total Articles: {status.get('total_articles', 0)}")
        print(f"Processed: {status.get('processed_articles', 0)}")
        print(f"Unprocessed: {status.get('unprocessed_articles', 0)}")
        print(f"Pipeline: {status.get('pipeline', 'unknown')}")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to get status: {e}")

def test_ner_entities():
    """Test NER entity extraction with specific content"""
    print("\n🧪 Testing NER Entity Extraction")
    print("=" * 40)
    
    test_article = {
        "title": "Taiwan Cybersecurity Threat from China",
        "body": """
        Intelligence reports indicate that China has been conducting sophisticated 
        cyber operations targeting Taiwan's critical infrastructure. The hacking 
        attempts have focused on naval command systems and surveillance networks.
        
        Singapore has offered cybersecurity assistance to Taiwan, while Malaysia 
        has reported similar cyber attacks on their maritime shipping systems in 
        the Strait of Malacca.
        
        Military experts believe these cyber operations are part of broader 
        intelligence gathering activities in the South China Sea region.
        """,
        "region": "East Asia",
        "topic": "Cybersecurity",
        "source_url": "https://example.com/taiwan-cyber-threat"
    }
    
    print("📝 Article Content:")
    print(f"   Title: {test_article['title']}")
    print(f"   Expected Geographic Entities: Taiwan, China, Singapore, Malaysia, Strait of Malacca, South China Sea")
    print(f"   Expected Security Entities: Cybersecurity, Hacking, Intelligence, Naval, Surveillance, Military")
    
    try:
        response = requests.post(INGEST_ENDPOINT_V2, json=test_article)
        response.raise_for_status()
        
        result = response.json()
        print(f"✅ NER test article submitted: {result['id']}")
        print(f"   Pipeline will extract entities and store with tags")
        
        return result['id']
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Failed to submit NER test: {e}")
        return None

def main():
    """Run all pipeline tests"""
    print("🚀 StraitWatch Upgraded Pipeline Tests")
    print("=" * 60)
    print("Pipeline: [Preprocess] → [NER Tag] → [Embed+Cluster] → [Store to Supabase]")
    print()
    
    # Check initial status
    check_pipeline_status()
    
    # Test single article
    single_id = test_single_article()
    time.sleep(2)
    
    # Test batch processing
    batch_task_id = test_batch_articles()
    time.sleep(2)
    
    # Test NER extraction
    ner_id = test_ner_entities()
    time.sleep(2)
    
    # Final status check
    print("\n" + "=" * 60)
    check_pipeline_status()
    
    print("\n🎯 Test Summary:")
    print(f"   Single Article ID: {single_id or 'Failed'}")
    print(f"   Batch Task ID: {batch_task_id or 'Failed'}")
    print(f"   NER Test ID: {ner_id or 'Failed'}")
    
    print("\n📋 Expected Processing Steps:")
    print("   1. 🧹 Clean HTML and normalize text")
    print("   2. 🏷️  Extract geographic and security entities")
    print("   3. 🔢 Generate embeddings and assign clusters")
    print("   4. 💾 Store to Supabase with tags and entities")
    
    print("\n💡 To monitor progress:")
    print("   - Check Celery worker logs for task execution")
    print("   - Query Supabase articles table for stored results")
    print("   - Look for tags and entities fields in database")

if __name__ == "__main__":
    main() 