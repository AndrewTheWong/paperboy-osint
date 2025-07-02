#!/usr/bin/env python3
"""
UPGRADE CELERY TASK PIPELINE FOR LOCAL SUPABASE

This pipeline processes articles via:
[Preprocess] → [NER Tag] → [Embed+Cluster] → [Store to Local Supabase]
"""

import logging
import re
import numpy as np
from typing import Dict, List, Any, Optional
from celery import Celery, chain
from sentence_transformers import SentenceTransformer
import hdbscan

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery (same pattern as other task files)
celery_app = Celery('straitwatch')
celery_app.config_from_object('app.celery_config')

# === LOAD MODELS ONCE ===
logger.info("🔄 Loading models for pipeline...")
sbert_model = None
clusterer = None

def initialize_models():
    """Initialize models once when worker starts"""
    global sbert_model, clusterer
    
    if sbert_model is None:
        logger.info("📚 Loading SentenceTransformer model...")
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("✅ SentenceTransformer model loaded")
    
    if clusterer is None:
        logger.info("🔍 Initializing HDBSCAN clusterer...")
        # Initialize with proper parameters for maritime intelligence
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        logger.info("✅ HDBSCAN clusterer initialized")

@celery_app.task(bind=True, max_retries=3)
def preprocess_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and preprocess article text
    """
    try:
        logger.info(f"🧹 Preprocessing article {article.get('article_id', 'unknown')}")
        
        def clean_text(text: str) -> str:
            """Clean HTML and normalize whitespace"""
            if not text:
                return ""
            
            # Remove HTML tags
            text = re.sub(r"<[^>]+>", "", text)
            # Normalize whitespace
            text = re.sub(r"\s+", " ", text).strip()
            # Remove special characters but keep punctuation
            text = re.sub(r"[^\w\s\.\,\!\?\;\:\-\(\)]", "", text)
            
            return text
        
        # Clean the article content
        article["cleaned_text"] = clean_text(article.get("body", ""))
        article["title"] = clean_text(article.get("title", ""))
        
        logger.info(f"✅ Preprocessed article {article.get('article_id')}: {len(article['cleaned_text'])} chars")
        return article
        
    except Exception as e:
        logger.error(f"❌ Error preprocessing article {article.get('article_id')}: {e}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def tag_article_ner(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply Named Entity Recognition and tagging
    """
    try:
        logger.info(f"🏷️  Tagging article {article.get('article_id', 'unknown')}")
        
        content = article.get("cleaned_text", "").lower()
        title = article.get("title", "").lower()
        full_text = f"{title} {content}"
        
        tags = []
        entities = []
        
        # Geographic entities
        geographic_terms = {
            "taiwan": "Taiwan",
            "china": "China", 
            "south china sea": "South China Sea",
            "strait of malacca": "Strait of Malacca",
            "singapore": "Singapore",
            "malaysia": "Malaysia",
            "philippines": "Philippines",
            "indonesia": "Indonesia",
            "vietnam": "Vietnam",
            "thailand": "Thailand",
            "myanmar": "Myanmar",
            "brunei": "Brunei"
        }
        
        # Security/Military entities  
        security_terms = {
            "naval": "Naval Operations",
            "military": "Military",
            "exercise": "Military Exercise",
            "cybersecurity": "Cybersecurity",
            "cyber": "Cyber Operations",
            "hacking": "Cyber Attack",
            "surveillance": "Surveillance",
            "intelligence": "Intelligence",
            "terrorism": "Terrorism",
            "piracy": "Maritime Piracy"
        }
        
        # Check for geographic entities
        for term, entity in geographic_terms.items():
            if term in full_text:
                if entity not in entities:
                    entities.append(entity)
                    tags.append(f"GEO:{entity}")
        
        # Check for security entities
        for term, entity in security_terms.items():
            if term in full_text:
                if entity not in entities:
                    entities.append(entity)
                    tags.append(f"SEC:{entity}")
        
        # Determine primary region
        region = article.get("region", "Unknown")
        if not region or region == "Unknown":
            if any(term in full_text for term in ["taiwan", "china", "south china sea"]):
                region = "East Asia"
            elif any(term in full_text for term in ["malacca", "singapore", "malaysia", "indonesia"]):
                region = "Southeast Asia"
            else:
                region = "Asia Pacific"
        
        # Determine primary topic
        topic = article.get("topic", "General")
        if not topic or topic == "General":
            if any(term in full_text for term in ["cyber", "hacking", "cybersecurity"]):
                topic = "Cybersecurity"
            elif any(term in full_text for term in ["naval", "military", "exercise"]):
                topic = "Maritime Security"
            elif any(term in full_text for term in ["terrorism", "intelligence"]):
                topic = "Security Intelligence"
            else:
                topic = "General"
        
        article["tags"] = tags
        article["entities"] = entities
        article["region"] = region
        article["topic"] = topic
        
        logger.info(f"✅ Tagged article {article.get('article_id')}: {len(tags)} tags, region={region}, topic={topic}")
        return article
        
    except Exception as e:
        logger.error(f"❌ Error tagging article {article.get('article_id')}: {e}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def embed_and_cluster_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate embedding and assign cluster
    """
    try:
        logger.info(f"🔢 Embedding and clustering article {article.get('article_id', 'unknown')}")
        
        # Initialize models if not already loaded
        initialize_models()
        
        # Generate embedding
        text_to_embed = f"{article.get('title', '')} {article.get('cleaned_text', '')}"
        embedding = sbert_model.encode(text_to_embed)
        
        # For now, assign a preliminary cluster ID based on topic similarity
        # In a full implementation, this would use existing clusters from database
        cluster_id = hash(f"{article.get('region', 'Unknown')}_{article.get('topic', 'General')}") % 1000
        
        article["embedding"] = embedding.tolist()
        article["cluster_id"] = cluster_id
        article["embedding_dimensions"] = len(embedding)
        
        logger.info(f"✅ Embedded article {article.get('article_id')}: {len(embedding)} dimensions, cluster={cluster_id}")
        return article
        
    except Exception as e:
        logger.error(f"❌ Error embedding article {article.get('article_id')}: {e}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def store_to_supabase(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store processed article to local Supabase (Updated Schema)
    """
    try:
        logger.info(f"💾 Storing article {article.get('article_id', 'unknown')} to Supabase")
        
        # Import Supabase client with proper path handling
        import sys
        import os
        
        # Add project root to path
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        try:
            from utils.supabase_client import get_supabase_client
            supabase = get_supabase_client()
        except ImportError:
            # Fallback to direct Supabase client creation
            from supabase import create_client, Client
            SUPABASE_URL = "http://localhost:54321"
            SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Prepare article data for updated schema
        article_data = {
            "title": article["title"],
            "content": article.get("body", article.get("content", "")),
            "cleaned": article.get("cleaned_text", article.get("cleaned", "")),
            "url": article.get("source_url", article.get("url", f"https://pipeline.local/{article['article_id']}")),
            "source": article.get("source", "StraitWatch Pipeline"),
            "region": article.get("region", ""),
            "topic": article.get("topic", ""),
            "tags": article.get("tags", []),
            "entities": article.get("entities", []),
            "cluster_id": str(article.get("cluster_id", "")),
            "confidence_score": article.get("confidence_score"),
            "embedding_dimensions": len(article.get("embedding", [])) if article.get("embedding") else None,
            "processed_by": "StraitWatch Pipeline v2"
        }
        
        # Only set original_id if it's a valid UUID
        try:
            import uuid
            uuid.UUID(article["article_id"])
            article_data["original_id"] = article["article_id"]
        except (ValueError, TypeError):
            # If not a valid UUID, let the database auto-generate the ID
            logger.info(f"Article ID '{article['article_id']}' is not a valid UUID, letting DB auto-generate")
        
        # Insert into articles table (let database auto-generate ID)
        response = supabase.table("articles").insert(article_data).execute()
        
        if response.data:
            db_id = response.data[0].get("id")
            logger.info(f"✅ Stored article {article.get('article_id')} to Supabase with DB ID {db_id}")
            
            return {
                "status": "stored",
                "article_id": article.get("article_id"),
                "database_id": db_id,
                "cluster_id": article.get("cluster_id"),
                "tags_count": len(article.get("tags", [])),
                "entities_count": len(article.get("entities", [])),
                "embedding_dimensions": len(article.get("embedding", [])) if article.get("embedding") else 0,
                "stored_schema": "updated_schema_v2"
            }
        else:
            raise Exception("No data returned from Supabase insert")
        
    except Exception as e:
        logger.error(f"❌ Error storing article {article.get('article_id')} to Supabase: {e}")
        raise self.retry(countdown=60, max_retries=3)

# === PIPELINE ORCHESTRATOR ===
@celery_app.task
def run_article_pipeline(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete article processing pipeline
    """
    logger.info(f"🚀 Starting pipeline for article {article.get('article_id', 'unknown')}")
    
    # Create and execute the processing chain
    pipeline = chain(
        preprocess_article.s(article),
        tag_article_ner.s(),
        embed_and_cluster_article.s(),
        store_to_supabase.s()
    )
    
    # Execute the pipeline
    result = pipeline.apply_async()
    
    logger.info(f"🎯 Pipeline initiated for article {article.get('article_id')}")
    return {"status": "pipeline_started", "task_id": result.id, "article_id": article.get("article_id")}

# === BATCH PROCESSING ===
@celery_app.task
def run_batch_pipeline(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple articles in batch
    """
    logger.info(f"🔄 Starting batch pipeline for {len(articles)} articles")
    
    task_ids = []
    for article in articles:
        result = run_article_pipeline.delay(article)
        task_ids.append(result.id)
    
    logger.info(f"✅ Batch pipeline started: {len(task_ids)} tasks")
    return {
        "status": "batch_started",
        "article_count": len(articles),
        "task_ids": task_ids
    }

# === BATCH STORAGE FUNCTIONS ===
@celery_app.task(bind=True, max_retries=3)
def store_batch_to_supabase(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Store multiple processed articles to Supabase in a single batch operation
    """
    try:
        logger.info(f"💾 Batch storing {len(articles)} articles to Supabase")
        
        # Import Supabase client with proper path handling
        import sys
        import os
        
        # Add project root to path
        project_root = os.path.join(os.path.dirname(__file__), '..', '..')
        if project_root not in sys.path:
            sys.path.append(project_root)
        
        try:
            from utils.supabase_client import get_supabase_client
            supabase = get_supabase_client()
        except ImportError:
            # Fallback to direct Supabase client creation
            from supabase import create_client, Client
            SUPABASE_URL = "http://localhost:54321"
            SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
            supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Prepare batch data for updated schema
        batch_data = []
        
        for article in articles:
            # Prepare article data for updated schema
            article_data = {
                "title": article["title"],
                "content": article.get("body", article.get("content", "")),
                "cleaned": article.get("cleaned_text", article.get("cleaned", "")),
                "url": article.get("source_url", article.get("url", f"https://pipeline.local/{article['article_id']}")),
                "source": article.get("source", "StraitWatch Pipeline"),
                "region": article.get("region", ""),
                "topic": article.get("topic", ""),
                "tags": article.get("tags", []),
                "entities": article.get("entities", []),
                "cluster_id": str(article.get("cluster_id", "")),
                "confidence_score": article.get("confidence_score"),
                "embedding_dimensions": len(article.get("embedding", [])) if article.get("embedding") else None,
                "processed_by": "StraitWatch Pipeline v2 Batch"
            }
            
            # Only set original_id if it's a valid UUID
            try:
                import uuid
                uuid.UUID(article["article_id"])
                article_data["original_id"] = article["article_id"]
            except (ValueError, TypeError):
                # If not a valid UUID, let the database auto-generate the ID
                pass
            batch_data.append(article_data)
        
        # Single batch insert
        response = supabase.table("articles").insert(batch_data).execute()
        
        if response.data and len(response.data) == len(articles):
            stored_ids = [row.get("id") for row in response.data]
            logger.info(f"✅ Batch stored {len(stored_ids)} articles to Supabase with DB IDs: {stored_ids[:5]}{'...' if len(stored_ids) > 5 else ''}")
            
            return {
                "status": "batch_stored",
                "article_count": len(articles),
                "database_ids": stored_ids,
                "stored_schema": "updated_schema_v2"
            }
        else:
            raise Exception(f"Batch insert failed: expected {len(articles)} records, got {len(response.data) if response.data else 0}")
        
    except Exception as e:
        logger.error(f"❌ Error batch storing articles to Supabase: {e}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def process_article_batch(self, articles: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
    """
    Process multiple articles through the complete pipeline and store in batches
    """
    try:
        logger.info(f"🔄 Processing batch of {len(articles)} articles")
        
        # Create individual pipeline chains for each article
        pipeline_tasks = []
        for article in articles:
            # Create a chain for each article: preprocess -> tag -> embed -> store
            pipeline = chain(
                preprocess_article.s(article),
                tag_article_ner.s(),
                embed_and_cluster_article.s(),
                store_to_supabase.s()
            )
            pipeline_tasks.append(pipeline.apply_async())
        
        logger.info(f"✅ Started {len(pipeline_tasks)} individual article pipelines")
        
        return {
            "status": "batch_pipelines_started",
            "total_articles": len(articles),
            "task_ids": [task.id for task in pipeline_tasks]
        }
        
    except Exception as e:
        logger.error(f"❌ Error in batch processing: {e}")
        raise self.retry(countdown=60, max_retries=3) 