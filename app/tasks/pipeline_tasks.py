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
    Apply comprehensive OSINT tagging using gazetteer, patterns, and NER
    """
    try:
        logger.info(f"🏷️  Tagging article {article.get('id', article.get('article_id', 'unknown'))}")
        
        # Import the tagging service
        from app.services.tagger import tag_article
        
        # Get article content
        content = article.get("cleaned_text", article.get("body", ""))
        title = article.get("title", "")
        
        if not content:
            logger.warning(f"⚠️ Article {article.get('id')} has no content for tagging")
            # Set default values
            article.update({
                "tag_categories": {},
                "tags": [],
                "entities": [],
                "confidence_score": 0.0,
                "priority_level": "LOW",
                "region": article.get("region", "Unknown"),
                "topic": article.get("topic", "General")
            })
            return article
        
        # Perform comprehensive tagging
        tagging_result = tag_article(content, title)
        
        # Extract entities from tags for backward compatibility
        entities = []
        for tag in tagging_result['tags']:
            if ':' in tag:
                entity = tag.split(":")[-1]
                if entity not in entities:
                    entities.append(entity)
        
        # Determine region and topic from tags if not already set
        region = article.get("region", "Unknown")
        topic = article.get("topic", "General")
        
        # Update region based on geographic tags
        geo_tags = tagging_result['tag_categories'].get('geo', [])
        if geo_tags:
            if any(geo in ['Taiwan', 'China', 'South China Sea', 'East China Sea'] for geo in geo_tags):
                region = "East Asia"
            elif any(geo in ['Philippines', 'Vietnam', 'Malaysia', 'Indonesia', 'Singapore'] for geo in geo_tags):
                region = "Southeast Asia"
            elif any(geo in ['Japan', 'South Korea'] for geo in geo_tags):
                region = "Northeast Asia"
            else:
                region = "Asia Pacific"
        
        # Update topic based on capability and event tags
        capability_tags = tagging_result['tag_categories'].get('capability', [])
        event_tags = tagging_result['tag_categories'].get('event', [])
        
        if any(cap in ['Cyber Warfare', 'Information Warfare', 'Cognitive Warfare'] for cap in capability_tags):
            topic = "Cybersecurity"
        elif any(event in ['Live Fire Drill', 'Military Exercise', 'Naval Patrol'] for event in event_tags):
            topic = "Maritime Security"
        elif any(cap in ['ISR', 'Surveillance'] for cap in capability_tags):
            topic = "Intelligence"
        else:
            topic = "General"
        
        # Update article with tagging results
        article.update({
            "tag_categories": tagging_result['tag_categories'],
            "tags": tagging_result['tags'],
            "entities": entities,
            "confidence_score": tagging_result['confidence_score'],
            "priority_level": tagging_result['priority_level'],
            "region": region,
            "topic": topic
        })
        
        logger.info(f"✅ Tagged article {article.get('id')}: {len(tagging_result['tags'])} tags, "
                   f"confidence={tagging_result['confidence_score']:.3f}, "
                   f"priority={tagging_result['priority_level']}, "
                   f"region={region}, topic={topic}")
        
        return article
        
    except Exception as e:
        logger.error(f"❌ Error tagging article {article.get('id', article.get('article_id'))}: {e}")
        # Set default values on error
        article.update({
            "tag_categories": {},
            "tags": [],
            "entities": [],
            "confidence_score": 0.0,
            "priority_level": "LOW",
            "region": article.get("region", "Unknown"),
            "topic": article.get("topic", "General")
        })
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def embed_and_cluster_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate embedding and assign to cluster using FAISS
    """
    try:
        logger.info(f"🔢 Embedding and clustering article {article.get('article_id', 'unknown')}")
        
        # Initialize models if not already loaded
        initialize_models()
        
        # Generate embedding
        text_to_embed = f"{article.get('title', '')} {article.get('cleaned_text', '')}"
        embedding = sbert_model.encode(text_to_embed)
        
        # Use FAISS for cluster assignment - ensure fresh data
        from app.services.faiss_cluster import assign_cluster, label_cluster, reload_faiss
        reload_faiss()  # Force reload to get latest data
        cluster_id = assign_cluster(embedding)
        
        # Generate cluster label with fresh data
        cluster_label, cluster_description = label_cluster(cluster_id)
        
        article["embedding"] = embedding.tolist()
        article["cluster_id"] = cluster_id
        article["cluster_label"] = cluster_label
        article["cluster_description"] = cluster_description
        article["embedding_dimensions"] = len(embedding)
        
        logger.info(f"✅ Embedded article {article.get('article_id')}: {len(embedding)} dims, cluster={cluster_id}, label='{cluster_label}'")
        return article
        
    except Exception as e:
        logger.error(f"❌ Error embedding article {article.get('article_id')}: {e}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def store_to_supabase(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store article to Supabase and update/create cluster entry
    """
    try:
        from app.services.supabase import get_supabase_client
        supabase = get_supabase_client()
        import uuid
        import datetime

        # Prepare article data
        article_data = {
            'id': article.get('article_id') or str(uuid.uuid4()),
            'title': article.get('title', ''),
            'url': article.get('source_url', ''),
            'content': article.get('body', ''),
            'cleaned': article.get('cleaned_text', ''),
            'tags': article.get('tags', []),
            'tag_categories': article.get('tag_categories', {}),
            'entities': article.get('entities', []),
            'region': article.get('region', 'Unknown'),
            'topic': article.get('topic', 'General'),
            'confidence_score': article.get('confidence_score', 0.0),
            'priority_level': article.get('priority_level', 'LOW'),
            'embedding': article.get('embedding', []),
            'embedding_dimensions': len(article.get('embedding', [])),
            'cluster_id': article.get('cluster_id', None),
            'cluster_label': article.get('cluster_label', ''),
            'inserted_at': datetime.datetime.now().isoformat(),
            'created_at': datetime.datetime.now().isoformat(),
            'updated_at': datetime.datetime.now().isoformat(),
            'status': 'processed',
        }
        # Store article
        res = supabase.table('articles').upsert(article_data).execute()
        logger.info(f"💾 Stored article {article_data['id']} to Supabase")

        # Handle cluster creation/update
        cluster_id = article_data.get('cluster_id')
        if cluster_id:
            # Try to fetch cluster
            cluster_res = supabase.table('clusters').select('*').eq('cluster_id', cluster_id).execute()
            if cluster_res.data and len(cluster_res.data) > 0:
                # Update existing cluster
                cluster = cluster_res.data[0]
                article_ids = cluster.get('article_ids') or []
                if article_data['id'] not in article_ids:
                    article_ids.append(article_data['id'])
                supabase.table('clusters').update({'article_ids': article_ids, 'updated_at': datetime.datetime.now().isoformat()}).eq('cluster_id', cluster_id).execute()
            else:
                # Create new cluster
                supabase.table('clusters').insert({
                    'cluster_id': cluster_id,
                    'article_ids': [article_data['id']],
                    'region': article_data['region'],
                    'topic': article_data['topic'],
                    'status': 'pending',
                    'created_at': datetime.datetime.now().isoformat(),
                    'updated_at': datetime.datetime.now().isoformat(),
                }).execute()

        # Update cluster label in clusters table
        if article.get('cluster_id') and article.get('cluster_label'):
            cluster_label = article.get('cluster_label', '')
            if cluster_label and cluster_label != 'Unlabeled':
                supabase.table('clusters').update({
                    'theme': cluster_label,
                    'updated_at': datetime.datetime.now().isoformat()
                }).eq('cluster_id', article['cluster_id']).execute()
                logger.info(f"🏷️ Updated cluster {article['cluster_id']} with label: {cluster_label}")

        return {'status': 'stored', 'article_id': article_data['id'], 'cluster_id': cluster_id}
    except Exception as e:
        logger.error(f"❌ Error storing article {article.get('article_id')}: {e}")
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
            from app.utils.supabase_client import get_supabase_client
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
            # Prepare article data for updated schema with enhanced tagging and clustering
            article_data = {
                "title": article["title"],
                "content": article.get("body", article.get("content", "")),
                "cleaned": article.get("cleaned_text", article.get("cleaned", "")),
                "url": article.get("source_url", article.get("url", f"https://pipeline.local/{article.get('id', article.get('article_id'))}")),
                "source": article.get("source", "StraitWatch Pipeline"),
                "region": article.get("region", ""),
                "topic": article.get("topic", ""),
                "tags": article.get("tags", []),
                "tag_categories": article.get("tag_categories", {}),
                "entities": article.get("entities", []),
                "cluster_id": str(article.get("cluster_id", "")),
                "cluster_label": article.get("cluster_label", ""),
                "cluster_description": article.get("cluster_description", ""),
                "embedding": article.get("embedding", []),
                "confidence_score": article.get("confidence_score", 0.0),
                "priority_level": article.get("priority_level", "LOW"),
                "embedding_dimensions": len(article.get("embedding", [])) if article.get("embedding") else None,
                "processed_by": "StraitWatch Pipeline v2 Enhanced Tagging & Clustering Batch"
            }
            
            # Only set original_id if it's a valid UUID
            article_id = article.get('id', article.get('article_id'))
            try:
                import uuid
                uuid.UUID(article_id)
                article_data["original_id"] = article_id
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
                "stored_schema": "updated_schema_v2_enhanced_tagging"
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