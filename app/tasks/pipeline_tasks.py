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
import hdbscan
import traceback

# Try to import SentenceTransformer, but make it optional
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError as e:
    print(f"SentenceTransformer not available: {e}")
    SENTENCE_TRANSFORMER_AVAILABLE = False
    SentenceTransformer = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Celery (same pattern as other task files)
celery_app = Celery('straitwatch')
celery_app.config_from_object('app.celery_config')

# === LOAD MODELS ONCE ===
logger.info("Loading models for pipeline...")
sbert_model = None
clusterer = None

def initialize_models():
    """Initialize models once when worker starts"""
    global sbert_model, clusterer
    
    if sbert_model is None and SENTENCE_TRANSFORMER_AVAILABLE:
        logger.info("Loading SentenceTransformer model...")
        sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("SUCCESS: SentenceTransformer model loaded")
    elif sbert_model is None and not SENTENCE_TRANSFORMER_AVAILABLE:
        logger.warning("SentenceTransformer not available, skipping model loading")
    
    if clusterer is None:
        logger.info("Initializing HDBSCAN clusterer...")
        # Initialize with proper parameters for maritime intelligence
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=2,
            min_samples=1,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        logger.info("SUCCESS: HDBSCAN clusterer initialized")

@celery_app.task(bind=True, max_retries=3)
def preprocess_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and preprocess article text
    """
    try:
        logger.info(f"[PIPELINE] Preprocessing article {article.get('article_id', 'unknown')}")
        
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
        
        logger.info(f"SUCCESS: Preprocessed article {article.get('article_id')}: {len(article['cleaned_text'])} chars")
        return article
        
    except Exception as e:
        logger.error(f"[PIPELINE] Error preprocessing article {article.get('article_id')}: {e}")
        logger.error(traceback.format_exc())
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def tag_article_ner(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tag article using comprehensive NER analysis
    """
    try:
        article_id = article.get('id', article.get('article_id', 'unknown'))
        logger.info(f"[PIPELINE] Tagging article {article_id}")
        logger.info(f"Article content length: {len(article.get('cleaned_text', article.get('body', '')))} chars")
        logger.info(f"Article title: {article.get('title', 'No title')}")
        
        from app.services.tagger import tag_article
        content = article.get("cleaned_text", article.get("body", ""))
        title = article.get("title", "")
        
        if not content:
            logger.warning(f"WARNING: Article {article_id} has no content for tagging")
            article.update({
                "tag_categories": {},
                "tags": [],
                "entities": [],
                "priority_level": "LOW",
                "region": article.get("region", "Unknown"),
                "topic": article.get("topic", "General")
            })
            return article
            
        logger.info(f"Calling tag_article with content length: {len(content)}")
        tagging_result = tag_article(content, title)
        
        # Extract entities from the comprehensive NER analysis
        entities = tagging_result.get('entities', [])
        
        # Determine region and topic from comprehensive tags
        region = article.get("region", "Unknown")
        topic = article.get("topic", "General")
        
        # Update region based on geographic tags
        geo_tags = tagging_result['tag_categories'].get('geo', [])
        if geo_tags:
            if any(geo in ['China', 'Taiwan', 'South China Sea', 'East China Sea'] for geo in geo_tags):
                region = "East Asia"
            elif any(geo in ['Philippines', 'Vietnam', 'Malaysia', 'Indonesia', 'Singapore'] for geo in geo_tags):
                region = "Southeast Asia"
            elif any(geo in ['Japan', 'South Korea'] for geo in geo_tags):
                region = "Northeast Asia"
            else:
                region = "Asia Pacific"
        
        # Update topic based on comprehensive entity analysis
        event_tags = tagging_result['tag_categories'].get('event', [])
        facility_tags = tagging_result['tag_categories'].get('facility', [])
        technology_tags = tagging_result['tag_categories'].get('technology', [])
        
        if event_tags or facility_tags:
            topic = "Security & Defense"
        elif technology_tags:
            topic = "Technology"
        elif tagging_result['tag_categories'].get('money'):
            topic = "Economy"
        elif tagging_result['tag_categories'].get('law'):
            topic = "Politics"
        else:
            topic = "General"
        
        article.update({
            "tag_categories": tagging_result['tag_categories'],
            "tags": tagging_result['tags'],
            "entities": entities,
            "priority_level": tagging_result['priority_level'],
            "region": region,
            "topic": topic
        })
        
        logger.info(f"SUCCESS: [PIPELINE] Tagged article {article.get('id')}: {len(tagging_result['tags'])} tags, {len(entities)} entities, priority={tagging_result['priority_level']}, region={region}, topic={topic}")
        
        # DETAILED LOGGING: Print all tagging results
        logger.info(f"[DETAILED-TAGGING] Article {article.get('id')} tagging results:")
        logger.info(f"   Tags: {tagging_result['tags']}")
        logger.info(f"   Entities: {entities}")
        logger.info(f"   Tag Categories: {tagging_result['tag_categories']}")
        logger.info(f"   Priority: {tagging_result['priority_level']}")
        logger.info(f"   Region: {region}")
        logger.info(f"   Topic: {topic}")
        
        return article
    except Exception as e:
        logger.error(f"[PIPELINE] ERROR: Tagging article {article.get('id', article.get('article_id'))}: {e}")
        logger.error(traceback.format_exc())
        article.update({
            "tag_categories": {},
            "tags": [],
            "entities": [],
            "priority_level": "LOW",
            "region": article.get("region", "Unknown"),
            "topic": article.get("topic", "General")
        })
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def embed_and_cluster_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate embedding and assign to cluster using fast clustering
    """
    try:
        logger.info(f"[PIPELINE] Embedding and clustering article {article.get('article_id', 'unknown')}")
        
        # Check if SentenceTransformer is available
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            logger.warning("SentenceTransformer not available, skipping embedding")
            article.update({
                "embedding": [],
                "cluster_id": "no_embedding",
                "cluster_label": "No Embedding",
                "cluster_description": "Article processed without embedding due to missing dependencies",
                "embedding_dimensions": 0
            })
            return article
        
        # Initialize models if not already loaded
        initialize_models()
        
        # Generate embedding
        text_to_embed = f"{article.get('title', '')} {article.get('cleaned_text', '')}"
        embedding = sbert_model.encode(text_to_embed)
        
        # For now, we'll assign a temporary cluster ID since fast clustering works on batches
        # The actual clustering will be done by the dedicated clustering task
        cluster_id = "temp_cluster"
        cluster_label = "Pending Clustering"
        cluster_description = "Article pending batch clustering"
        
        article["embedding"] = embedding.tolist()
        article["cluster_id"] = cluster_id
        article["cluster_label"] = cluster_label
        article["cluster_description"] = cluster_description
        article["embedding_dimensions"] = len(embedding)
        
        logger.info(f"SUCCESS: Embedded article {article.get('article_id')}: {len(embedding)} dims, cluster={cluster_id}, label='{cluster_label}'")
        return article
        
    except Exception as e:
        logger.error(f"[PIPELINE] Error embedding article {article.get('article_id')}: {e}")
        logger.error(traceback.format_exc())
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def store_to_supabase(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store article to Supabase with enhanced schema including tags and clustering
    """
    try:
        from app.services.supabase import store_article
        import uuid
        import datetime

        # Prepare article data
        article_id = article.get('article_id') or str(uuid.uuid4())
        title = article.get('title', '')
        raw_text = article.get('body', '')
        cleaned_text = article.get('cleaned_text', '')
        embedding = article.get('embedding', [])
        region = article.get('region', 'Unknown')
        topic = article.get('topic', 'General')
        source_url = article.get('source_url', '')
        
        # Enhanced tagging data
        tags = article.get('tags', [])
        tag_categories = article.get('tag_categories', {})
        entities = article.get('entities', [])
        
        # Clustering data
        cluster_id = article.get('cluster_id')
        cluster_label = article.get('cluster_label', '')
        cluster_description = article.get('cluster_description', '')
        
        # Store article with enhanced data
        db_id = store_article(
            article_id=article_id,
            title=title,
            raw_text=raw_text,
            cleaned_text=cleaned_text,
            embedding=embedding,
            region=region,
            topic=topic,
            source_url=source_url,
            tags=tags,
            tag_categories=tag_categories,
            entities=entities,
            cluster_id=cluster_id,
            cluster_label=cluster_label,
            cluster_description=cluster_description
        )
        
        if db_id:
            logger.info(f"SUCCESS: Article {article_id} stored successfully with {len(tags)} tags")
            return {'status': 'stored', 'article_id': article_id, 'db_id': db_id, 'cluster_id': cluster_id}
        else:
            logger.error(f"❌ Failed to store article {article_id}")
            raise Exception(f"Failed to store article {article_id}")
            
    except Exception as e:
        logger.error(f"[PIPELINE] Error storing article {article.get('article_id')}: {e}")
        logger.error(traceback.format_exc())
        raise self.retry(countdown=60, max_retries=3)

# === PIPELINE ORCHESTRATOR ===
@celery_app.task
def run_article_pipeline(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run the complete article processing pipeline
    """
    logger.info(f"[PIPELINE] Starting pipeline for article {getattr(article, 'id', 'unknown')}")
    try:
        logger.info("[PIPELINE] Preprocessing article...")
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
        logger.info(f"[PIPELINE] Finished pipeline for article {getattr(article, 'id', 'unknown')}")
        return {"status": "pipeline_started", "task_id": result.id, "article_id": article.get("article_id")}
    except Exception as e:
        logger.error(f"[PIPELINE] Error in pipeline for article {getattr(article, 'id', 'unknown')}: {e}")
        logger.error(traceback.format_exc())
        return {'status': 'error', 'error': str(e), 'traceback': traceback.format_exc()}

# === BATCH PROCESSING ===
@celery_app.task
def run_batch_pipeline(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple articles in batch
    """
    logger.info(f"STARTING: Batch pipeline for {len(articles)} articles")
    
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
    Store multiple processed articles to Supabase using enhanced storage function
    """
    try:
        logger.info(f"STORING: Batch storing {len(articles)} articles to Supabase")
        
        from app.services.supabase import store_article
        import uuid
        import datetime
        
        stored_count = 0
        failed_count = 0
        
        for article in articles:
            try:
                # Prepare article data
                article_id = article.get('article_id') or str(uuid.uuid4())
                title = article.get('title', '')
                raw_text = article.get('body', '')
                cleaned_text = article.get('cleaned_text', '')
                embedding = article.get('embedding', [])
                region = article.get('region', 'Unknown')
                topic = article.get('topic', 'General')
                source_url = article.get('source_url', '')
                
                # Enhanced tagging data
                tags = article.get('tags', [])
                tag_categories = article.get('tag_categories', {})
                entities = article.get('entities', [])
                
                # Clustering data
                cluster_id = article.get('cluster_id')
                cluster_label = article.get('cluster_label', '')
                cluster_description = article.get('cluster_description', '')
                
                # Store article with enhanced data using the enhanced function
                db_id = store_article(
                    article_id=article_id,
                    title=title,
                    raw_text=raw_text,
                    cleaned_text=cleaned_text,
                    embedding=embedding,
                    region=region,
                    topic=topic,
                    source_url=source_url,
                    tags=tags,
                    tag_categories=tag_categories,
                    entities=entities,
                    cluster_id=cluster_id,
                    cluster_label=cluster_label,
                    cluster_description=cluster_description
                )
                
                if db_id:
                    stored_count += 1
                    logger.info(f"✅ Stored article {article_id} with {len(tags)} tags")
                else:
                    failed_count += 1
                    logger.warning(f"⚠️ Failed to store article {article_id}")
                    
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Error storing article {article.get('article_id')}: {e}")
        
        logger.info(f"✅ Batch storage complete: {stored_count} stored, {failed_count} failed")
        
        return {
            "status": "batch_stored",
            "total_articles": len(articles),
            "stored_count": stored_count,
            "failed_count": failed_count,
            "stored_schema": "enhanced_schema_v2"
        }
        
    except Exception as e:
        logger.error(f"❌ Error in batch storage: {e}")
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def process_article_batch(self, articles: List[Dict[str, Any]], batch_size: int = 10) -> Dict[str, Any]:
    """
    Process multiple articles through the complete pipeline and store in batches
    """
    try:
        logger.info(f"PROCESSING: Processing batch of {len(articles)} articles")
        
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