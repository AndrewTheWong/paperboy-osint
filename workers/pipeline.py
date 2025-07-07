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
import asyncio
from services.async_scraper import RobustAsyncScraper
from services.tagger import tag_article_batch
from services.faiss_index import get_faiss_service

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
celery_app.config_from_object('config.celery_config')

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

def run_async_scrape(sources):
    scraper = RobustAsyncScraper()
    # Use the same event loop handling as in scraper.py
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(scraper.scrape_sources(sources))
            finally:
                loop.close()
        else:
            return loop.run_until_complete(scraper.scrape_sources(sources))
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(scraper.scrape_sources(sources))
        finally:
            loop.close()

@celery_app.task(bind=True, max_retries=3)
def preprocess_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Clean and preprocess article text
    """
    try:
        logger.info(f"[PIPELINE] Preprocessing article {article.get('article_id', 'unknown')}")
        
        # Clean the article content
        article["cleaned_text"] = article.get("body", "")
        article["title"] = article.get("title", "")
        
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
        
        from services.tagger import tag_article
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
        
                    # Calculate priority level based on entities and tags
            priority_level = "LOW"
            if entities:
                # Check for high-priority entities
                high_priority_entities = ["China", "Taiwan", "South China Sea", "US Navy", "Philippines", "Japan"]
                if any(entity in high_priority_entities for entity in entities):
                    priority_level = "HIGH"
                elif len(entities) > 3:
                    priority_level = "MEDIUM"
            
            article.update({
                "tag_categories": tagging_result['tag_categories'],
                "tags": tagging_result['tags'],
                "entities": entities,
                "priority_level": priority_level,
                "region": region,
                "topic": topic
            })
        
        logger.info(f"SUCCESS: [PIPELINE] Tagged article {article.get('id')}: {len(tagging_result['tags'])} tags, {len(entities)} entities, priority={priority_level}, region={region}, topic={topic}")
        
        # DETAILED LOGGING: Print all tagging results
        logger.info(f"[DETAILED-TAGGING] Article {article.get('id')} tagging results:")
        logger.info(f"   Tags: {tagging_result['tags']}")
        logger.info(f"   Entities: {entities}")
        logger.info(f"   Tag Categories: {tagging_result['tag_categories']}")
        logger.info(f"   Priority: {priority_level}")
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
    Generate embedding and assign to cluster using real clustering service
    """
    try:
        logger.info(f"[PIPELINE] Embedding and clustering article {article.get('article_id', 'unknown')}")
        
        # Check if SentenceTransformer is available
        if not SENTENCE_TRANSFORMER_AVAILABLE:
            logger.warning("SentenceTransformer not available, using mock embedding")
            # Generate mock embedding for testing
            np.random.seed(hash(article.get('title', '')) % 2**32)
            embedding = np.random.normal(0, 1, 384).tolist()
        else:
            # Initialize models if not already loaded
            initialize_models()
            
            # Generate embedding
            text_to_embed = f"{article.get('title', '')} {article.get('cleaned_text', '')}"
            embedding = sbert_model.encode(text_to_embed).tolist()
        
        # Store embedding in article
        article["embedding"] = embedding
        article["embedding_dimensions"] = len(embedding)
        
        # Use real clustering service
        try:
            from services.clusterer import cluster_articles_complete
            
            # Create a single-article batch for clustering
            articles_batch = [article]
            embeddings_batch = [embedding]
            
            # Perform clustering
            clustering_result = cluster_articles_complete(embeddings_batch, articles_batch)
            
            # Extract cluster information
            if clustering_result and 'clusters' in clustering_result:
                clusters = clustering_result['clusters']
                cluster_summaries = clustering_result.get('summaries', {})
                # Find which cluster this article belongs to
                cluster_id = None
                cluster_label = "Unclustered"
                cluster_description = "Article not assigned to any cluster"
                
                # Get article ID (handle both 'id' and 'article_id' fields)
                article_id = article.get('id') or article.get('article_id') or 'unknown'
                
                for cluster_info in clusters.values():
                    if article_id in cluster_info.get('article_ids', []):
                        cluster_id = cluster_info.get('cluster_id', 'cluster_1')
                        cluster_label = cluster_info.get('representative_title', 'Cluster')
                        cluster_description = f"Cluster with {len(cluster_info.get('article_ids', []))} articles"
                        break
                
                if cluster_id is None:
                    # Create individual cluster for this article
                    cluster_id = f"cluster_{article_id}"
                    cluster_label = article.get('title', 'Single Article')
                    cluster_description = "Individual article cluster"
                
                article["cluster_id"] = cluster_id
                article["cluster_label"] = cluster_label
                article["cluster_description"] = cluster_description
                
            else:
                # Fallback to individual cluster
                article_id = article.get('id') or article.get('article_id') or 'single'
                cluster_id = f"cluster_{article_id}"
                article["cluster_id"] = cluster_id
                article["cluster_label"] = article.get('title', 'Single Article')
                article["cluster_description"] = "Individual article cluster"
                
        except Exception as clustering_error:
            logger.warning(f"Clustering failed, using fallback: {clustering_error}")
            # Fallback to individual cluster
            article_id = article.get('id') or article.get('article_id') or 'single'
            cluster_id = f"cluster_{article_id}"
            article["cluster_id"] = cluster_id
            article["cluster_label"] = article.get('title', 'Single Article')
            article["cluster_description"] = "Individual article cluster (fallback)"
        
        logger.info(f"SUCCESS: Embedded and clustered article {article.get('article_id')}: {len(embedding)} dims, cluster={article.get('cluster_id')}")
        return article
        
    except Exception as e:
        logger.error(f"[PIPELINE] Error embedding/clustering article {article.get('article_id')}: {e}")
        logger.error(traceback.format_exc())
        raise self.retry(countdown=60, max_retries=3)

@celery_app.task(bind=True, max_retries=3)
def store_to_supabase(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store article to Supabase with enhanced schema including tags and clustering
    """
    try:
        from db.supabase_client import store_article
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

@celery_app.task(bind=True, max_retries=3)
def store_to_faiss(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Store article embedding to Faiss index
    """
    try:
        article_id = article.get('id', article.get('article_id', 'unknown'))
        logger.info(f"[PIPELINE] Storing article {article_id} to Faiss")
        
        # Get the embedding from the article
        embedding = article.get('embedding')
        if not embedding:
            logger.warning(f"WARNING: Article {article_id} has no embedding for Faiss storage")
            return {
                "status": "failed",
                "message": "No embedding available",
                "article_id": article_id
            }
        
        # Get Faiss service and store embedding
        faiss_service = get_faiss_service()
        if not faiss_service:
            logger.error(f"ERROR: Faiss service not available for article {article_id}")
            return {
                "status": "failed", 
                "message": "Faiss service not available",
                "article_id": article_id
            }
        
        # Store the embedding
        success = faiss_service.add_embedding(embedding, article_id)
        
        if success:
            logger.info(f"SUCCESS: [PIPELINE] Stored article {article_id} to Faiss")
            return {
                "status": "success",
                "message": "Embedding stored to Faiss",
                "article_id": article_id
            }
        else:
            logger.error(f"ERROR: [PIPELINE] Failed to store article {article_id} to Faiss")
            return {
                "status": "failed",
                "message": "Failed to store embedding",
                "article_id": article_id
            }
            
    except Exception as e:
        logger.error(f"[PIPELINE] ERROR: Storing article {article.get('id', article.get('article_id'))} to Faiss: {e}")
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
            store_to_supabase.s(),
            store_to_faiss.s()
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

def embed_articles_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a batch of articles using SentenceTransformer
    """
    logger.info(f"[EMBEDDING] Starting batch embedding for {len(articles)} articles")
    
    try:
        from services.embedder import generate_embeddings_batch
        
        # Prepare texts for embedding
        texts = []
        for article in articles:
            title = article.get('title', '')
            content = article.get('cleaned_text', article.get('body', ''))
            text_to_embed = f"{title} {content}".strip()
            texts.append(text_to_embed)
        
        # Generate embeddings in batch
        embeddings = generate_embeddings_batch(texts)
        
        # Add embeddings to articles
        for i, article in enumerate(articles):
            article['embedding'] = embeddings[i]
            article['embedding_dimensions'] = len(embeddings[i])
        
        logger.info(f"[EMBEDDING] Successfully embedded {len(articles)} articles")
        return articles
        
    except Exception as e:
        logger.error(f"[EMBEDDING] Error in batch embedding: {e}")
        # Return articles with mock embeddings if embedding fails
        for article in articles:
            import numpy as np
            np.random.seed(hash(article.get('title', '')) % 2**32)
            article['embedding'] = np.random.normal(0, 1, 384).tolist()
            article['embedding_dimensions'] = 384
        return articles

@celery_app.task
def run_batch_pipeline(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Process multiple articles in batch using batch tagging and embedding
    """
    logger.info(f"STARTING: Batch pipeline for {len(articles)} articles")
    
    # Batch preprocess (if needed)
    for article in articles:
        # Map async scraper fields to expected fields
        article["cleaned_text"] = article.get("content", article.get("body", ""))
        article["title"] = article.get("title", "")
        article["body"] = article.get("content", article.get("body", ""))
        article["url"] = article.get("url", "")
    
    # Batch translate
    logger.info("🔄 Starting batch translation...")
    from services.translator import translate_articles_batch_simple
    articles = translate_articles_batch_simple(articles)
    
    # Batch tag
    logger.info("🔄 Starting batch tagging...")
    tagging_results = tag_article_batch(articles, text_key="cleaned_text", title_key="title_translated")
    for article, tag_result in zip(articles, tagging_results):
        article.update(tag_result)
    
    # Batch embed
    logger.info("🔄 Starting batch embedding...")
    articles = embed_articles_batch(articles)
    
    # Store to Supabase
    logger.info("🔄 Starting batch storage to Supabase...")
    stored_count = 0
    for article in articles:
        try:
            from db.supabase_client import store_article
            import uuid
            
            # Prepare article data
            article_id = article.get('article_id') or str(uuid.uuid4())
            title = article.get('title_translated', article.get('title', ''))
            title_original = article.get('title_original', article.get('title', ''))
            raw_text = article.get('content_translated', article.get('body', ''))
            raw_text_original = article.get('content_original', article.get('body', ''))
            cleaned_text = article.get('cleaned_text', '')
            embedding = article.get('embedding', [])
            region = article.get('region', 'Unknown')
            topic = article.get('topic', 'General')
            source_url = article.get('url', '')
            
            # Language information
            title_language = article.get('title_language', 'en')
            content_language = article.get('content_language', 'en')
            
            # Enhanced tagging data
            tags = article.get('tags', [])
            tag_categories = article.get('tag_categories', {})
            entities = article.get('entities', [])
            
            # Store article with enhanced data including translation info
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
                cluster_id=None,  # Will be set by clustering later
                cluster_label='',
                cluster_description='',
                title_original=title_original,
                content_original=raw_text_original,
                title_language=title_language,
                content_language=content_language
            )
            
            if db_id:
                stored_count += 1
                logger.info(f"✅ Stored article {article_id} with {len(tags)} tags")
            else:
                logger.warning(f"⚠️ Failed to store article {article_id}")
                
        except Exception as e:
            logger.error(f"❌ Error storing article {article.get('article_id')}: {e}")
    
    # Store embeddings in Faiss
    logger.info("🔄 Starting batch storage to Faiss...")
    faiss_service = get_faiss_service()
    faiss_count = faiss_service.add_embeddings_batch(articles)
    
    # Perform clustering
    logger.info("🔄 Starting clustering...")
    from services.clusterer import cluster_articles_complete
    
    # Extract embeddings for clustering
    embeddings = [article.get('embedding', []) for article in articles]
    
    # Filter out articles without embeddings
    valid_articles = []
    valid_embeddings = []
    for article, embedding in zip(articles, embeddings):
        if embedding and len(embedding) > 0:
            valid_articles.append(article)
            valid_embeddings.append(embedding)
    
    if valid_embeddings:
        # Perform clustering
        clustering_result = cluster_articles_complete(valid_embeddings, valid_articles)
        clusters = clustering_result.get('clusters', {})
        cluster_summaries = clustering_result.get('cluster_summaries', {})
        
        # Assign cluster IDs to articles
        for cluster_id, article_indices in clusters.items():
            for idx in article_indices:
                if idx < len(valid_articles):
                    article = valid_articles[idx]
                    article['cluster_id'] = f"cluster_{cluster_id}"
                    
                    # Add cluster metadata
                    if cluster_id in cluster_summaries:
                        summary = cluster_summaries[cluster_id]
                        article['cluster_label'] = summary.get('primary_topic', 'Unknown')
                        article['cluster_description'] = f"Cluster with {summary.get('size', 0)} articles"
        
        # Store clusters to database
        logger.info("🔄 Storing clusters to database...")
        from db.supabase_client import save_cluster
        
        for cluster_id, summary in cluster_summaries.items():
            article_ids = [valid_articles[idx].get('article_id', f'article_{idx}') for idx in clusters.get(cluster_id, [])]
            theme = summary.get('primary_topic', 'Unknown')
            cluster_summary = f"Cluster with {summary.get('size', 0)} articles about {theme}"
            
            save_cluster(
                cluster_id=f"cluster_{cluster_id}",
                article_ids=article_ids,
                status='complete',
                theme=theme,
                summary=cluster_summary
            )
        
        logger.info(f"✅ Clustering complete: {len(clusters)} clusters created")
    else:
        logger.warning("⚠️ No valid embeddings for clustering")
    
    logger.info(f"✅ Batch pipeline complete: {stored_count}/{len(articles)} articles stored to Supabase, {faiss_count}/{len(articles)} embeddings stored to Faiss")
    return {
        "status": "batch_complete",
        "article_count": len(articles),
        "stored_count": stored_count,
        "faiss_count": faiss_count,
        "clusters_created": len(clusters) if 'clusters' in locals() else 0
    }

# === BATCH STORAGE FUNCTIONS ===
@celery_app.task(bind=True, max_retries=3)
def store_batch_to_supabase(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Store multiple processed articles to Supabase using enhanced storage function
    """
    try:
        logger.info(f"STORING: Batch storing {len(articles)} articles to Supabase")
        
        from db.supabase_client import store_article
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
    Process multiple articles through the complete pipeline and store in batches, using batch tagging
    """
    try:
        logger.info(f"PROCESSING: Processing batch of {len(articles)} articles")
        # Batch preprocess (if needed)
        # Batch tag
        tagging_results = tag_article_batch(articles, text_key="cleaned_text", title_key="title")
        for article, tag_result in zip(articles, tagging_results):
            article.update(tag_result)
        # Continue with embedding, clustering, and storage as before
        pipeline_tasks = []
        for article in articles:
            pipeline = chain(
                embed_and_cluster_article.s(article),
                store_to_supabase.s(),
                store_to_faiss.s()
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

if __name__ == "__main__":
    """Main entry point for the pipeline"""
    import json
    import os
    
    logger.info("🚀 Starting StraitWatch Pipeline")
    
    try:
        # Load sources
        sources_path = "sources/taiwan_strait_sources.json"
        if not os.path.exists(sources_path):
            logger.error(f"❌ Sources file not found: {sources_path}")
            exit(1)
            
        with open(sources_path, 'r', encoding='utf-8') as f:
            sources = json.load(f)
        
        logger.info(f"📰 Loaded {len(sources)} sources")
        
        # Run async scraper
        logger.info("🕷️ Starting async scraping...")
        articles = run_async_scrape(sources)
        
        logger.info(f"✅ Scraped {len(articles)} articles")
        
        if not articles:
            logger.warning("⚠️ No articles found, exiting")
            exit(0)
        
        # Process articles through pipeline using batch processing
        logger.info("🔄 Processing articles through batch pipeline...")
        
        # Initialize models
        initialize_models()
        
        # Add article_ids if not present
        for i, article in enumerate(articles):
            if 'article_id' not in article:
                article['article_id'] = f"article_{i}_{article.get('source_name', 'unknown')}"
        
        # Run batch pipeline
        result = run_batch_pipeline(articles)
        
        logger.info(f"🎯 Pipeline completed successfully! {result}")
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        logger.error(traceback.format_exc())
        exit(1) 