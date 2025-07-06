#!/usr/bin/env python3
"""
Embedding Worker for Paperboy Backend
Handles text embedding generation for articles
"""

import logging
from celery import shared_task
from typing import List, Dict, Any
from services.embedder import embed_articles_batch, embed_text

logger = logging.getLogger(__name__)

@shared_task(bind=True, max_retries=3)
def embed_single_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate embedding for a single article
    
    Args:
        article: Article dictionary with title and content
        
    Returns:
        dict: Article with embedding added
    """
    try:
        logger.info(f"🔢 Embedding article {article.get('article_id', 'unknown')}")
        
        # Extract text for embedding
        title = article.get('title_translated', article.get('title', ''))
        content = article.get('content_translated', article.get('body', article.get('content', '')))
        
        # Combine title and content for embedding
        text_for_embedding = f"{title}\n\n{content}"
        
        # Generate embedding
        embedding = embed_text(text_for_embedding)
        
        if embedding:
            article['embedding'] = embedding
            logger.info(f"✅ Embedded article {article.get('article_id', 'unknown')}: "
                       f"{len(embedding)} dimensions")
        else:
            logger.warning(f"⚠️ Failed to generate embedding for article {article.get('article_id', 'unknown')}")
        
        return article
        
    except Exception as e:
        logger.error(f"❌ Embedding failed for article {article.get('article_id', 'unknown')}: {e}")
        raise self.retry(countdown=60, max_retries=3)

@shared_task(bind=True, max_retries=3)
def embed_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate embeddings for a batch of articles
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        list: List of articles with embeddings added
    """
    try:
        logger.info(f"🔢 Starting batch embedding of {len(articles)} articles")
        
        # Prepare articles for batch embedding
        articles_for_embedding = []
        for article in articles:
            title = article.get('title_translated', article.get('title', ''))
            content = article.get('content_translated', article.get('body', article.get('content', '')))
            
            articles_for_embedding.append({
                'title': title,
                'content': content,
                'article_id': article.get('article_id', 'unknown')
            })
        
        # Generate embeddings in batch
        embedded_articles = embed_articles_batch(articles_for_embedding)
        
        # Merge embeddings back to original articles
        for i, (article, embedded_article) in enumerate(zip(articles, embedded_articles)):
            if 'embedding' in embedded_article:
                article['embedding'] = embedded_article['embedding']
        
        # Count statistics
        successful_embeddings = sum(1 for article in articles if article.get('embedding'))
        total_dimensions = sum(len(article.get('embedding', [])) for article in articles)
        
        logger.info(f"✅ Batch embedding completed: {successful_embeddings}/{len(articles)} articles embedded")
        logger.info(f"📊 Total dimensions: {total_dimensions}")
        
        return articles
        
    except Exception as e:
        logger.error(f"❌ Batch embedding failed: {e}")
        raise self.retry(countdown=120, max_retries=3)

@shared_task(bind=True, max_retries=3)
def embed_from_queue(self, queue_name: str = "embedding_queue") -> Dict[str, Any]:
    """
    Generate embeddings for articles from a Redis queue
    
    Args:
        queue_name: Name of the Redis queue to process
        
    Returns:
        dict: Embedding results
    """
    try:
        logger.info(f"🔢 Starting embedding from queue: {queue_name}")
        
        # Import Redis queue functions
        from db.redis_queue import get_from_queue, get_queue_size, add_to_queue
        
        # Get queue size
        queue_size = get_queue_size(queue_name)
        logger.info(f"📊 Found {queue_size} articles in {queue_name}")
        
        if queue_size == 0:
            logger.warning(f"⚠️ No articles found in {queue_name}")
            return {"status": "no_data", "embedded_count": 0}
        
        # Process articles from queue
        articles_to_embed = []
        processed_count = 0
        max_articles = min(queue_size, 20)  # Process up to 20 articles at a time
        
        while processed_count < max_articles:
            article_data = get_from_queue(queue_name)
            if not article_data:
                break
            
            articles_to_embed.append(article_data)
            processed_count += 1
        
        if not articles_to_embed:
            logger.warning("⚠️ No articles retrieved from queue")
            return {"status": "no_articles", "embedded_count": 0}
        
        # Generate embeddings
        embedded_articles = embed_articles_batch(articles_to_embed)
        
        # Store embedded articles back to queue for next step
        for article in embedded_articles:
            add_to_queue("clustering_queue", article)
        
        # Count statistics
        successful_embeddings = sum(1 for article in embedded_articles if article.get('embedding'))
        total_dimensions = sum(len(article.get('embedding', [])) for article in embedded_articles)
        
        logger.info(f"✅ Embedding from queue completed: {successful_embeddings}/{len(embedded_articles)} articles embedded")
        logger.info(f"📊 Total dimensions: {total_dimensions}")
        
        return {
            "status": "success",
            "embedded_count": successful_embeddings,
            "total_articles": len(embedded_articles),
            "total_dimensions": total_dimensions,
            "queue_processed": queue_name
        }
        
    except Exception as e:
        logger.error(f"❌ Embedding from queue failed: {e}")
        raise self.retry(countdown=300, max_retries=3)

@shared_task(bind=True, max_retries=3)
def store_embeddings_to_faiss(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Store embeddings to Faiss index
    
    Args:
        articles: List of articles with embeddings
        
    Returns:
        dict: Storage results
    """
    try:
        logger.info(f"💾 Storing {len(articles)} embeddings to Faiss")
        
        from services.faiss_index import get_faiss_service
        
        # Get Faiss service
        faiss_service = get_faiss_service()
        
        # Store embeddings
        stored_count = faiss_service.add_embeddings_batch(articles)
        
        logger.info(f"✅ Stored {stored_count}/{len(articles)} embeddings to Faiss")
        
        return {
            "status": "success",
            "stored_count": stored_count,
            "total_articles": len(articles)
        }
        
    except Exception as e:
        logger.error(f"❌ Faiss storage failed: {e}")
        raise self.retry(countdown=120, max_retries=3)

@shared_task(bind=True, max_retries=3)
def embed_and_store_batch(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate embeddings and store to Faiss in one operation
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        dict: Combined embedding and storage results
    """
    try:
        logger.info(f"🔢 Embedding and storing {len(articles)} articles")
        
        # Generate embeddings
        embedded_articles = embed_articles_batch(articles)
        
        # Store to Faiss
        storage_result = store_embeddings_to_faiss(embedded_articles)
        
        successful_embeddings = sum(1 for article in embedded_articles if article.get('embedding'))
        
        logger.info(f"✅ Embed and store completed: {successful_embeddings} embeddings, "
                   f"{storage_result['stored_count']} stored to Faiss")
        
        return {
            "status": "success",
            "embedded_count": successful_embeddings,
            "stored_count": storage_result['stored_count'],
            "total_articles": len(articles)
        }
        
    except Exception as e:
        logger.error(f"❌ Embed and store failed: {e}")
        raise self.retry(countdown=180, max_retries=3)

@shared_task(bind=True, max_retries=3)
def similarity_search(self, query_text: str, top_k: int = 10) -> Dict[str, Any]:
    """
    Perform similarity search using Faiss
    
    Args:
        query_text: Text to search for
        top_k: Number of top results to return
        
    Returns:
        dict: Search results
    """
    try:
        logger.info(f"🔍 Performing similarity search for: {query_text[:50]}...")
        
        from services.faiss_index import get_faiss_service
        
        # Get Faiss service
        faiss_service = get_faiss_service()
        
        # Generate embedding for query
        from services.embedder import embed_text
        query_embedding = embed_text(query_text)
        
        if not query_embedding:
            logger.error("❌ Failed to generate embedding for query")
            return {"status": "error", "message": "Failed to generate query embedding"}
        
        # Perform search
        results = faiss_service.search(query_embedding, top_k)
        
        logger.info(f"✅ Similarity search completed: {len(results)} results found")
        
        return {
            "status": "success",
            "query": query_text,
            "results": results,
            "top_k": top_k
        }
        
    except Exception as e:
        logger.error(f"❌ Similarity search failed: {e}")
        raise self.retry(countdown=60, max_retries=3) 