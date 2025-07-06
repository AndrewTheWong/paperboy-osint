#!/usr/bin/env python3
"""
Embedding service for article analysis
"""

import numpy as np
from typing import List, Optional
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global model instance
_embedding_model = None
_tfidf_model = None
_hdbscan_model = None

def get_embedding_model():
    """Get or create SentenceTransformer model"""
    global _embedding_model
    if _embedding_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Try a simpler model first
            _embedding_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
            logger.info("✅ Loaded SentenceTransformer model (paraphrase-MiniLM-L3-v2)")
        except Exception as e:
            logger.warning(f"⚠️ SentenceTransformer failed: {e}")
            logger.info("🔄 Falling back to TF-IDF embeddings")
            _embedding_model = None
    return _embedding_model

def get_tfidf_model():
    """Get or create TF-IDF model as fallback"""
    global _tfidf_model
    if _tfidf_model is None:
        _tfidf_model = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        logger.info("✅ Loaded TF-IDF model")
    return _tfidf_model

def get_hdbscan_model():
    """Get or create HDBSCAN model"""
    global _hdbscan_model
    if _hdbscan_model is None:
        try:
            import hdbscan
            _hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=2,  # Minimum allowed by HDBSCAN
                min_samples=1,
                metric='euclidean',
                cluster_selection_method='eom'
            )
            logger.info("✅ Loaded HDBSCAN model")
        except Exception as e:
            logger.error(f"❌ Error loading HDBSCAN model: {e}")
            raise
    return _hdbscan_model

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for text using SentenceTransformer or TF-IDF fallback
    
    Args:
        text: Input text to embed
        
    Returns:
        List[float]: Embedding vector
    """
    try:
        model = get_embedding_model()
        
        if model is not None:
            # Use SentenceTransformer
            embedding = model.encode(text)
            return embedding.tolist()
        else:
            # Use TF-IDF fallback
            tfidf_model = get_tfidf_model()
            embedding = tfidf_model.fit_transform([text]).toarray()[0]
            return embedding.tolist()
        
    except Exception as e:
        logger.error(f"❌ Error generating embedding: {e}")
        raise

def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for a batch of texts
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List[List[float]]: List of embedding vectors
    """
    try:
        model = get_embedding_model()
        
        if model is not None:
            # Use SentenceTransformer
            embeddings = model.encode(texts)
            return embeddings.tolist()
        else:
            # Use TF-IDF fallback
            tfidf_model = get_tfidf_model()
            embeddings = tfidf_model.fit_transform(texts).toarray()
            return embeddings.tolist()
        
    except Exception as e:
        logger.error(f"❌ Error generating batch embeddings: {e}")
        raise

def apply_hdbscan_clustering(embeddings: List[List[float]]) -> List[int]:
    """
    Apply HDBSCAN clustering to embeddings
    
    Args:
        embeddings: List of embedding vectors
        
    Returns:
        List[int]: Cluster labels (-1 for noise points)
    """
    try:
        model = get_hdbscan_model()
        
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Apply clustering
        cluster_labels = model.fit_predict(embeddings_array)
        
        logger.info(f"🔍 Applied HDBSCAN clustering: {len(set(cluster_labels))} clusters found")
        
        return cluster_labels.tolist()
        
    except Exception as e:
        logger.error(f"❌ Error applying HDBSCAN clustering: {e}")
        raise

def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """
    Compute cosine similarity between two embeddings
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        float: Cosine similarity score
    """
    try:
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
        
        return float(similarity)
        
    except Exception as e:
        logger.error(f"❌ Error computing similarity: {e}")
        return 0.0 