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

def preprocess_text_for_embedding(title: str, content: str, tags: List[str] = None, entities: List[str] = None) -> str:
    """
    Preprocess text for better topic-based embeddings
    
    Args:
        title: Article title
        content: Article body content
        tags: List of tags (optional)
        entities: List of entities (optional)
        
    Returns:
        str: Preprocessed text optimized for topic clustering
    """
    import re
    
    # Clean and normalize text
    def clean_text(text: str) -> str:
        if not text:
            return ""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove special characters but keep important punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\-\:]', ' ', text)
        return text
    
    # Combine all text elements
    parts = []
    
    # Add title (weighted more heavily)
    if title:
        clean_title = clean_text(title)
        if clean_title:
            parts.append(clean_title)
    
    # Add content (main body)
    if content:
        clean_content = clean_text(content)
        if clean_content:
            # Truncate very long content to focus on key topics
            if len(clean_content) > 2000:
                clean_content = clean_content[:2000] + "..."
            parts.append(clean_content)
    
    # Add tags (if available) - these are strong topic indicators
    if tags:
        clean_tags = [clean_text(tag) for tag in tags if tag]
        if clean_tags:
            tags_text = " ".join(clean_tags)
            parts.append(f"Topics: {tags_text}")
    
    # Add entities (if available) - these are important topic markers
    if entities:
        clean_entities = [clean_text(entity) for entity in entities if entity]
        if clean_entities:
            entities_text = " ".join(clean_entities)
            parts.append(f"Entities: {entities_text}")
    
    # Combine all parts
    combined_text = " ".join(parts)
    
    return combined_text

def generate_topic_aware_embedding(title: str, content: str, tags: List[str] = None, entities: List[str] = None) -> List[float]:
    """
    Generate embedding optimized for topic clustering
    
    Args:
        title: Article title
        content: Article body content
        tags: List of tags (optional)
        entities: List of entities (optional)
        
    Returns:
        List[float]: Topic-aware embedding vector
    """
    # Preprocess text for better topic representation
    preprocessed_text = preprocess_text_for_embedding(title, content, tags, entities)
    
    # Generate embedding from preprocessed text
    return generate_embedding(preprocessed_text) 

def create_multimodal_embedding(text_embedding: List[float], tags: List[str] = None, entities: List[str] = None, 
                               tag_weight: float = 0.3, entity_weight: float = 0.2) -> List[float]:
    """
    Create multi-modal embedding by combining text embedding with tag/entity information
    
    Args:
        text_embedding: Base text embedding (384-dim)
        tags: List of tags
        entities: List of entities
        tag_weight: Weight for tag influence (0-1)
        entity_weight: Weight for entity influence (0-1)
        
    Returns:
        List[float]: Combined embedding (384-dim)
    """
    import numpy as np
    
    if not text_embedding or len(text_embedding) != 384:
        return text_embedding
    
    # Convert to numpy for easier manipulation
    base_embedding = np.array(text_embedding, dtype=np.float32)
    
    # Create tag embedding (if tags available)
    if tags and len(tags) > 0:
        # Simple approach: create embedding from tag text
        tag_text = " ".join(tags)
        tag_embedding = generate_embedding(tag_text)
        if tag_embedding and len(tag_embedding) == 384:
            tag_vector = np.array(tag_embedding, dtype=np.float32)
            # Weighted combination
            base_embedding = (1 - tag_weight) * base_embedding + tag_weight * tag_vector
    
    # Create entity embedding (if entities available)
    if entities and len(entities) > 0:
        # Simple approach: create embedding from entity text
        entity_text = " ".join(entities)
        entity_embedding = generate_embedding(entity_text)
        if entity_embedding and len(entity_embedding) == 384:
            entity_vector = np.array(entity_embedding, dtype=np.float32)
            # Weighted combination
            base_embedding = (1 - entity_weight) * base_embedding + entity_weight * entity_vector
    
    # Normalize the final embedding
    norm = np.linalg.norm(base_embedding)
    if norm > 0:
        base_embedding = base_embedding / norm
    
    return base_embedding.tolist()

def generate_enhanced_embedding(title: str, content: str, tags: List[str] = None, entities: List[str] = None,
                              use_multimodal: bool = True) -> List[float]:
    """
    Generate enhanced embedding for topic clustering
    
    Args:
        title: Article title
        content: Article content
        tags: List of tags
        entities: List of entities
        use_multimodal: Whether to use multi-modal approach
        
    Returns:
        List[float]: Enhanced embedding
    """
    if use_multimodal and (tags or entities):
        # Use multi-modal approach
        # First get base text embedding
        preprocessed_text = preprocess_text_for_embedding(title, content, tags, entities)
        text_embedding = generate_embedding(preprocessed_text)
        
        # Then combine with tag/entity information
        return create_multimodal_embedding(text_embedding, tags, entities)
    else:
        # Fall back to topic-aware embedding
        return generate_topic_aware_embedding(title, content, tags, entities) 