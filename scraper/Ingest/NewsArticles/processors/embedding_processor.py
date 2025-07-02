#!/usr/bin/env python3
"""
SBERT Embedding Processor
Generates and manages sentence embeddings for articles using SBERT
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional
import re

logger = logging.getLogger(__name__)

try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    logger.warning("sentence-transformers not available. Embeddings will be disabled.")
    HAS_SBERT = False

class EmbeddingProcessor:
    """High-performance SBERT embedding generation"""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """Initialize the embedding processor"""
        self.model_name = model_name
        self.model = None
        self.embedding_dim = 384  # Default for all-MiniLM-L6-v2
        
        if HAS_SBERT:
            self._load_model()
        else:
            logger.error("SBERT not available - embeddings will be None")
    
    def _load_model(self):
        """Load the SBERT model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            logger.info(f"✅ Loaded SBERT model: {self.model_name} ({self.embedding_dim} dimensions)")
        except Exception as e:
            logger.error(f"❌ Failed to load SBERT model: {e}")
            self.model = None
    
    def _clean_text(self, text: str, max_length: int = 512) -> str:
        """Clean and prepare text for embedding"""
        if not text:
            return ""
        
        # Remove excessive whitespace and newlines
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Limit length for performance
        if len(text) > max_length:
            text = text[:max_length]
        
        return text.strip()
    
    def generate_embedding(self, text: str) -> Optional[List[float]]:
        """Generate embedding for a single text"""
        if not self.model or not text:
            return None
        
        try:
            # Clean the text
            clean_text = self._clean_text(text)
            
            if not clean_text:
                return None
            
            # Generate embedding
            embedding = self.model.encode(clean_text, convert_to_tensor=False)
            
            # Convert to list if numpy array
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            # Validate embedding
            if len(embedding) != self.embedding_dim:
                logger.warning(f"Unexpected embedding dimension: {len(embedding)}")
                return None
            
            return embedding
        
        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return None
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Generate embeddings for multiple texts efficiently"""
        if not self.model or not texts:
            return [None] * len(texts)
        
        try:
            # Clean all texts
            clean_texts = [self._clean_text(text) for text in texts]
            
            # Filter out empty texts but keep track of indices
            valid_texts = []
            valid_indices = []
            
            for i, text in enumerate(clean_texts):
                if text:
                    valid_texts.append(text)
                    valid_indices.append(i)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Generate embeddings for valid texts
            embeddings = self.model.encode(valid_texts, convert_to_tensor=False, show_progress_bar=False)
            
            # Map back to original indices
            result = [None] * len(texts)
            
            for i, embedding in enumerate(embeddings):
                original_index = valid_indices[i]
                
                # Convert to list if numpy array
                if hasattr(embedding, 'tolist'):
                    embedding = embedding.tolist()
                
                result[original_index] = embedding
            
            return result
        
        except Exception as e:
            logger.error(f"Failed to generate batch embeddings: {e}")
            return [None] * len(texts)
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single article to add embeddings"""
        result = article.copy()
        
        # Extract title and content
        title = article.get('title', '')
        content = article.get('content', '')
        
        # Generate embeddings
        title_embedding = self.generate_embedding(title)
        content_embedding = self.generate_embedding(content)
        
        # Add embeddings to result
        result['title_embedding'] = title_embedding
        result['content_embedding'] = content_embedding
        result['embedding_model'] = self.model_name if title_embedding or content_embedding else None
        
        # Add metadata
        result['has_title_embedding'] = title_embedding is not None
        result['has_content_embedding'] = content_embedding is not None
        
        return result
    
    def process_articles_batch(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process multiple articles efficiently"""
        if not articles:
            return []
        
        logger.info(f"Processing embeddings for {len(articles)} articles...")
        
        # Extract texts
        titles = [article.get('title', '') for article in articles]
        contents = [article.get('content', '') for article in articles]
        
        # Generate embeddings in batches
        title_embeddings = self.generate_embeddings_batch(titles)
        content_embeddings = self.generate_embeddings_batch(contents)
        
        # Add embeddings to articles
        results = []
        successful_embeddings = 0
        
        for i, article in enumerate(articles):
            result = article.copy()
            
            title_emb = title_embeddings[i]
            content_emb = content_embeddings[i]
            
            result['title_embedding'] = title_emb
            result['content_embedding'] = content_emb
            result['embedding_model'] = self.model_name if title_emb or content_emb else None
            result['has_title_embedding'] = title_emb is not None
            result['has_content_embedding'] = content_emb is not None
            
            if title_emb or content_emb:
                successful_embeddings += 1
            
            results.append(result)
        
        logger.info(f"✅ Generated embeddings for {successful_embeddings}/{len(articles)} articles")
        
        return results
    
    def validate_embedding(self, embedding: Any) -> bool:
        """Validate that an embedding is properly formatted"""
        if embedding is None:
            return False
        
        if not isinstance(embedding, (list, np.ndarray)):
            return False
        
        if len(embedding) != self.embedding_dim:
            return False
        
        # Check for NaN or infinite values
        try:
            arr = np.array(embedding)
            if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
                return False
        except:
            return False
        
        return True
    
    def get_embedding_stats(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about embeddings in articles"""
        total_articles = len(articles)
        title_embeddings = sum(1 for a in articles if a.get('has_title_embedding', False))
        content_embeddings = sum(1 for a in articles if a.get('has_content_embedding', False))
        
        return {
            'total_articles': total_articles,
            'title_embeddings': title_embeddings,
            'content_embeddings': content_embeddings,
            'title_coverage': title_embeddings / total_articles if total_articles > 0 else 0,
            'content_coverage': content_embeddings / total_articles if total_articles > 0 else 0,
            'model_name': self.model_name,
            'embedding_dimension': self.embedding_dim
        } 