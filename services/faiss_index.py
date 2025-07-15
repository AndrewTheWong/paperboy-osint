#!/usr/bin/env python3
"""
Faiss Index Service for Vector Storage
"""

import logging
import numpy as np
import os
from typing import List, Dict, Optional, Tuple, Any
import faiss
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FaissIndexService:
    def __init__(self, dimension: int = 384, index_path: str = "faiss_index"):
        """
        Initialize Faiss index service
        
        Args:
            dimension: Vector dimension (default 384 for MiniLM)
            index_path: Path to persist index
        """
        self.dimension = dimension
        self.index_path = index_path
        self.metadata_path = f"{index_path}_metadata.pkl"
        self.insert_count = 0
        self.persistence_threshold = 100
        
        # Initialize or load index
        self.index, self.article_ids = self._load_or_create_index()
        logger.info(f"✅ Faiss index initialized with {len(self.article_ids)} vectors")
    
    def _load_or_create_index(self) -> Tuple[faiss.Index, List[str]]:
        """Load existing index or create new one"""
        try:
            if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
                # Load existing index
                index = faiss.read_index(self.index_path)
                with open(self.metadata_path, 'rb') as f:
                    article_ids = pickle.load(f)
                logger.info(f"📂 Loaded existing Faiss index with {len(article_ids)} vectors")
                return index, article_ids
            else:
                # Create new index
                index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                article_ids = []
                logger.info(f"🆕 Created new Faiss index with dimension {self.dimension}")
                return index, article_ids
        except Exception as e:
            logger.error(f"❌ Error loading/creating Faiss index: {e}")
            # Fallback to new index
            index = faiss.IndexFlatIP(self.dimension)
            article_ids = []
            return index, article_ids
    
    def _persist_index(self):
        """Persist index and metadata to disk"""
        try:
            faiss.write_index(self.index, self.index_path)
            with open(self.metadata_path, 'wb') as f:
                pickle.dump(self.article_ids, f)
            logger.info(f"💾 Persisted Faiss index with {len(self.article_ids)} vectors")
        except Exception as e:
            logger.error(f"❌ Error persisting Faiss index: {e}")
    
    def add_embedding(self, article_id: str, embedding: List[float]) -> bool:
        """
        Add embedding to Faiss index
        
        Args:
            article_id: Unique article identifier
            embedding: Vector embedding
            
        Returns:
            bool: Success status
        """
        try:
            # Validate embedding
            if not embedding or len(embedding) != self.dimension:
                logger.warning(f"⚠️ Skipping article {article_id}: invalid embedding (expected {self.dimension}, got {len(embedding) if embedding else 0})")
                return False
            
            # Convert to numpy array and normalize
            vector = np.array(embedding, dtype=np.float32)
            faiss.normalize_L2(vector.reshape(1, -1))  # Normalize for cosine similarity
            
            # Add to index
            self.index.add(vector.reshape(1, -1))
            self.article_ids.append(article_id)
            self.insert_count += 1
            
            logger.info(f"✅ Added embedding for article {article_id} to Faiss index")
            
            # Persist if threshold reached
            if self.insert_count % self.persistence_threshold == 0:
                self._persist_index()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error adding embedding for article {article_id}: {e}")
            return False
    
    def add_embeddings_batch(self, articles: List[Dict[str, Any]]) -> int:
        """
        Add multiple embeddings to Faiss index
        
        Args:
            articles: List of articles with embeddings
            
        Returns:
            int: Number of successfully added embeddings
        """
        logger.info(f"[FAISS] Adding {len(articles)} embeddings to index")
        
        added_count = 0
        for article in articles:
            article_id = article.get('article_id', article.get('id', 'unknown'))
            embedding = article.get('embedding')
            
            if self.add_embedding(article_id, embedding):
                added_count += 1
        
        logger.info(f"[FAISS] Successfully added {added_count}/{len(articles)} embeddings")
        return added_count
    
    def search_similar(self, query_embedding: List[float], k: int = 5) -> List[Tuple[str, float]]:
        """
        Search for similar articles
        
        Args:
            query_embedding: Query vector
            k: Number of results to return
            
        Returns:
            List of (article_id, similarity_score) tuples
        """
        try:
            # Validate query embedding
            if not query_embedding or len(query_embedding) != self.dimension:
                logger.warning(f"⚠️ Invalid query embedding (expected {self.dimension}, got {len(query_embedding) if query_embedding else 0})")
                return []
            
            # Convert to numpy array and normalize
            query_vector = np.array(query_embedding, dtype=np.float32)
            faiss.normalize_L2(query_vector.reshape(1, -1))
            
            # Search
            similarities, indices = self.index.search(query_vector.reshape(1, -1), min(k, len(self.article_ids)))
            
            # Return results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.article_ids):
                    article_id = self.article_ids[idx]
                    results.append((article_id, float(similarity)))
            
            logger.info(f"🔍 Found {len(results)} similar articles")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error searching Faiss index: {e}")
            return []
    
    def get_index_stats(self) -> Dict[str, Any]:
        """Get index statistics"""
        return {
            "total_vectors": len(self.article_ids),
            "dimension": self.dimension,
            "insert_count": self.insert_count,
            "is_trained": self.index.is_trained,
            "ntotal": self.index.ntotal
        }
    
    def clear_index(self):
        """Clear the index"""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.article_ids = []
        self.insert_count = 0
        logger.info("🗑️ Cleared Faiss index")

# Global instance
faiss_service = None

def get_faiss_service() -> FaissIndexService:
    """Get or create global Faiss service instance"""
    global faiss_service
    if faiss_service is None:
        faiss_service = FaissIndexService()
    return faiss_service 