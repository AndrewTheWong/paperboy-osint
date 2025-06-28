"""
Similarity search module for finding similar articles using SBERT embeddings and PCA.

This module provides functionality to:
1. Connect to Supabase and query the osint_articles table
2. Load PCA model and SBERT model
3. Embed query text and find similar articles using cosine similarity
"""

import logging
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from utils.supabase_client import get_supabase

# Configure logging
logger = logging.getLogger(__name__)

# Global variables for models (loaded once)
_pca_model = None
_sbert_model = None

def load_models():
    """Load PCA and SBERT models if not already loaded."""
    global _pca_model, _sbert_model
    
    try:
        # Load PCA model
        if _pca_model is None:
            pca_path = Path("models/article_pca_model.pkl")
            if pca_path.exists():
                _pca_model = joblib.load(pca_path)
                logger.info("PCA model loaded successfully")
            else:
                logger.warning("PCA model file not found, creating dummy model")
                # Create a dummy PCA model for testing
                from sklearn.decomposition import PCA
                _pca_model = PCA(n_components=100)
                # Set dummy components
                _pca_model.components_ = np.random.rand(100, 385)
                _pca_model.mean_ = np.zeros(385)
        
        # Load SBERT model
        if _sbert_model is None:
            _sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("SBERT model loaded successfully")
            
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def query_articles_from_supabase():
    """Query articles and embeddings from Supabase."""
    try:
        client = get_supabase()
        
        # Query the osint_articles table
        response = client.table('osint_articles').select(
            'id, title, date, tags, goldstein_score, sbert_embedding'
        ).execute()
        
        if response.data:
            logger.info(f"Retrieved {len(response.data)} articles from Supabase")
            return response.data
        else:
            logger.warning("No articles found in Supabase")
            return []
            
    except Exception as e:
        logger.error(f"Error querying Supabase: {str(e)}")
        # Return empty list for testing purposes
        return []

def embed_and_search(query_text: str, top_k: int = 5) -> pd.DataFrame:
    """
    Embed query text and find top_k most similar articles.
    
    Args:
        query_text (str): The text to search for similar articles
        top_k (int): Number of top similar articles to return
        
    Returns:
        pd.DataFrame: DataFrame containing similar articles with metadata and similarity scores
    """
    try:
        # Handle empty queries
        if not query_text or not query_text.strip():
            logger.warning("Empty query provided")
            return pd.DataFrame(columns=['id', 'title', 'date', 'tags', 'goldstein_score', 'similarity'])
        
        # Load models
        load_models()
        
        # Get articles from Supabase
        articles_data = query_articles_from_supabase()
        
        if not articles_data:
            logger.warning("No articles available for search")
            return pd.DataFrame(columns=['id', 'title', 'date', 'tags', 'goldstein_score', 'similarity'])
        
        # Embed the query text
        query_embedding = _sbert_model.encode([query_text])
        
        # Transform to PCA space
        query_pca = _pca_model.transform(query_embedding)
        
        # Process article embeddings and calculate similarities
        similarities = []
        processed_articles = []
        
        for article in articles_data:
            try:
                # Extract embedding (assume it's stored as a list)
                sbert_embedding = article.get('sbert_embedding', [])
                
                if not sbert_embedding or len(sbert_embedding) == 0:
                    # Skip articles without embeddings
                    continue
                
                # Convert to numpy array and reshape
                article_embedding = np.array(sbert_embedding).reshape(1, -1)
                
                # Transform to PCA space
                article_pca = _pca_model.transform(article_embedding)
                
                # Calculate cosine similarity
                similarity = cosine_similarity(query_pca, article_pca)[0][0]
                
                # Ensure similarity is between 0 and 1
                similarity = max(0, min(1, (similarity + 1) / 2))
                
                similarities.append(similarity)
                processed_articles.append(article)
                
            except Exception as e:
                logger.warning(f"Error processing article {article.get('id', 'unknown')}: {str(e)}")
                continue
        
        if not processed_articles:
            logger.warning("No articles could be processed for similarity calculation")
            return pd.DataFrame(columns=['id', 'title', 'date', 'tags', 'goldstein_score', 'similarity'])
        
        # Create DataFrame with results
        results_df = pd.DataFrame(processed_articles)
        results_df['similarity'] = similarities
        
        # Sort by similarity (descending) and get top_k
        results_df = results_df.sort_values('similarity', ascending=False).head(top_k)
        
        # Round similarity scores for better display
        results_df['similarity'] = results_df['similarity'].round(4)
        
        # Ensure we have the required columns
        required_columns = ['id', 'title', 'date', 'tags', 'goldstein_score', 'similarity']
        for col in required_columns:
            if col not in results_df.columns:
                results_df[col] = None
        
        # Select only the required columns
        results_df = results_df[required_columns]
        
        logger.info(f"Found {len(results_df)} similar articles")
        return results_df
        
    except Exception as e:
        logger.error(f"Error in embed_and_search: {str(e)}")
        # Return empty DataFrame with correct columns on error
        return pd.DataFrame(columns=['id', 'title', 'date', 'tags', 'goldstein_score', 'similarity']) 