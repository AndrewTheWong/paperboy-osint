#!/usr/bin/env python3
"""
Pipeline for generating sentence embeddings for articles.
Uses intfloat/multilingual-e5-base model for high-quality multilingual embeddings.
"""
import json
import logging
import os
import time
from typing import List, Dict, Any, Union, Tuple
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('embedding_pipeline')

# Try importing SentenceTransformer, but continue with a stub if it's not available
try:
    from sentence_transformers import SentenceTransformer
    HAS_SBERT = True
except ImportError:
    logger.error("Missing dependency: sentence-transformers. Install with: pip install sentence-transformers")
    logger.info("Without sentence-transformers, embeddings will not be generated.")
    HAS_SBERT = False

def load_embedding_model(model_name: str = "intfloat/multilingual-e5-base") -> Any:
    """
    Load the sentence transformer model with error handling.
    
    Args:
        model_name: Name of the model to load
        
    Returns:
        Loaded model or None if failed
    """
    if not HAS_SBERT:
        return None
    
    try:
        logger.info(f"Loading sentence transformer model: {model_name}")
        start_time = time.time()
        model = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Log model info
        embedding_dim = model.get_sentence_embedding_dimension()
        logger.info(f"Model embedding dimension: {embedding_dim}")
        
        return model
    except Exception as e:
        logger.error(f"Error loading model {model_name}: {str(e)}")
        logger.warning("Will continue without embeddings")
        return None

def embed_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Generate sentence embeddings for a list of articles.
    
    Args:
        articles: List of article dictionaries, each containing at least a 'translated_text' field
        
    Returns:
        List of articles with added 'embedding' field containing the vector representation
    """
    # Load the sentence transformer model
    model = load_embedding_model("intfloat/multilingual-e5-base")
    
    if not model:
        logger.warning("No embedding model available. Returning articles without embeddings.")
        return articles
    
    # Track statistics
    start_time = time.time()
    texts_to_embed = []
    skipped_indices = []
    embed_indices = []
    
    # Extract texts to embed and track which articles have text
    for i, article in enumerate(articles):
        if 'translated_text' in article and article['translated_text']:
            texts_to_embed.append(article['translated_text'])
            embed_indices.append(i)
        elif 'title' in article and article['title']:
            # Fallback to title if translated_text is not available
            texts_to_embed.append(article['title'])
            embed_indices.append(i)
        else:
            # No text to embed
            skipped_indices.append(i)
    
    logger.info(f"Preparing to embed {len(texts_to_embed)} articles (skipping {len(skipped_indices)} with no text)")
    
    # Generate embeddings in batches for memory efficiency
    batch_size = 32
    all_embeddings = []
    
    if texts_to_embed:
        try:
            # Process in batches
            for i in range(0, len(texts_to_embed), batch_size):
                batch_texts = texts_to_embed[i:i+batch_size]
                logger.info(f"Embedding batch {i//batch_size + 1}/{(len(texts_to_embed)-1)//batch_size + 1} ({len(batch_texts)} articles)")
                
                # Generate embeddings
                batch_embeddings = model.encode(
                    batch_texts,
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                
                all_embeddings.extend(batch_embeddings)
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            logger.warning("Continuing without embeddings")
            # Return original articles if embedding fails
            return articles
    
    # Add embeddings to articles
    embedded_articles = []
    embedding_idx = 0
    
    for i, article in enumerate(articles):
        article_copy = article.copy()
        
        if i in embed_indices:
            # Convert numpy array to list for JSON serialization
            article_copy['embedding'] = all_embeddings[embedding_idx].tolist()
            article_copy['embedding_model'] = "intfloat/multilingual-e5-base"
            embedding_idx += 1
        else:
            # Add empty embedding for skipped articles
            article_copy['embedding'] = []
            article_copy['embedding_model'] = None
        
        embedded_articles.append(article_copy)
    
    # Calculate stats
    total_time = time.time() - start_time
    embedding_dim = len(all_embeddings[0].tolist()) if all_embeddings else 0
    
    logger.info(f"✅ Embedded {len(texts_to_embed)} articles | ❌ Skipped {len(skipped_indices)} (no text)")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Total embedding time: {total_time:.2f} seconds")
    
    return embedded_articles

def save_embedded_articles(articles: List[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    """
    Save articles with embeddings to a JSON file.
    
    Args:
        articles: List of article dictionaries with embeddings
        output_path: Path to save the embedded articles
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    # Calculate file size in MB
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    logger.info(f"Saved {len(articles)} embedded articles to {output_path} ({file_size_mb:.2f} MB)")

def load_articles(input_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        input_path: Path to the articles JSON file
        
    Returns:
        List of article dictionaries
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return []
    
    try:
        with open(input_path, "r", encoding="utf-8") as f:
            articles = json.load(f)
        
        logger.info(f"Loaded {len(articles)} articles from {input_path}")
        return articles
    except Exception as e:
        logger.error(f"Error loading articles: {str(e)}")
        return []

if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate embeddings for articles")
    parser.add_argument("--input", default="data/tagged_articles.json", help="Input file path")
    parser.add_argument("--output", default="data/embedded_articles.json", help="Output file path")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base", help="Embedding model name")
    
    args = parser.parse_args()
    
    # Load articles
    articles = load_articles(args.input)
    
    if not articles:
        logger.error("No articles loaded. Exiting.")
        exit(1)
    
    # Generate embeddings
    embedded_articles = embed_articles(articles)
    
    # Save embedded articles
    save_embedded_articles(embedded_articles, args.output)
    
    # Print example
    if embedded_articles:
        sample = embedded_articles[0]
        emb_len = len(sample.get('embedding', []))
        if emb_len > 0:
            logger.info(f"🧬 Sample embedding dimension: {emb_len}")
            logger.info(f"Model used: {sample.get('embedding_model', 'unknown')}")
        else:
            logger.warning("Sample article has no embedding") 