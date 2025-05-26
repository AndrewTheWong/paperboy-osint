#!/usr/bin/env python3
"""
Pipeline for automatically tagging articles based on content analysis.
Implements a hybrid tagging system combining keyword-based and ML-based approaches.
"""
import os
import json
import logging
from typing import List, Dict, Any, Set, Tuple
from pathlib import Path
from datetime import datetime

# Import tag utils
from tagging.tag_utils import KEYWORD_MAP, extract_tags_from_text, needs_human_review

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('tagging_pipeline')

# Flag for ML availability
HAS_ML_DEPS = False

# Try to import ML dependencies
try:
    # Try Flair first
    import flair
    from flair.data import Sentence
    from flair.models import TextClassifier
    logger.info("Successfully imported Flair for ML-based tagging")
    HAS_ML_DEPS = True
    USE_FLAIR = True
except ImportError:
    logger.warning("Flair not available. Will try Hugging Face transformers.")
    USE_FLAIR = False
    try:
        # Try Hugging Face as fallback
        import torch
        from transformers import pipeline
        logger.info("Successfully imported Hugging Face transformers for ML-based tagging")
        HAS_ML_DEPS = True
    except ImportError:
        logger.warning("ML dependencies not available. Using keyword-only tagging.")
        HAS_ML_DEPS = False

def get_ml_tagger():
    """
    Initialize and return the ML-based tagger based on available dependencies.
    
    Returns:
        ML tagger object or None if dependencies not available
    """
    if not HAS_ML_DEPS:
        return None
    
    try:
        if USE_FLAIR:
            # Load Flair text classifier
            model_path = os.path.join("models", "flair-news-topic-classifier")
            
            # Check if model exists, otherwise use a default model
            if os.path.exists(model_path):
                classifier = TextClassifier.load(model_path)
            else:
                logger.warning(f"Model not found at {model_path}. Using a default model.")
                classifier = TextClassifier.load('en-sentiment')
            
            return classifier
        else:
            # Use Hugging Face pipeline
            classifier = pipeline(
                "text-classification", 
                model="facebook/bart-large-mnli",
                device=-1  # Use CPU
            )
            return classifier
    except Exception as e:
        logger.error(f"Error initializing ML tagger: {str(e)}")
        return None

def get_ml_tags(text: str, ml_tagger) -> List[str]:
    """
    Get ML-based tags for the given text.
    
    Args:
        text: Text to analyze
        ml_tagger: ML model for tagging
        
    Returns:
        List of tags
    """
    if not ml_tagger:
        return []
    
    try:
        # Prepare tag candidates from ALL_TAGS
        tag_candidates = []
        for category_tags in KEYWORD_MAP.values():
            tag_candidates.extend(category_tags)
        
        # Limit text length to avoid issues
        text = text[:5000]
        
        if USE_FLAIR:
            # Create a Flair Sentence
            sentence = Sentence(text)
            
            # Predict
            ml_tagger.predict(sentence)
            
            # Extract labels
            ml_tags = []
            for label in sentence.labels:
                tag = label.value.lower()
                # Map to our tag system if possible
                if tag in tag_candidates:
                    ml_tags.append(tag)
                elif tag in ["politics", "political"]:
                    ml_tags.extend(["diplomacy", "governance"])
                elif tag in ["military", "defense", "defence"]:
                    ml_tags.extend(["military", "naval"])
                elif tag in ["economics", "economy", "business"]:
                    ml_tags.extend(["trade", "investment"])
                elif tag in ["technology", "tech"]:
                    ml_tags.extend(["technology", "cyber"])
            
            return ml_tags[:5]  # Limit to top 5 tags
            
        else:
            # Using Hugging Face zero-shot classification
            # Convert our tags to candidate labels
            candidate_labels = list(set(tag_candidates))[:10]  # Limit to 10 candidates for performance
            
            # Predict
            result = ml_tagger(text, candidate_labels, multi_label=True)
            
            # Extract labels with score > 0.5
            ml_tags = [label for label, score in zip(result['labels'], result['scores']) if score > 0.5]
            
            return ml_tags
            
    except Exception as e:
        logger.error(f"Error in ML tagging: {str(e)}")
        return []

def tag_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a list of articles and add tags based on content analysis using a hybrid approach.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        List of articles with added tags and review flags
    """
    tagged_articles = []
    review_count = 0
    unknown_count = 0
    keyword_only_count = 0
    hybrid_count = 0
    ml_only_count = 0
    
    # Initialize ML tagger if available
    ml_tagger = get_ml_tagger() if HAS_ML_DEPS else None
    if ml_tagger:
        logger.info("ML tagger initialized and ready")
    else:
        logger.warning("ML tagger not available, using keyword-only tagging")
    
    for article in articles:
        # Skip already tagged articles unless they need review
        if "tags" in article and not article.get("needs_review", False):
            tagged_articles.append(article)
            continue
        
        # Extract text to analyze
        text = ""
        if "translated_text" in article and article["translated_text"]:
            text = article["translated_text"]
        elif "title" in article:
            text = article["title"]
        else:
            # No text to analyze
            article["tags"] = ["unknown"]
            article["ml_tags"] = []
            article["needs_review"] = True
            tagged_articles.append(article)
            unknown_count += 1
            continue
        
        # STEP 1: Extract keyword-based tags
        tag_counts = extract_tags_from_text(text)
        
        if tag_counts:
            # Sort tags by occurrence count (highest first)
            keyword_tags = sorted(tag_counts.keys(), key=lambda k: tag_counts[k], reverse=True)
        else:
            keyword_tags = []
        
        # STEP 2: If needed, use ML-based tagging
        ml_tags = []
        
        # Use ML tagging if:
        # 1. Less than 2 keyword tags OR
        # 2. Only keyword tag is "unknown"
        if (len(keyword_tags) < 2 or keyword_tags == ["unknown"]) and ml_tagger:
            ml_tags = get_ml_tags(text, ml_tagger)
        
        # Combine tags, removing duplicates
        all_tags = list(set(keyword_tags + ml_tags))
        
        # Sort by frequency (keeping keyword tags first)
        if tag_counts:
            all_tags = sorted(all_tags, 
                            key=lambda tag: tag_counts.get(tag, 0) if tag in tag_counts else -1,
                            reverse=True)
        
        # If both methods failed, mark as unknown
        if not all_tags:
            all_tags = ["unknown"]
            article["needs_review"] = True
            unknown_count += 1
        else:
            # Determine if human review is needed
            article["needs_review"] = needs_human_review(all_tags)
            
            # Track tagging method
            if keyword_tags and ml_tags:
                hybrid_count += 1
            elif keyword_tags:
                keyword_only_count += 1
            elif ml_tags:
                ml_only_count += 1
        
        # Add tags to article
        article["tags"] = all_tags
        article["ml_tags"] = ml_tags
        article["keyword_tags"] = keyword_tags
        
        if article.get("needs_review", False):
            review_count += 1
        
        tagged_articles.append(article)
    
    # Log statistics
    logger.info(f"Tagged {len(articles)} articles")
    logger.info(f"Tagging method breakdown: {keyword_only_count} keyword-only, {ml_only_count} ML-only, {hybrid_count} hybrid")
    logger.info(f"{review_count} articles need human review")
    logger.info(f"{unknown_count} articles couldn't be automatically tagged")
    
    return tagged_articles

def save_tagged_articles(articles: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save tagged articles to a JSON file.
    
    Args:
        articles: List of tagged article dictionaries
        output_path: Path to save the tagged articles
    """
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved tagged articles to {output_path}")
    except Exception as e:
        logger.error(f"Error saving tagged articles: {str(e)}")
        raise

def load_articles(input_path: str) -> List[Dict[str, Any]]:
    """
    Load articles from a JSON file.
    
    Args:
        input_path: Path to the articles JSON file
        
    Returns:
        List of article dictionaries
    """
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {input_path}")
        return articles
    except Exception as e:
        logger.error(f"Error loading articles: {str(e)}")
        raise

if __name__ == "__main__":
    # Define file paths
    input_file = "data/translated_articles.json"
    output_file = "data/tagged_articles.json"
    
    # Load articles
    articles = load_articles(input_file)
    
    # Tag articles
    tagged_articles = tag_articles(articles)
    
    # Save tagged articles
    save_tagged_articles(tagged_articles, output_file)
    
    # Log summary
    print(f"Tagged {len(tagged_articles)} articles")
    keyword_only = sum(1 for a in tagged_articles if a.get('keyword_tags') and not a.get('ml_tags'))
    ml_only = sum(1 for a in tagged_articles if a.get('ml_tags') and not a.get('keyword_tags'))
    hybrid = sum(1 for a in tagged_articles if a.get('keyword_tags') and a.get('ml_tags'))
    print(f"Tagging method breakdown: {keyword_only} keyword-only, {ml_only} ML-only, {hybrid} hybrid")
    print(f"{sum(1 for a in tagged_articles if a.get('needs_review', False))} articles need human review")
    print(f"{sum(1 for a in tagged_articles if 'unknown' in a.get('tags', []))} articles couldn't be automatically tagged") 