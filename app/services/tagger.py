#!/usr/bin/env python3
"""
OSINT Article Tagger Service
Combines gazetteer lookup, pattern matching, and NER for comprehensive article tagging
"""

import re
import logging
from typing import Dict, List, Any, Set
import spacy

from app.services.tag_data import TAGS
from app.services.patterns import PATTERNS
from app.services.escalation_score import compute_confidence_score, get_priority_level

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load spaCy model for NER
try:
    nlp = spacy.load("en_core_web_sm")
    logger.info("✅ Loaded spaCy model for NER")
except OSError:
    logger.warning("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

def tag_article(text: str, title: str = "") -> Dict[str, Any]:
    """
    Tag article with structured categories, flat tags, and confidence score
    
    Args:
        text: Article body text
        title: Article title (optional, for additional context)
        
    Returns:
        Dict containing:
        - tag_categories: Dict of categories with lists of values
        - tags: Flat list of tags in "CATEGORY:Value" format
        - confidence_score: Float between 0.0 and 1.0
        - priority_level: "HIGH", "MEDIUM", or "LOW"
    """
    # Combine title and text for analysis
    full_text = f"{title} {text}".strip()
    text_lower = full_text.lower()
    
    # Initialize tag categories
    tag_categories = {k: [] for k in TAGS.keys()}
    flat_tags: Set[str] = set()
    
    # 1. Gazetteer matching (exact string matching)
    logger.info("🔍 Performing gazetteer matching...")
    for category, values in TAGS.items():
        for val in values:
            # Case-insensitive matching
            if val.lower() in text_lower:
                tag_categories[category].append(val)
                flat_tags.add(f"{category.upper()[:4]}:{val}")
                logger.debug(f"Found gazetteer match: {category}:{val}")
    
    # 2. Pattern matching (regex patterns)
    logger.info("🔍 Performing pattern matching...")
    for pattern, category, val in PATTERNS:
        if re.search(pattern, text_lower):
            tag_categories[category].append(val)
            flat_tags.add(f"{category.upper()[:4]}:{val}")
            logger.debug(f"Found pattern match: {category}:{val}")
    
    # 3. NER fallback (if spaCy is available)
    if nlp:
        logger.info("🔍 Performing NER analysis...")
        doc = nlp(full_text)
        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()
            
            # Map NER labels to our categories
            if label == "GPE" and value not in [tag.split(":")[-1] for tag in flat_tags if tag.startswith("GEO:")]:
                tag_categories["geo"].append(value)
                flat_tags.add(f"GEO:{value}")
                logger.debug(f"Found NER GPE: {value}")
            elif label == "ORG" and value not in [tag.split(":")[-1] for tag in flat_tags if tag.startswith("ACT:")]:
                tag_categories["actor"].append(value)
                flat_tags.add(f"ACT:{value}")
                logger.debug(f"Found NER ORG: {value}")
            elif label == "PERSON" and value not in [tag.split(":")[-1] for tag in flat_tags if tag.startswith("CMD:")]:
                tag_categories["command"].append(value)
                flat_tags.add(f"CMD:{value}")
                logger.debug(f"Found NER PERSON: {value}")
    
    # 4. Deduplicate and sort
    for k in tag_categories:
        tag_categories[k] = sorted(list(set(tag_categories[k])))
    flat_tags = sorted(list(flat_tags))
    
    # 5. Compute confidence score
    confidence_score = compute_confidence_score(flat_tags)
    priority_level = get_priority_level(confidence_score)
    
    logger.info(f"✅ Tagging complete: {len(flat_tags)} tags, confidence: {confidence_score:.3f}, priority: {priority_level}")
    
    return {
        "tag_categories": tag_categories,
        "tags": flat_tags,
        "confidence_score": confidence_score,
        "priority_level": priority_level
    }

def tag_article_batch(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Tag a batch of articles
    
    Args:
        articles: List of article dictionaries with 'title' and 'body' fields
        
    Returns:
        List of articles with added tagging information
    """
    logger.info(f"🏷️ Tagging batch of {len(articles)} articles...")
    
    tagged_articles = []
    for i, article in enumerate(articles):
        try:
            title = article.get('title', '')
            body = article.get('body', '')
            
            if not body:
                logger.warning(f"⚠️ Article {i+1} has no body text, skipping tagging")
                tagged_articles.append(article)
                continue
            
            # Perform tagging
            tagging_result = tag_article(body, title)
            
            # Add tagging results to article
            article.update({
                'tag_categories': tagging_result['tag_categories'],
                'tags': tagging_result['tags'],
                'confidence_score': tagging_result['confidence_score'],
                'priority_level': tagging_result['priority_level']
            })
            
            tagged_articles.append(article)
            logger.info(f"✅ Tagged article {i+1}/{len(articles)}: {len(tagging_result['tags'])} tags, score: {tagging_result['confidence_score']:.3f}")
            
        except Exception as e:
            logger.error(f"❌ Error tagging article {i+1}: {e}")
            # Add empty tagging data to maintain structure
            article.update({
                'tag_categories': {},
                'tags': [],
                'confidence_score': 0.0,
                'priority_level': 'LOW'
            })
            tagged_articles.append(article)
    
    logger.info(f"✅ Batch tagging complete: {len(tagged_articles)} articles processed")
    return tagged_articles

def get_tag_summary(tag_categories: Dict[str, List[str]]) -> str:
    """
    Generate a human-readable summary of tags
    
    Args:
        tag_categories: Dictionary of tag categories
        
    Returns:
        str: Formatted tag summary
    """
    summary_parts = []
    
    for category, values in tag_categories.items():
        if values:
            category_name = category.replace('_', ' ').title()
            summary_parts.append(f"{category_name}: {', '.join(values[:3])}{'...' if len(values) > 3 else ''}")
    
    return "; ".join(summary_parts) if summary_parts else "No tags identified"

def get_high_priority_tags(tags: List[str], threshold: float = 0.7) -> List[str]:
    """
    Get high-priority tags based on escalation weights
    
    Args:
        tags: List of tags in "CATEGORY:Value" format
        threshold: Minimum weight for high priority
        
    Returns:
        List of high-priority tags
    """
    from app.services.escalation_score import ESCALATION_WEIGHTS
    
    high_priority = []
    for tag in tags:
        if ':' in tag:
            value = tag.split(":")[-1]
            if ESCALATION_WEIGHTS.get(value, 0.0) >= threshold:
                high_priority.append(tag)
    
    return high_priority 