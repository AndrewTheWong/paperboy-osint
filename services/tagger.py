#!/usr/bin/env python3
"""
OSINT Article Tagger Service
NER-only tagging using spaCy
"""

import logging
from typing import Dict, List, Any, Set
import spacy

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
    Tag article with comprehensive NER analysis to fill all Supabase schema fields
    """
    logger.info(f"[TAGGER] Starting NER analysis")
    logger.info(f"[TAGGER] Input text length: {len(text)} chars")
    logger.info(f"[TAGGER] Input title length: {len(title)} chars")
    
    # Combine title and text for analysis
    full_text = f"{title} {text}".strip()
    logger.info(f"[TAGGER] Combined text length: {len(full_text)} chars")
    
    # Initialize comprehensive tag categories
    tag_categories = {
        "geo": [],           # Geographic locations (GPE, LOC)
        "actor": [],         # Organizations, companies, institutions (ORG)
        "command": [],       # People, leaders, officials (PERSON)
        "event": [],         # Events, incidents, activities (EVENT)
        "facility": [],      # Facilities, buildings, installations (FAC)
        "technology": [],    # Technology, products, systems (PRODUCT)
        "time": [],          # Dates, times, periods (DATE, TIME)
        "quantity": [],      # Numbers, amounts, measurements (QUANTITY, CARDINAL)
        "money": [],         # Financial amounts, currencies (MONEY)
        "law": [],           # Laws, regulations, policies (LAW)
        "language": [],      # Languages, dialects (LANGUAGE)
        "nationality": [],   # Nationalities, ethnic groups (NORP)
        "ordinal": []        # Ordinal numbers, rankings (ORDINAL)
    }
    
    flat_tags: Set[str] = set()
    all_entities: Set[str] = set()
    
    # Comprehensive NER analysis
    if nlp:
        logger.info("[TAGGER] Performing comprehensive NER analysis...")
        logger.info(f"[TAGGER] spaCy model loaded: {nlp is not None}")
        doc = nlp(full_text)
        logger.info(f"[TAGGER] Processed document, found {len(doc.ents)} entities")
        
        entity_count = 0
        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()
            entity_count += 1
            
            logger.info(f"[TAGGER] Entity {entity_count}: '{value}' (type: {label})")
            
            # Map spaCy entity types to our categories
            if label == "GPE" or label == "LOC":
                tag_categories["geo"].append(value)
                flat_tags.add(f"GEO:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added geographic entity: {value}")
            elif label == "ORG":
                tag_categories["actor"].append(value)
                flat_tags.add(f"ACT:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added organization: {value}")
            elif label == "PERSON":
                tag_categories["command"].append(value)
                flat_tags.add(f"CMD:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added person: {value}")
            elif label == "EVENT":
                tag_categories["event"].append(value)
                flat_tags.add(f"EVT:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added event: {value}")
            elif label == "FAC":
                tag_categories["facility"].append(value)
                flat_tags.add(f"FAC:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added facility: {value}")
            elif label == "PRODUCT":
                tag_categories["technology"].append(value)
                flat_tags.add(f"TECH:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added technology: {value}")
            elif label == "DATE":
                tag_categories["time"].append(value)
                flat_tags.add(f"TIME:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added date: {value}")
            elif label == "TIME":
                tag_categories["time"].append(value)
                flat_tags.add(f"TIME:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added time: {value}")
            elif label == "QUANTITY":
                tag_categories["quantity"].append(value)
                flat_tags.add(f"QTY:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added quantity: {value}")
            elif label == "CARDINAL":
                tag_categories["quantity"].append(value)
                flat_tags.add(f"QTY:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added cardinal: {value}")
            elif label == "MONEY":
                tag_categories["money"].append(value)
                flat_tags.add(f"MONEY:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added money: {value}")
            elif label == "LAW":
                tag_categories["law"].append(value)
                flat_tags.add(f"LAW:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added law: {value}")
            elif label == "LANGUAGE":
                tag_categories["language"].append(value)
                flat_tags.add(f"LANG:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added language: {value}")
            elif label == "NORP":
                tag_categories["nationality"].append(value)
                flat_tags.add(f"NAT:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added nationality: {value}")
            elif label == "ORDINAL":
                tag_categories["ordinal"].append(value)
                flat_tags.add(f"ORD:{value}")
                all_entities.add(value)
                logger.info(f"[TAGGER] Added ordinal: {value}")
            else:
                # Catch any other entity types
                all_entities.add(value)
                logger.info(f"[TAGGER] Added unknown entity type '{label}': {value}")
    
    # Deduplicate and sort
    for k in tag_categories:
        tag_categories[k] = sorted(list(set(tag_categories[k])))
    flat_tags = sorted(list(flat_tags))
    all_entities = sorted(list(all_entities))
    
    logger.info(f"SUCCESS: [TAGGER] Comprehensive NER tagging complete:")
    logger.info(f"   Tags: {len(flat_tags)}")
    logger.info(f"   Entities: {entity_count}")
    logger.info(f"   Tag categories: {tag_categories}")
    logger.info(f"   All entities: {all_entities}")
    
    return {
        "tag_categories": tag_categories,
        "tags": flat_tags,
        "entities": all_entities,
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
                'entities': tagging_result['entities'],
            })
            
            tagged_articles.append(article)
            logger.info(f"✅ Tagged article {i+1}/{len(articles)}: {len(tagging_result['tags'])} tags, {len(tagging_result['entities'])} entities")
            
        except Exception as e:
            logger.error(f"❌ Error tagging article {i+1}: {e}")
            # Add empty tagging data to maintain structure
            article.update({
                'tag_categories': {},
                'tags': [],
                'entities': [],
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
    high_priority = []
    for tag in tags:
        if ':' in tag:
            value = tag.split(":")[-1]
            if value in ESCALATION_WEIGHTS and ESCALATION_WEIGHTS[value] >= threshold:
                high_priority.append(tag)
    
    return high_priority 