#!/usr/bin/env python3
"""
OSINT Article Tagger Service
Enhanced NER tagging with keyword extraction, sentiment analysis, and topic detection
"""

import logging
import re
from typing import Dict, List, Any, Set, Tuple
import spacy
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import string

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Load spaCy model once globally with optimizations
try:
    # Try to use GPU if available
    nlp = spacy.load("en_core_web_sm")
    
    # Disable unnecessary pipeline components for speed
    nlp.disable_pipes(["tagger", "parser"])
    
    # Use GPU if available
    try:
        if spacy.util.is_gpu_available():
            nlp.to_disk("temp_model")
            nlp = spacy.load("temp_model")
            logger.info("✅ Loaded spaCy model with GPU acceleration")
        else:
            logger.info("✅ Loaded spaCy model (CPU mode)")
    except AttributeError:
        # Fallback for older spaCy versions
        logger.info("✅ Loaded spaCy model (CPU mode)")
        
except OSError:
    logger.warning("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
    nlp = None

def chunk_text(text: str, max_length: int = 100000) -> List[str]:
    """Split long text into smaller chunks for faster processing"""
    if len(text) <= max_length:
        return [text]
    
    # Split by sentences to avoid breaking entities
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < max_length:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def normalize_entity(entity: str) -> str:
    """Normalize entity text for better matching"""
    # Remove extra whitespace and normalize case
    normalized = re.sub(r'\s+', ' ', entity.strip()).lower()
    # Remove common prefixes/suffixes
    normalized = re.sub(r'^(the|a|an)\s+', '', normalized)
    normalized = re.sub(r'\s+(inc|corp|llc|ltd|co|company|corporation)$', '', normalized)
    return normalized

def extract_keywords(text: str, title: str = "") -> Dict[str, List[str]]:
    """Extract keywords using NLTK and custom patterns"""
    # Combine text
    full_text = f"{title} {text}".lower()
    
    # Tokenize and clean
    tokens = word_tokenize(full_text)
    stop_words = set(stopwords.words('english'))
    
    # Remove punctuation and stopwords
    keywords = []
    for token in tokens:
        if token not in stop_words and token not in string.punctuation and len(token) > 2:
            keywords.append(token)
    
    # Count frequencies
    keyword_freq = Counter(keywords)
    
    # Extract by category
    categorized_keywords = {}
    for category, patterns in STRAITWATCH_KEYWORDS.items():
        found = []
        for pattern in patterns:
            if pattern in full_text:
                found.append(pattern)
        if found:
            categorized_keywords[category] = found
    
    # Add top general keywords
    top_keywords = [word for word, freq in keyword_freq.most_common(20) if freq > 1]
    if top_keywords:
        categorized_keywords["general"] = top_keywords[:10]
    
    return categorized_keywords

def detect_sentiment(text: str) -> Dict[str, Any]:
    """Simple sentiment detection based on keyword analysis"""
    positive_words = {
        "cooperation", "peace", "dialogue", "agreement", "partnership", "stability", "progress",
        "development", "growth", "success", "positive", "constructive", "friendly", "mutual"
    }
    negative_words = {
        "conflict", "tension", "threat", "aggression", "hostility", "sanction", "blockade",
        "invasion", "war", "crisis", "danger", "risk", "escalation", "provocation"
    }
    
    text_lower = text.lower()
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
    if positive_count > negative_count:
        sentiment = "positive"
        confidence = min(0.9, (positive_count - negative_count) / max(positive_count, 1))
    elif negative_count > positive_count:
        sentiment = "negative"
        confidence = min(0.9, (negative_count - positive_count) / max(negative_count, 1))
    else:
        sentiment = "neutral"
        confidence = 0.5
    
    return {
        "sentiment": sentiment,
        "confidence": confidence,
        "positive_score": positive_count,
        "negative_score": negative_count
    }

def detect_topics_from_ner(entities_by_type: Dict[str, List[str]]) -> List[str]:
    """
    Determine topics based on NER tag categories.
    Returns a list of topics like 'military', 'diplomatic', 'economic', 'political', 'technology', 'security'.
    """
    topics = set()
    # Example logic, can be expanded
    orgs = [e.lower() for e in entities_by_type.get('actor', [])]
    products = [e.lower() for e in entities_by_type.get('product', [])]
    commands = [e.lower() for e in entities_by_type.get('command', [])]
    techs = [e.lower() for e in entities_by_type.get('technology', [])]
    events = [e.lower() for e in entities_by_type.get('event', [])]

    # Military
    if any(x in orgs for x in ["pla", "ministry of defense", "navy"]) or any(x in products for x in ["missile", "aircraft"]):
        topics.add("military")
    # Political
    if any(x in commands for x in ["president", "ministry", "minister", "government"]) or any(x in orgs for x in ["party", "congress", "ministry"]):
        topics.add("political")
    # Technology
    if any(x in techs for x in ["ai", "artificial intelligence", "r&d", "research"]):
        topics.add("technology")
    # Diplomatic
    if any(x in events for x in ["summit", "talks", "negotiation", "diplomacy"]):
        topics.add("diplomatic")
    # Economic
    if any(x in orgs for x in ["bank", "commerce", "trade", "finance"]):
        topics.add("economic")
    # Security
    if any(x in orgs for x in ["security", "coast guard", "police"]):
        topics.add("security")
    return list(topics)

def score_escalation_from_entities(entities: List[str]) -> float:
    """
    Score escalation based on weighted entity matches.
    Returns a float between 0 and 1.0.
    """
    ESCALATION_WEIGHTS = {
        "taiwan": 1.0,
        "pla": 0.9,
        "invasion": 0.95,
        "missile": 0.85,
        "south china sea": 0.7,
        "navy": 0.7
    }
    score = sum(ESCALATION_WEIGHTS.get(e.lower(), 0) for e in entities)
    normalized = min(score / 3.0, 1.0)
    return round(normalized, 3)

def tag_article(text: str, title: str = "") -> Dict[str, Any]:
    """
    Enhanced article tagging with comprehensive analysis - OPTIMIZED VERSION
    """
    logger.info(f"[TAGGER] Starting optimized NER analysis")
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
    
    # OPTIMIZED NER analysis with chunking
    if nlp:
        logger.info("[TAGGER] Performing optimized NER analysis with chunking...")
        
        # Split text into chunks for faster processing
        chunks = chunk_text(full_text, max_length=50000)  # Smaller chunks for speed
        logger.info(f"[TAGGER] Processing {len(chunks)} chunks")
        
        # Process chunks in batches
        for i, chunk in enumerate(chunks):
            logger.info(f"[TAGGER] Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
            doc = nlp(chunk)
            
            for ent in doc.ents:
                label = ent.label_
                value = ent.text.strip()
                normalized_value = normalize_entity(value)
                
                # Map spaCy entity types to our categories with normalization
                if label == "GPE" or label == "LOC":
                    tag_categories["geo"].append(normalized_value)
                    flat_tags.add(f"GEO:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "ORG":
                    tag_categories["actor"].append(normalized_value)
                    flat_tags.add(f"ACT:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "PERSON":
                    tag_categories["command"].append(normalized_value)
                    flat_tags.add(f"CMD:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "EVENT":
                    tag_categories["event"].append(normalized_value)
                    flat_tags.add(f"EVT:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "FAC":
                    tag_categories["facility"].append(normalized_value)
                    flat_tags.add(f"FAC:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "PRODUCT":
                    tag_categories["technology"].append(normalized_value)
                    flat_tags.add(f"TECH:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "DATE":
                    tag_categories["time"].append(normalized_value)
                    flat_tags.add(f"TIME:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "TIME":
                    tag_categories["time"].append(normalized_value)
                    flat_tags.add(f"TIME:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "QUANTITY":
                    tag_categories["quantity"].append(normalized_value)
                    flat_tags.add(f"QTY:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "CARDINAL":
                    tag_categories["quantity"].append(normalized_value)
                    flat_tags.add(f"QTY:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "MONEY":
                    tag_categories["money"].append(normalized_value)
                    flat_tags.add(f"MONEY:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "LAW":
                    tag_categories["law"].append(normalized_value)
                    flat_tags.add(f"LAW:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "LANGUAGE":
                    tag_categories["language"].append(normalized_value)
                    flat_tags.add(f"LANG:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "NORP":
                    tag_categories["nationality"].append(normalized_value)
                    flat_tags.add(f"NAT:{normalized_value}")
                    all_entities.add(normalized_value)
                elif label == "ORDINAL":
                    tag_categories["ordinal"].append(normalized_value)
                    flat_tags.add(f"ORD:{normalized_value}")
                    all_entities.add(normalized_value)
                else:
                    # Catch any other entity types
                    all_entities.add(normalized_value)
        
        logger.info(f"[TAGGER] Found {len(all_entities)} total entities across {len(chunks)} chunks")
    
    # Enhanced analysis: Sentiment detection
    logger.info("[TAGGER] Detecting sentiment...")
    sentiment = detect_sentiment(full_text)
    
    # Enhanced analysis: Topic detection
    logger.info("[TAGGER] Detecting topics...")
    topics = detect_topics_from_ner(tag_categories)
    
    # Enhanced analysis: Priority calculation
    logger.info("[TAGGER] Calculating priority...")
    escalation_score = score_escalation_from_entities(list(all_entities))
    
    # Deduplicate and sort
    for k in tag_categories:
        tag_categories[k] = sorted(list(set(tag_categories[k])))
    flat_tags = sorted(list(flat_tags))
    all_entities = sorted(list(all_entities))
    
    logger.info(f"SUCCESS: [TAGGER] Optimized tagging complete:")
    logger.info(f"   Tags: {len(flat_tags)}")
    logger.info(f"   Entities: {len(all_entities)}")
    logger.info(f"   Sentiment: {sentiment}")
    logger.info(f"   Topics: {topics}")
    logger.info(f"   Priority: {escalation_score}")
    
    return {
        "tag_categories": tag_categories,
        "tags": flat_tags,
        "entities": all_entities,
        "sentiment": sentiment,
        "topics": topics,
        "priority_level": escalation_score
    }

def tag_article_batch(articles: List[Dict[str, Any]], text_key: str = "body", title_key: str = "title") -> List[Dict[str, Any]]:
    """
    Batch tag a list of articles using spaCy's nlp.pipe for fast NER.
    Each article should be a dict with at least a text_key and optionally a title_key.
    Returns a list of tagging results (same order as input).
    """
    if not nlp:
        raise RuntimeError("spaCy model not loaded")
    
    # Prepare texts (with chunking for long articles)
    texts = []
    article_indices = []  # To map back to original articles
    for idx, article in enumerate(articles):
        title = article.get(title_key, "")
        text = article.get(text_key, "")
        full_text = f"{title} {text}".strip()
        chunks = chunk_text(full_text, max_length=50000)
        for chunk in chunks:
            texts.append(chunk)
            article_indices.append(idx)
    
    # Run NER in batch
    results = [{} for _ in articles]
    for doc, idx in zip(nlp.pipe(texts, batch_size=16, disable=["tagger", "parser"]), article_indices):
        # Use the same logic as tag_article for each chunk
        tag_categories = {
            "geo": [], "actor": [], "command": [], "event": [], "facility": [], "technology": [],
            "time": [], "quantity": [], "money": [], "law": [], "language": [], "nationality": [], "ordinal": []
        }
        flat_tags: Set[str] = set()
        all_entities: Set[str] = set()
        for ent in doc.ents:
            label = ent.label_
            value = ent.text.strip()
            normalized_value = normalize_entity(value)
            if label == "GPE" or label == "LOC":
                tag_categories["geo"].append(normalized_value)
                flat_tags.add(f"GEO:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "ORG":
                tag_categories["actor"].append(normalized_value)
                flat_tags.add(f"ACT:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "PERSON":
                tag_categories["command"].append(normalized_value)
                flat_tags.add(f"CMD:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "EVENT":
                tag_categories["event"].append(normalized_value)
                flat_tags.add(f"EVT:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "FAC":
                tag_categories["facility"].append(normalized_value)
                flat_tags.add(f"FAC:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "PRODUCT":
                tag_categories["technology"].append(normalized_value)
                flat_tags.add(f"TECH:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "DATE":
                tag_categories["time"].append(normalized_value)
                flat_tags.add(f"TIME:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "TIME":
                tag_categories["time"].append(normalized_value)
                flat_tags.add(f"TIME:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "QUANTITY":
                tag_categories["quantity"].append(normalized_value)
                flat_tags.add(f"QTY:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "CARDINAL":
                tag_categories["quantity"].append(normalized_value)
                flat_tags.add(f"QTY:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "MONEY":
                tag_categories["money"].append(normalized_value)
                flat_tags.add(f"MONEY:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "LAW":
                tag_categories["law"].append(normalized_value)
                flat_tags.add(f"LAW:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "LANGUAGE":
                tag_categories["language"].append(normalized_value)
                flat_tags.add(f"LANG:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "NORP":
                tag_categories["nationality"].append(normalized_value)
                flat_tags.add(f"NAT:{normalized_value}")
                all_entities.add(normalized_value)
            elif label == "ORDINAL":
                tag_categories["ordinal"].append(normalized_value)
                flat_tags.add(f"ORD:{normalized_value}")
                all_entities.add(normalized_value)
            else:
                all_entities.add(normalized_value)
        # Deduplicate and sort
        for k in tag_categories:
            tag_categories[k] = sorted(list(set(tag_categories[k])))
        flat_tags = sorted(list(flat_tags))
        all_entities = sorted(list(all_entities))
        # Store result for this article (merge if multiple chunks)
        if not results[idx]:
            results[idx] = {
                "tag_categories": tag_categories,
                "tags": flat_tags,
                "entities": all_entities
            }
        else:
            # Merge with previous chunk
            for k in tag_categories:
                results[idx]["tag_categories"][k].extend(tag_categories[k])
            results[idx]["tags"].extend(flat_tags)
            results[idx]["entities"].extend(all_entities)
    # Final deduplication
    for res in results:
        for k in res.get("tag_categories", {}):
            res["tag_categories"][k] = sorted(list(set(res["tag_categories"][k])))
        if "tags" in res:
            res["tags"] = sorted(list(set(res["tags"])) )
        if "entities" in res:
            res["entities"] = sorted(list(set(res["entities"])) )
    return results

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