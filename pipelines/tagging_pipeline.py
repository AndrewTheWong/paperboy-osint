#!/usr/bin/env python3
"""
Article Tagging Pipeline

This module provides keyword-based and ML-based tagging for news articles.
It extracts relevant tags from translated text and flags articles for review
when tags are weak or unknown.

Features:
- Keyword-based tagging using predefined tag vocabularies
- ML-based tagging using pre-trained models
- Confidence scoring for tag quality
- Review flagging for low-confidence results
- Configurable tag thresholds

Output:
- tags: Combined final tags
- ml_tags: ML-predicted tags
- keyword_tags: Keyword-matched tags
- needs_review: Boolean flag for manual review
"""

import os
import sys
import json
import logging
import re
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TagConfig:
    """Configuration for tagging thresholds and parameters."""
    keyword_confidence_threshold: float = 0.6
    ml_confidence_threshold: float = 0.7
    review_threshold: float = 0.5
    max_tags_per_article: int = 5
    min_text_length: int = 50

# Tag vocabulary and keyword mappings
TAG_VOCABULARY = {
    "military": {
        "keywords": [
            "military", "army", "navy", "air force", "marines", "troops", "soldiers",
            "drill", "exercise", "maneuver", "deployment", "base", "command",
            "defense", "weapons", "missile", "aircraft", "warship", "tank",
            "special forces", "battalion", "regiment", "division", "fleet"
        ],
        "weight": 1.0,
        "description": "Military activities and personnel"
    },
    "conflict": {
        "keywords": [
            "conflict", "war", "battle", "fight", "clash", "combat", "violence",
            "attack", "assault", "strike", "raid", "invasion", "offensive",
            "skirmish", "engagement", "confrontation", "hostility", "aggression"
        ],
        "weight": 1.2,
        "description": "Armed conflicts and violent confrontations"
    },
    "diplomatic": {
        "keywords": [
            "diplomatic", "diplomacy", "talks", "negotiation", "summit", "meeting",
            "ambassador", "embassy", "treaty", "agreement", "accord", "dialogue",
            "foreign minister", "secretary of state", "bilateral", "multilateral",
            "peace talks", "mediation", "arbitration"
        ],
        "weight": 0.8,
        "description": "Diplomatic activities and negotiations"
    },
    "cyber": {
        "keywords": [
            "cyber", "cyberattack", "hack", "hacker", "malware", "ransomware",
            "breach", "digital", "online", "internet", "network", "computer",
            "infrastructure", "data", "information warfare", "espionage"
        ],
        "weight": 1.1,
        "description": "Cyber warfare and digital security"
    },
    "nuclear": {
        "keywords": [
            "nuclear", "atomic", "reactor", "uranium", "plutonium", "enrichment",
            "warhead", "bomb", "missile", "icbm", "deterrent", "proliferation",
            "nonproliferation", "iaea", "sanctions", "inspection"
        ],
        "weight": 1.3,
        "description": "Nuclear weapons and technology"
    },
    "economic": {
        "keywords": [
            "economic", "trade", "economy", "sanctions", "embargo", "tariff",
            "investment", "market", "business", "commerce", "finance",
            "currency", "supply chain", "export", "import", "gdp"
        ],
        "weight": 0.9,
        "description": "Economic and trade relations"
    },
    "territorial": {
        "keywords": [
            "territory", "territorial", "border", "boundary", "sovereignty",
            "island", "strait", "sea", "zone", "eez", "adiz", "claim",
            "disputed", "occupation", "annexation", "recognition"
        ],
        "weight": 1.1,
        "description": "Territorial disputes and sovereignty"
    },
    "intelligence": {
        "keywords": [
            "intelligence", "spy", "espionage", "surveillance", "cia", "fbi",
            "reconnaissance", "monitoring", "classified", "secret", "covert",
            "undercover", "agent", "operative", "leak", "whistleblower"
        ],
        "weight": 1.0,
        "description": "Intelligence and espionage activities"
    },
    "protest": {
        "keywords": [
            "protest", "demonstration", "rally", "march", "riot", "unrest",
            "activist", "dissent", "opposition", "civil disobedience",
            "uprising", "rebellion", "revolution", "crackdown", "suppression"
        ],
        "weight": 0.7,
        "description": "Protests and civil unrest"
    },
    "terrorism": {
        "keywords": [
            "terror", "terrorist", "terrorism", "extremist", "radical",
            "militant", "insurgent", "bomb", "explosion", "attack",
            "suicide bomber", "hostage", "kidnapping", "assassination"
        ],
        "weight": 1.4,
        "description": "Terrorist activities and extremism"
    }
}

# Regional keywords for context
REGIONAL_KEYWORDS = {
    "asia_pacific": [
        "taiwan", "china", "japan", "korea", "philippines", "vietnam",
        "south china sea", "east china sea", "taiwan strait", "senkaku",
        "diaoyu", "spratly", "paracel", "asean", "quad"
    ],
    "europe": [
        "ukraine", "russia", "nato", "eu", "european union", "belarus",
        "poland", "germany", "france", "uk", "baltic", "crimea"
    ],
    "middle_east": [
        "israel", "palestine", "iran", "iraq", "syria", "lebanon",
        "saudi arabia", "yemen", "gaza", "west bank", "gulf"
    ],
    "africa": [
        "ethiopia", "sudan", "somalia", "mali", "nigeria", "congo",
        "sahel", "horn of africa", "african union"
    ]
}

class KeywordTagger:
    """Keyword-based tagging engine."""
    
    def __init__(self, config: TagConfig = TagConfig()):
        self.config = config
        self.tag_vocab = TAG_VOCABULARY
        self.regional_vocab = REGIONAL_KEYWORDS
    
    def extract_keyword_tags(self, text: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Extract tags using keyword matching.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (tags, confidence_scores)
        """
        if not text or len(text) < self.config.min_text_length:
            return [], {}
        
        text_lower = text.lower()
        tag_scores = {}
        
        # Score each tag category
        for tag, info in self.tag_vocab.items():
            keywords = info["keywords"]
            weight = info["weight"]
            
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                # Calculate confidence based on match count and weight
                confidence = min(1.0, (matches / len(keywords)) * weight * 2)
                tag_scores[tag] = confidence
        
        # Filter by confidence threshold
        valid_tags = [
            tag for tag, score in tag_scores.items() 
            if score >= self.config.keyword_confidence_threshold
        ]
        
        # Sort by confidence and limit count
        valid_tags = sorted(valid_tags, key=lambda x: tag_scores[x], reverse=True)
        valid_tags = valid_tags[:self.config.max_tags_per_article]
        
        return valid_tags, tag_scores
    
    def extract_regional_context(self, text: str) -> List[str]:
        """Extract regional context from text."""
        text_lower = text.lower()
        regions = []
        
        for region, keywords in self.regional_vocab.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            if matches > 0:
                regions.append(region)
        
        return regions

class MLTagger:
    """Machine learning-based tagging engine."""
    
    def __init__(self, config: TagConfig = TagConfig()):
        self.config = config
        self.model = None
        self.vectorizer = None
        self._load_models()
    
    def _load_models(self):
        """Load pre-trained ML models for tagging."""
        # In production, load actual trained models
        # For now, we'll simulate ML tagging using keyword patterns
        logger.info("ML tagging models not yet trained - using enhanced keyword patterns")
        
    def extract_ml_tags(self, text: str) -> Tuple[List[str], Dict[str, float]]:
        """
        Extract tags using ML models.
        
        Args:
            text: Input text to analyze
            
        Returns:
            Tuple of (tags, confidence_scores)
        """
        if not text or len(text) < self.config.min_text_length:
            return [], {}
        
        # Simulate ML tagging with enhanced pattern matching
        # In production, this would use trained classifiers
        ml_scores = self._simulate_ml_scoring(text)
        
        # Filter by confidence threshold
        valid_tags = [
            tag for tag, score in ml_scores.items() 
            if score >= self.config.ml_confidence_threshold
        ]
        
        # Sort by confidence and limit count
        valid_tags = sorted(valid_tags, key=lambda x: ml_scores[x], reverse=True)
        valid_tags = valid_tags[:self.config.max_tags_per_article]
        
        return valid_tags, ml_scores
    
    def _simulate_ml_scoring(self, text: str) -> Dict[str, float]:
        """Simulate ML scoring using advanced pattern matching."""
        text_lower = text.lower()
        scores = {}
        
        # Enhanced patterns that ML might detect
        advanced_patterns = {
            "escalation": [
                r"\b(escalat|heighten|intensif|increas.*tension)\w*\b",
                r"\b(rising.*tension|growing.*concern|mounting.*pressure)\b",
                r"\b(deteriorat.*relation|worsen.*situation)\b"
            ],
            "de_escalation": [
                r"\b(de-?escalat|calm|ease.*tension|reduc.*tension)\w*\b",
                r"\b(peace.*talk|diplomatic.*solution|negotiat.*settlement)\b",
                r"\b(ceasefire|truce|armistice)\b"
            ],
            "military_buildup": [
                r"\b(buildup|build.*up|deploy.*force|position.*troop)\b",
                r"\b(reinforce|strengthen.*defense|mobiliz)\w*\b",
                r"\b(military.*exercise.*near|show.*of.*force)\b"
            ],
            "economic_pressure": [
                r"\b(sanction|embargo|restrict.*trade)\b",
                r"\b(economic.*pressure|financial.*measure)\b",
                r"\b(block.*asset|freeze.*fund)\b"
            ]
        }
        
        # Score based on pattern matches
        for tag, patterns in advanced_patterns.items():
            pattern_matches = sum(
                len(re.findall(pattern, text_lower, re.IGNORECASE))
                for pattern in patterns
            )
            if pattern_matches > 0:
                scores[tag] = min(1.0, pattern_matches * 0.3 + 0.4)
        
        return scores

class ArticleTaggingPipeline:
    """Main tagging pipeline combining keyword and ML approaches."""
    
    def __init__(self, config: TagConfig = TagConfig()):
        self.config = config
        self.keyword_tagger = KeywordTagger(config)
        self.ml_tagger = MLTagger(config)
    
    def tag_article(self, article: Dict) -> Dict:
        """
        Tag a single article with all available methods.
        
        Args:
            article: Article dictionary with 'text' field
            
        Returns:
            Article dictionary with added tagging fields
        """
        text = article.get('translated_text', article.get('text', ''))
        if not text:
            logger.warning(f"No text found for article {article.get('id', 'unknown')}")
            return self._add_empty_tags(article)
        
        # Extract keyword tags
        keyword_tags, keyword_scores = self.keyword_tagger.extract_keyword_tags(text)
        
        # Extract ML tags
        ml_tags, ml_scores = self.ml_tagger.extract_ml_tags(text)
        
        # Extract regional context
        regional_context = self.keyword_tagger.extract_regional_context(text)
        
        # Combine tags intelligently
        combined_tags, overall_confidence = self._combine_tags(
            keyword_tags, keyword_scores, ml_tags, ml_scores
        )
        
        # Determine if manual review is needed
        needs_review = self._needs_review(combined_tags, overall_confidence, text)
        
        # Add tagging results to article
        tagged_article = article.copy()
        tagged_article.update({
            'tags': combined_tags,
            'keyword_tags': keyword_tags,
            'ml_tags': ml_tags,
            'regional_context': regional_context,
            'tag_confidence': overall_confidence,
            'needs_review': needs_review,
            'tagging_metadata': {
                'keyword_scores': keyword_scores,
                'ml_scores': ml_scores,
                'text_length': len(text),
                'tag_count': len(combined_tags)
            }
        })
        
        return tagged_article
    
    def _combine_tags(self, keyword_tags: List[str], keyword_scores: Dict[str, float],
                     ml_tags: List[str], ml_scores: Dict[str, float]) -> Tuple[List[str], float]:
        """Combine keyword and ML tags intelligently."""
        
        # Start with all unique tags
        all_tags = set(keyword_tags + ml_tags)
        
        # Calculate combined confidence for each tag
        tag_confidences = {}
        for tag in all_tags:
            kw_conf = keyword_scores.get(tag, 0)
            ml_conf = ml_scores.get(tag, 0)
            
            # Weighted combination (keyword: 0.4, ML: 0.6)
            combined_conf = 0.4 * kw_conf + 0.6 * ml_conf
            tag_confidences[tag] = combined_conf
        
        # Sort by confidence and take top tags
        sorted_tags = sorted(tag_confidences.items(), key=lambda x: x[1], reverse=True)
        final_tags = [tag for tag, conf in sorted_tags[:self.config.max_tags_per_article]]
        
        # Calculate overall confidence
        if final_tags:
            overall_confidence = np.mean([tag_confidences[tag] for tag in final_tags])
        else:
            overall_confidence = 0.0
        
        return final_tags, overall_confidence
    
    def _needs_review(self, tags: List[str], confidence: float, text: str) -> bool:
        """Determine if article needs manual review."""
        
        # Review if confidence is low
        if confidence < self.config.review_threshold:
            return True
        
        # Review if no tags found
        if not tags:
            return True
        
        # Review if text is very short
        if len(text) < self.config.min_text_length * 2:
            return True
        
        # Review if high-stakes tags are detected
        high_stakes_tags = {"nuclear", "terrorism", "conflict"}
        if any(tag in high_stakes_tags for tag in tags):
            return True
        
        return False
    
    def _add_empty_tags(self, article: Dict) -> Dict:
        """Add empty tagging fields to article."""
        tagged_article = article.copy()
        tagged_article.update({
            'tags': [],
            'keyword_tags': [],
            'ml_tags': [],
            'regional_context': [],
            'tag_confidence': 0.0,
            'needs_review': True,
            'tagging_metadata': {
                'keyword_scores': {},
                'ml_scores': {},
                'text_length': 0,
                'tag_count': 0
            }
        })
        return tagged_article
    
    def tag_articles_batch(self, articles: List[Dict]) -> List[Dict]:
        """Tag a batch of articles."""
        logger.info(f"Tagging {len(articles)} articles...")
        
        tagged_articles = []
        review_count = 0
        
        for i, article in enumerate(articles):
            tagged_article = self.tag_article(article)
            tagged_articles.append(tagged_article)
            
            if tagged_article['needs_review']:
                review_count += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Tagged {i + 1}/{len(articles)} articles")
        
        # Log summary statistics
        total_tags = sum(len(article['tags']) for article in tagged_articles)
        avg_tags = total_tags / len(articles) if articles else 0
        review_pct = (review_count / len(articles)) * 100 if articles else 0
        
        logger.info(f"Tagging complete: {avg_tags:.1f} avg tags/article, {review_count} need review ({review_pct:.1f}%)")
        
        return tagged_articles

def load_articles_from_json(filepath: str) -> List[Dict]:
    """Load articles from JSON file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            articles = json.load(f)
        logger.info(f"Loaded {len(articles)} articles from {filepath}")
        return articles
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []
    except json.JSONDecodeError as e:
        logger.error(f"JSON decode error in {filepath}: {e}")
        return []

def save_tagged_articles(articles: List[Dict], filepath: str):
    """Save tagged articles to JSON file."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(articles, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(articles)} tagged articles to {filepath}")
    except Exception as e:
        logger.error(f"Error saving to {filepath}: {e}")

def main():
    """Main tagging pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Tag articles using keyword and ML approaches')
    parser.add_argument('--input', default='data/translated_articles.json', 
                       help='Input JSON file with articles')
    parser.add_argument('--output', default='data/tagged_articles.json',
                       help='Output JSON file for tagged articles')
    parser.add_argument('--config', help='JSON config file for tagging parameters')
    args = parser.parse_args()
    
    # Load configuration
    config = TagConfig()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = TagConfig(**config_dict)
    
    # Initialize pipeline
    pipeline = ArticleTaggingPipeline(config)
    
    # Load articles
    articles = load_articles_from_json(args.input)
    if not articles:
        logger.error("No articles to process")
        return
    
    # Tag articles
    tagged_articles = pipeline.tag_articles_batch(articles)
    
    # Save results
    save_tagged_articles(tagged_articles, args.output)
    
    # Print summary
    review_articles = [a for a in tagged_articles if a['needs_review']]
    logger.info(f"✅ Tagging pipeline completed")
    logger.info(f"📊 Summary: {len(tagged_articles)} articles processed, {len(review_articles)} flagged for review")

if __name__ == "__main__":
    main() 