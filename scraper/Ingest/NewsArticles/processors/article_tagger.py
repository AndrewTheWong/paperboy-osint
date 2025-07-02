#!/usr/bin/env python3
"""
Unified Article Tagger

Consolidates all article tagging functionality including:
- NER (Named Entity Recognition)  
- Geographic extraction and tagging
- Event extraction
- Keyword extraction
- Sentiment analysis
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime

# Text processing
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

from textblob import TextBlob

logger = logging.getLogger(__name__)

class UnifiedArticleTagger:
    """Unified article tagger with all tagging capabilities."""
    
    def __init__(self):
        """Initialize the unified tagger."""
        self.nlp = None
        if HAS_SPACY:
            self._load_spacy_model()
            
        # Enhanced location database
        self.location_db = self._build_location_database()
        
        # Event keywords
        self.event_keywords = self._build_event_keywords()
        
        # Crisis/Escalation keywords
        self.crisis_keywords = self._build_crisis_keywords()
        
    def _load_spacy_model(self):
        """Load spaCy model for NER."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model for NER")
        except OSError:
            logger.warning("⚠️ spaCy model not found")
            self.nlp = None
            
    def _build_location_database(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive location database."""
        return {
            # Major Cities
            'beijing': {'country': 'China', 'region': 'Asia', 'type': 'city'},
            'shanghai': {'country': 'China', 'region': 'Asia', 'type': 'city'},
            'hong kong': {'country': 'Hong Kong', 'region': 'Asia', 'type': 'city'},
            'taipei': {'country': 'Taiwan', 'region': 'Asia', 'type': 'city'},
            'tokyo': {'country': 'Japan', 'region': 'Asia', 'type': 'city'},
            'seoul': {'country': 'South Korea', 'region': 'Asia', 'type': 'city'},
            'pyongyang': {'country': 'North Korea', 'region': 'Asia', 'type': 'city'},
            'moscow': {'country': 'Russia', 'region': 'Europe', 'type': 'city'},
            'washington': {'country': 'USA', 'region': 'North America', 'type': 'city'},
            'london': {'country': 'UK', 'region': 'Europe', 'type': 'city'},
            'paris': {'country': 'France', 'region': 'Europe', 'type': 'city'},
            'berlin': {'country': 'Germany', 'region': 'Europe', 'type': 'city'},
            'kiev': {'country': 'Ukraine', 'region': 'Europe', 'type': 'city'},
            'kyiv': {'country': 'Ukraine', 'region': 'Europe', 'type': 'city'},
            'tehran': {'country': 'Iran', 'region': 'Middle East', 'type': 'city'},
            
            # Countries
            'china': {'region': 'Asia', 'type': 'country'},
            'taiwan': {'region': 'Asia', 'type': 'country'},
            'japan': {'region': 'Asia', 'type': 'country'},
            'south korea': {'region': 'Asia', 'type': 'country'},
            'north korea': {'region': 'Asia', 'type': 'country'},
            'russia': {'region': 'Europe', 'type': 'country'},
            'usa': {'region': 'North America', 'type': 'country'},
            'united states': {'region': 'North America', 'type': 'country'},
            'uk': {'region': 'Europe', 'type': 'country'},
            'united kingdom': {'region': 'Europe', 'type': 'country'},
            'ukraine': {'region': 'Europe', 'type': 'country'},
            'iran': {'region': 'Middle East', 'type': 'country'},
            'israel': {'region': 'Middle East', 'type': 'country'},
        }
        
    def _build_event_keywords(self) -> Dict[str, List[str]]:
        """Build event keyword categories."""
        return {
            'military': [
                'military', 'army', 'navy', 'troops', 'soldiers', 'deployment',
                'exercise', 'defense', 'weapons', 'missile', 'fighter'
            ],
            'conflict': [
                'war', 'conflict', 'battle', 'attack', 'strike', 'bombing',
                'invasion', 'violence', 'clash', 'hostilities'
            ],
            'diplomatic': [
                'diplomatic', 'embassy', 'summit', 'talks', 'negotiation',
                'agreement', 'treaty', 'alliance', 'relations'
            ],
            'economic': [
                'trade', 'economy', 'sanctions', 'tariff', 'investment',
                'market', 'financial', 'business'
            ],
            'political': [
                'government', 'president', 'minister', 'election', 'policy',
                'politics', 'democracy', 'leader'
            ]
        }
        
    def _build_crisis_keywords(self) -> List[str]:
        """Build crisis/escalation keywords."""
        return [
            'crisis', 'emergency', 'urgent', 'escalation', 'tension', 'threat',
            'critical', 'concern', 'risk', 'danger', 'incident', 'attack',
            'violence', 'protest', 'unrest', 'instability'
        ]
        
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        entities = {
            'PERSON': [],
            'ORG': [],
            'GPE': [],
            'LOCATION': []
        }
        
        if not text or not self.nlp:
            return entities
            
        try:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in ['PERSON']:
                    entities['PERSON'].append(ent.text)
                elif ent.label_ in ['ORG']:
                    entities['ORG'].append(ent.text)
                elif ent.label_ in ['GPE']:
                    entities['GPE'].append(ent.text)
                elif ent.label_ in ['LOC']:
                    entities['LOCATION'].append(ent.text)
                    
            # Remove duplicates
            for key in entities:
                entities[key] = list(set(entities[key]))
                
        except Exception as e:
            logger.warning(f"Entity extraction failed: {e}")
            
        return entities
        
    def extract_geographic_info(self, text: str) -> Dict[str, Any]:
        """Extract geographic information from text."""
        result = {
            'locations': [],
            'countries': [],
            'cities': [],
            'regions': [],
            'primary_country': None,
            'primary_region': None,
            'confidence': 0.0
        }
        
        if not text:
            return result
            
        text_lower = text.lower()
        found_locations = []
        
        # Search for locations in database
        for location_name, location_info in self.location_db.items():
            if location_name in text_lower:
                found_locations.append({
                    'name': location_name.title(),
                    'info': location_info
                })
                
        if found_locations:
            # Process found locations
            for location in found_locations:
                name = location['name']
                info = location['info']
                
                result['locations'].append(name)
                
                if info['type'] == 'country':
                    result['countries'].append(name)
                    if not result['primary_country']:
                        result['primary_country'] = name
                        result['primary_region'] = info.get('region')
                        
                elif info['type'] == 'city':
                    result['cities'].append(name)
                    if 'country' in info:
                        country = info['country']
                        if country not in result['countries']:
                            result['countries'].append(country)
                        if not result['primary_country']:
                            result['primary_country'] = country
                            result['primary_region'] = info.get('region')
                            
                # Add region
                if 'region' in info and info['region'] not in result['regions']:
                    result['regions'].append(info['region'])
                    
            # Calculate confidence
            result['confidence'] = min(1.0, len(found_locations) * 0.2)
            
        return result
        
    def extract_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """Extract keywords from text."""
        if not text:
            return []
            
        # Simple keyword extraction
        words = re.findall(r'\w+', text.lower())
        
        # Filter out short words and common words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        keywords = [word for word in words 
                   if len(word) > 3 and word not in stop_words]
        
        # Get word frequency
        word_freq = {}
        for word in keywords:
            word_freq[word] = word_freq.get(word, 0) + 1
            
        # Sort by frequency and return top keywords
        sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_keywords[:max_keywords]]
        
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment of text."""
        result = {
            'polarity': 0.0,
            'subjectivity': 0.0,
            'sentiment_label': 'neutral'
        }
        
        if not text:
            return result
            
        try:
            blob = TextBlob(text)
            result['polarity'] = blob.sentiment.polarity
            result['subjectivity'] = blob.sentiment.subjectivity
            
            # Label sentiment
            if result['polarity'] > 0.1:
                result['sentiment_label'] = 'positive'
            elif result['polarity'] < -0.1:
                result['sentiment_label'] = 'negative'
            else:
                result['sentiment_label'] = 'neutral'
                
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            
        return result
        
    def classify_events(self, text: str) -> Dict[str, float]:
        """Classify events in text."""
        event_scores = {}
        
        if not text:
            return event_scores
            
        text_lower = text.lower()
        
        for event_type, keywords in self.event_keywords.items():
            score = 0.0
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
                    
            # Normalize score
            if score > 0:
                event_scores[event_type] = min(1.0, score / len(keywords) * 5)
                
        return event_scores
        
    def calculate_escalation_score(self, text: str) -> float:
        """Calculate escalation/crisis score for text."""
        if not text:
            return 0.0
            
        text_lower = text.lower()
        crisis_count = 0
        
        for keyword in self.crisis_keywords:
            if keyword in text_lower:
                crisis_count += 1
                
        # Normalize to 0-1 scale
        return min(1.0, crisis_count / 10.0)
        
    def tag_article(self, title: str, content: str) -> Dict[str, Any]:
        """Complete article tagging pipeline."""
        
        # Combine title and content for analysis
        full_text = f"{title} {content}".strip()
        
        result = {
            'title': title,
            'content_length': len(content),
            'word_count': len(content.split()) if content else 0,
            'timestamp': datetime.now().isoformat(),
            
            # NER
            'entities': self.extract_entities(full_text),
            
            # Geographic
            'geographic_info': self.extract_geographic_info(full_text),
            
            # Keywords
            'keywords': self.extract_keywords(full_text),
            
            # Sentiment
            'sentiment': self.analyze_sentiment(full_text),
            
            # Events
            'event_classification': self.classify_events(full_text),
            
            # Escalation
            'escalation_score': self.calculate_escalation_score(full_text),
            
            # Quality scores
            'needs_human_review': False,
            'tagging_confidence': 0.8
        }
        
        # Determine if needs human review
        result['needs_human_review'] = (
            result['escalation_score'] > 0.5 or
            len(result['geographic_info']['countries']) > 2 or
            result['sentiment']['polarity'] < -0.5
        )
        
        return result