"""
Named Entity Recognition system for intelligence analysis.
Extracts people, places, organizations, and other entities from multilingual text.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any
import re
from dataclasses import dataclass
from datetime import datetime
import json
import spacy
from transformers import (
    AutoTokenizer, AutoModelForTokenClassification, 
    pipeline, BertTokenizer, BertForTokenClassification
)
import torch

@dataclass
class EntityResult:
    """Represents a named entity extraction result."""
    text: str
    label: str
    confidence: float
    start_pos: int
    end_pos: int
    context: str
    language: str

@dataclass
class NERAnalysis:
    """Complete NER analysis result."""
    entities: List[EntityResult]
    people: List[str]
    organizations: List[str]
    locations: List[str]
    geopolitical_entities: List[str]
    military_terms: List[str]
    confidence_score: float
    processing_time: float
    language: str

class IntelligenceNER:
    """
    Advanced Named Entity Recognition for intelligence analysis.
    Supports both English and Chinese text with specialized entity categories.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.models = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Military and geopolitical keywords for enhanced detection
        self.military_keywords = {
            'en': ['military', 'defense', 'army', 'navy', 'air force', 'marines', 
                   'troops', 'soldiers', 'weapons', 'missile', 'fighter jet', 
                   'submarine', 'warship', 'base', 'deployment', 'exercise'],
            'zh': ['军事', '国防', '军队', '海军', '空军', '陆军', '部队', '士兵', 
                   '武器', '导弹', '战斗机', '潜艇', '军舰', '基地', '部署', '演习']
        }
        
        self.geopolitical_keywords = {
            'en': ['strait', 'border', 'territory', 'sovereignty', 'diplomatic', 
                   'embassy', 'consulate', 'summit', 'treaty', 'alliance'],
            'zh': ['海峡', '边界', '领土', '主权', '外交', '大使馆', '领事馆', 
                   '峰会', '条约', '联盟']
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize NER models for different languages."""
        try:
            # English NER model
            self.logger.info("Loading English NER model...")
            self.pipelines['en'] = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Chinese NER model
            self.logger.info("Loading Chinese NER model...")
            self.pipelines['zh'] = pipeline(
                "ner",
                model="ckiplab/bert-base-chinese-ner",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Multilingual model as fallback
            self.logger.info("Loading multilingual NER model...")
            self.pipelines['multi'] = pipeline(
                "ner",
                model="xlm-roberta-large-finetuned-conll03-english",
                aggregation_strategy="simple",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.logger.info("All NER models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Error initializing NER models: {e}")
            # Fallback to basic spacy models
            try:
                import spacy
                self.nlp_en = spacy.load("en_core_web_sm")
                self.nlp_zh = spacy.load("zh_core_web_sm")
                self.use_spacy_fallback = True
                self.logger.info("Using spaCy models as fallback")
            except:
                self.logger.error("No NER models available")
                self.use_spacy_fallback = False
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English."""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        
        if total_chars == 0:
            return 'en'
        
        chinese_ratio = chinese_chars / total_chars
        return 'zh' if chinese_ratio > 0.3 else 'en'
    
    def extract_entities(self, text: str, language: Optional[str] = None) -> NERAnalysis:
        """
        Extract named entities from text with intelligence-focused categorization.
        """
        start_time = datetime.now()
        
        if not text or not text.strip():
            return NERAnalysis(
                entities=[], people=[], organizations=[], locations=[],
                geopolitical_entities=[], military_terms=[],
                confidence_score=0.0, processing_time=0.0, language='unknown'
            )
        
        # Auto-detect language if not provided
        if language is None:
            language = self.detect_language(text)
        
        entities = []
        
        try:
            # Use appropriate model based on language
            if language == 'zh' and 'zh' in self.pipelines:
                ner_results = self.pipelines['zh'](text)
            elif language == 'en' and 'en' in self.pipelines:
                ner_results = self.pipelines['en'](text)
            else:
                # Use multilingual model as fallback
                ner_results = self.pipelines.get('multi', [])
            
            # Process NER results
            for result in ner_results:
                entity = EntityResult(
                    text=result['word'].replace('##', ''),  # Remove subword tokens
                    label=self._standardize_label(result['entity_group']),
                    confidence=result['score'],
                    start_pos=result['start'],
                    end_pos=result['end'],
                    context=self._extract_context(text, result['start'], result['end']),
                    language=language
                )
                entities.append(entity)
                
        except Exception as e:
            self.logger.error(f"Error in transformer NER: {e}")
            # Fallback to spacy if available
            if hasattr(self, 'use_spacy_fallback') and self.use_spacy_fallback:
                entities = self._spacy_fallback(text, language)
        
        # Enhance with keyword-based detection
        entities.extend(self._keyword_based_detection(text, language))
        
        # Categorize entities
        people = [e.text for e in entities if e.label in ['PERSON', 'PER']]
        organizations = [e.text for e in entities if e.label in ['ORG', 'ORGANIZATION']]
        locations = [e.text for e in entities if e.label in ['LOC', 'LOCATION', 'GPE']]
        geopolitical_entities = [e.text for e in entities if e.label == 'GEOPOLITICAL']
        military_terms = [e.text for e in entities if e.label == 'MILITARY']
        
        # Calculate overall confidence
        confidence_score = sum(e.confidence for e in entities) / len(entities) if entities else 0.0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return NERAnalysis(
            entities=entities,
            people=list(set(people)),
            organizations=list(set(organizations)),
            locations=list(set(locations)),
            geopolitical_entities=list(set(geopolitical_entities)),
            military_terms=list(set(military_terms)),
            confidence_score=confidence_score,
            processing_time=processing_time,
            language=language
        )
    
    def _standardize_label(self, label: str) -> str:
        """Standardize entity labels across different models."""
        label_mapping = {
            'PER': 'PERSON',
            'PERSON': 'PERSON',
            'ORG': 'ORGANIZATION',
            'ORGANIZATION': 'ORGANIZATION',
            'LOC': 'LOCATION',
            'LOCATION': 'LOCATION',
            'GPE': 'GEOPOLITICAL',
            'MISC': 'MISCELLANEOUS'
        }
        return label_mapping.get(label.upper(), label.upper())
    
    def _extract_context(self, text: str, start: int, end: int, context_size: int = 50) -> str:
        """Extract context around an entity."""
        context_start = max(0, start - context_size)
        context_end = min(len(text), end + context_size)
        return text[context_start:context_end].strip()
    
    def _keyword_based_detection(self, text: str, language: str) -> List[EntityResult]:
        """Detect military and geopolitical terms using keyword matching."""
        entities = []
        
        # Military terms
        military_keywords = self.military_keywords.get(language, [])
        for keyword in military_keywords:
            matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))
            for match in matches:
                entities.append(EntityResult(
                    text=match.group(),
                    label='MILITARY',
                    confidence=0.9,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self._extract_context(text, match.start(), match.end()),
                    language=language
                ))
        
        # Geopolitical terms
        geo_keywords = self.geopolitical_keywords.get(language, [])
        for keyword in geo_keywords:
            matches = list(re.finditer(re.escape(keyword), text, re.IGNORECASE))
            for match in matches:
                entities.append(EntityResult(
                    text=match.group(),
                    label='GEOPOLITICAL',
                    confidence=0.85,
                    start_pos=match.start(),
                    end_pos=match.end(),
                    context=self._extract_context(text, match.start(), match.end()),
                    language=language
                ))
        
        return entities
    
    def _spacy_fallback(self, text: str, language: str) -> List[EntityResult]:
        """Fallback NER using spaCy models."""
        entities = []
        
        try:
            if language == 'zh' and hasattr(self, 'nlp_zh'):
                doc = self.nlp_zh(text)
            else:
                doc = self.nlp_en(text)
            
            for ent in doc.ents:
                entities.append(EntityResult(
                    text=ent.text,
                    label=self._standardize_label(ent.label_),
                    confidence=0.8,  # spaCy doesn't provide confidence scores
                    start_pos=ent.start_char,
                    end_pos=ent.end_char,
                    context=self._extract_context(text, ent.start_char, ent.end_char),
                    language=language
                ))
                
        except Exception as e:
            self.logger.error(f"Error in spaCy fallback: {e}")
        
        return entities
    
    def analyze_article(self, article_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an article and add NER results.
        """
        text_to_analyze = ""
        
        # Combine title, content, and summary for analysis
        if article_data.get('title'):
            text_to_analyze += article_data['title'] + "\n"
        if article_data.get('content'):
            text_to_analyze += article_data['content'] + "\n"
        if article_data.get('summary'):
            text_to_analyze += article_data['summary']
        
        ner_analysis = self.extract_entities(text_to_analyze)
        
        # Add NER results to article data
        article_data['ner_analysis'] = {
            'people': ner_analysis.people,
            'organizations': ner_analysis.organizations,
            'locations': ner_analysis.locations,
            'geopolitical_entities': ner_analysis.geopolitical_entities,
            'military_terms': ner_analysis.military_terms,
            'confidence_score': ner_analysis.confidence_score,
            'language': ner_analysis.language,
            'entity_count': len(ner_analysis.entities)
        }
        
        return article_data
    
    def get_intelligence_summary(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate intelligence summary from multiple articles.
        """
        all_people = set()
        all_organizations = set()
        all_locations = set()
        all_geopolitical = set()
        all_military = set()
        
        for article in articles:
            if 'ner_analysis' in article:
                ner = article['ner_analysis']
                all_people.update(ner.get('people', []))
                all_organizations.update(ner.get('organizations', []))
                all_locations.update(ner.get('locations', []))
                all_geopolitical.update(ner.get('geopolitical_entities', []))
                all_military.update(ner.get('military_terms', []))
        
        return {
            'summary': {
                'total_people': len(all_people),
                'total_organizations': len(all_organizations),
                'total_locations': len(all_locations),
                'total_geopolitical_entities': len(all_geopolitical),
                'total_military_terms': len(all_military)
            },
            'entities': {
                'people': sorted(list(all_people)),
                'organizations': sorted(list(all_organizations)),
                'locations': sorted(list(all_locations)),
                'geopolitical_entities': sorted(list(all_geopolitical)),
                'military_terms': sorted(list(all_military))
            },
            'analysis_timestamp': datetime.now().isoformat()
        }


def main():
    """Test the NER system."""
    logging.basicConfig(level=logging.INFO)
    
    ner = IntelligenceNER()
    
    # Test with sample text
    test_text = """
    The Taiwan Strait has seen increased military activity as Chinese forces conducted exercises near Taipei. 
    Defense Minister Wang Yi met with US Secretary of State Antony Blinken to discuss regional security.
    """
    
    analysis = ner.extract_entities(test_text)
    
    print("NER Analysis Results:")
    print(f"Language: {analysis.language}")
    print(f"Confidence: {analysis.confidence_score:.3f}")
    print(f"People: {analysis.people}")
    print(f"Organizations: {analysis.organizations}")
    print(f"Locations: {analysis.locations}")
    print(f"Geopolitical: {analysis.geopolitical_entities}")
    print(f"Military: {analysis.military_terms}")


if __name__ == "__main__":
    main() 