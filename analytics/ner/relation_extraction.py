#!/usr/bin/env python3
"""
Relation Extraction Module

Extracts relationships between named entities using multiple approaches:
1. Dependency parsing with spaCy
2. Pattern-based extraction
3. Transformer-based models (when available)
4. Geopolitical relation patterns
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict

try:
    import spacy
    from spacy.tokens import Doc, Token, Span
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """Represents an extracted relation between entities."""
    subject: str
    subject_type: str
    relation_type: str
    object: str
    object_type: str
    confidence: float
    context: str
    sentence_idx: int = 0
    start_idx: int = 0
    end_idx: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'subject': self.subject,
            'subject_type': self.subject_type,
            'relation_type': self.relation_type,
            'object': self.object,
            'object_type': self.object_type,
            'confidence': self.confidence,
            'context': self.context,
            'sentence_idx': self.sentence_idx,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx
        }

class RelationExtractor:
    """Comprehensive relation extraction system."""
    
    def __init__(self):
        """Initialize the relation extractor."""
        self.nlp = None
        self.relation_classifier = None
        
        if HAS_SPACY:
            self._load_spacy_model()
        
        # Initialize relation patterns
        self.relation_patterns = self._build_relation_patterns()
        self.geopolitical_patterns = self._build_geopolitical_patterns()
        
        # Relation type mappings
        self.relation_types = self._build_relation_types()
        
        logger.info("RelationExtractor initialized")
    
    def _load_spacy_model(self):
        """Load spaCy model for dependency parsing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model for relation extraction")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using fallback methods")
            self.nlp = None
    
    def _build_relation_patterns(self) -> Dict[str, List[Dict]]:
        """Build regex patterns for relation extraction."""
        return {
            'location_relations': [
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:is\s+)?(?:located\s+)?in\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'located_in',
                    'confidence': 0.8
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:from|of)\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'from_location',
                    'confidence': 0.7
                },
                {
                    'pattern': r'(?P<obj>\w+(?:\s+\w+)*)\s+(?:capital|city)\s+(?P<subj>\w+(?:\s+\w+)*)',
                    'relation': 'capital_of',
                    'confidence': 0.9
                }
            ],
            'political_relations': [
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:president|leader|prime\s+minister|minister)\s+of\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'leads',
                    'confidence': 0.9
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:allied|alliance)\s+with\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'allied_with',
                    'confidence': 0.8
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:against|opposes|opposed\s+to)\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'opposes',
                    'confidence': 0.8
                }
            ],
            'conflict_relations': [
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:attacked|strikes|bombed|invaded)\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'attacks',
                    'confidence': 0.9
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:war|conflict|fighting)\s+(?:with|against)\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'in_conflict_with',
                    'confidence': 0.8
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:sanctions|sanctioned)\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'sanctions',
                    'confidence': 0.8
                }
            ],
            'diplomatic_relations': [
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:met|meeting|talks)\s+with\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'meets_with',
                    'confidence': 0.7
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:treaty|agreement|deal)\s+with\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'has_agreement_with',
                    'confidence': 0.8
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:embassy|ambassador)\s+(?:in|to)\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'diplomatic_presence_in',
                    'confidence': 0.8
                }
            ],
            'economic_relations': [
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:trade|trading|exports?)\s+(?:with|to)\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'trades_with',
                    'confidence': 0.8
                },
                {
                    'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:investment|invested|investing)\s+in\s+(?P<obj>\w+(?:\s+\w+)*)',
                    'relation': 'invests_in',
                    'confidence': 0.8
                }
            ]
        }
    
    def _build_geopolitical_patterns(self) -> List[Dict]:
        """Build patterns specifically for geopolitical relations."""
        return [
            {
                'keywords': ['tensions', 'tension', 'dispute', 'disagreement'],
                'relation': 'has_tensions_with',
                'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:tensions?|dispute|disagreement)\s+(?:with|between)\s+(?P<obj>\w+(?:\s+\w+)*)',
                'confidence': 0.8
            },
            {
                'keywords': ['cooperation', 'collaborate', 'partnership'],
                'relation': 'cooperates_with',
                'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:cooperation|collaborate|partnership)\s+with\s+(?P<obj>\w+(?:\s+\w+)*)',
                'confidence': 0.8
            },
            {
                'keywords': ['border', 'boundary', 'frontier'],
                'relation': 'borders',
                'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:border|boundary|frontier)\s+(?:with|of)\s+(?P<obj>\w+(?:\s+\w+)*)',
                'confidence': 0.9
            },
            {
                'keywords': ['support', 'supports', 'backing'],
                'relation': 'supports',
                'pattern': r'(?P<subj>\w+(?:\s+\w+)*)\s+(?:supports?|backing)\s+(?P<obj>\w+(?:\s+\w+)*)',
                'confidence': 0.8
            }
        ]
    
    def _build_relation_types(self) -> Dict[str, str]:
        """Build mapping of relation types to categories."""
        return {
            'located_in': 'geographic',
            'from_location': 'geographic',
            'capital_of': 'geographic',
            'borders': 'geographic',
            'leads': 'political',
            'allied_with': 'political',
            'opposes': 'political',
            'supports': 'political',
            'attacks': 'conflict',
            'in_conflict_with': 'conflict',
            'sanctions': 'conflict',
            'has_tensions_with': 'conflict',
            'meets_with': 'diplomatic',
            'has_agreement_with': 'diplomatic',
            'diplomatic_presence_in': 'diplomatic',
            'cooperates_with': 'diplomatic',
            'trades_with': 'economic',
            'invests_in': 'economic'
        }
    
    def extract_relations(self, text: str, entities: Dict[str, List[str]] = None) -> List[Relation]:
        """
        Extract relations from text using multiple approaches.
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            
        Returns:
            List of extracted relations
        """
        relations = []
        
        try:
            # Method 1: Pattern-based extraction
            pattern_relations = self._extract_pattern_based_relations(text)
            relations.extend(pattern_relations)
            
            # Method 2: Dependency parsing (if spaCy available)
            if self.nlp:
                dependency_relations = self._extract_dependency_relations(text, entities)
                relations.extend(dependency_relations)
            
            # Method 3: Geopolitical patterns
            geo_relations = self._extract_geopolitical_relations(text)
            relations.extend(geo_relations)
            
            # Remove duplicates and rank by confidence
            relations = self._deduplicate_relations(relations)
            relations.sort(key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
        
        return relations
    
    def _extract_pattern_based_relations(self, text: str) -> List[Relation]:
        """Extract relations using regex patterns."""
        relations = []
        sentences = self._split_sentences(text)
        
        for sent_idx, sentence in enumerate(sentences):
            for category, patterns in self.relation_patterns.items():
                for pattern_dict in patterns:
                    pattern = pattern_dict['pattern']
                    relation_type = pattern_dict['relation']
                    confidence = pattern_dict['confidence']
                    
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        try:
                            subject = match.group('subj').strip()
                            obj = match.group('obj').strip()
                            
                            if subject and obj and subject.lower() != obj.lower():
                                relation = Relation(
                                    subject=subject,
                                    subject_type='ENTITY',
                                    relation_type=relation_type,
                                    object=obj,
                                    object_type='ENTITY',
                                    confidence=confidence,
                                    context=sentence,
                                    sentence_idx=sent_idx,
                                    start_idx=match.start(),
                                    end_idx=match.end()
                                )
                                relations.append(relation)
                        except Exception as e:
                            logger.debug(f"Error processing pattern match: {e}")
        
        return relations
    
    def _extract_dependency_relations(self, text: str, entities: Dict[str, List[str]] = None) -> List[Relation]:
        """Extract relations using dependency parsing."""
        relations = []
        
        if not self.nlp:
            return relations
        
        try:
            doc = self.nlp(text)
            
            for sent in doc.sents:
                # Find subject-verb-object patterns
                for token in sent:
                    if token.dep_ == "nsubj" and token.head.pos_ == "VERB":
                        verb = token.head
                        subject = token
                        
                        # Find objects
                        for child in verb.children:
                            if child.dep_ in ["dobj", "pobj"]:
                                obj = child
                                
                                # Extract relation
                                relation = self._create_dependency_relation(
                                    subject, verb, obj, str(sent), 0
                                )
                                if relation:
                                    relations.append(relation)
                
                # Find named entity relations
                entities_in_sent = [(ent.text, ent.label_, ent.start, ent.end) for ent in sent.ents]
                for i, (ent1_text, ent1_label, ent1_start, ent1_end) in enumerate(entities_in_sent):
                    for j, (ent2_text, ent2_label, ent2_start, ent2_end) in enumerate(entities_in_sent):
                        if i != j:
                            relation = self._extract_entity_relation(
                                ent1_text, ent1_label, ent2_text, ent2_label, 
                                str(sent), ent1_start, ent2_end
                            )
                            if relation:
                                relations.append(relation)
        
        except Exception as e:
            logger.error(f"Error in dependency parsing: {e}")
        
        return relations
    
    def _extract_geopolitical_relations(self, text: str) -> List[Relation]:
        """Extract geopolitical relations using specialized patterns."""
        relations = []
        sentences = self._split_sentences(text)
        
        for sent_idx, sentence in enumerate(sentences):
            for pattern_dict in self.geopolitical_patterns:
                keywords = pattern_dict['keywords']
                relation_type = pattern_dict['relation']
                pattern = pattern_dict['pattern']
                confidence = pattern_dict['confidence']
                
                # Check if any keywords are present
                if any(keyword.lower() in sentence.lower() for keyword in keywords):
                    matches = re.finditer(pattern, sentence, re.IGNORECASE)
                    for match in matches:
                        try:
                            subject = match.group('subj').strip()
                            obj = match.group('obj').strip()
                            
                            if subject and obj and subject.lower() != obj.lower():
                                relation = Relation(
                                    subject=subject,
                                    subject_type='GEOPOLITICAL',
                                    relation_type=relation_type,
                                    object=obj,
                                    object_type='GEOPOLITICAL',
                                    confidence=confidence,
                                    context=sentence,
                                    sentence_idx=sent_idx,
                                    start_idx=match.start(),
                                    end_idx=match.end()
                                )
                                relations.append(relation)
                        except Exception as e:
                            logger.debug(f"Error processing geopolitical pattern: {e}")
        
        return relations
    
    def _create_dependency_relation(self, subject: Token, verb: Token, obj: Token, 
                                  context: str, sent_idx: int) -> Optional[Relation]:
        """Create relation from dependency parse results."""
        try:
            # Map verb to relation type
            verb_text = verb.lemma_.lower()
            relation_type = self._map_verb_to_relation(verb_text)
            
            if relation_type:
                return Relation(
                    subject=subject.text,
                    subject_type=subject.ent_type_ or 'ENTITY',
                    relation_type=relation_type,
                    object=obj.text,
                    object_type=obj.ent_type_ or 'ENTITY',
                    confidence=0.7,
                    context=context,
                    sentence_idx=sent_idx,
                    start_idx=subject.idx,
                    end_idx=obj.idx + len(obj.text)
                )
        except Exception as e:
            logger.debug(f"Error creating dependency relation: {e}")
        
        return None
    
    def _extract_entity_relation(self, ent1_text: str, ent1_label: str, ent2_text: str, 
                               ent2_label: str, context: str, start_idx: int, 
                               end_idx: int) -> Optional[Relation]:
        """Extract relation between two named entities."""
        try:
            # Determine relation type based on entity types
            relation_type = self._infer_relation_from_entities(ent1_label, ent2_label, context)
            
            if relation_type:
                return Relation(
                    subject=ent1_text,
                    subject_type=ent1_label,
                    relation_type=relation_type,
                    object=ent2_text,
                    object_type=ent2_label,
                    confidence=0.6,
                    context=context,
                    sentence_idx=0,
                    start_idx=start_idx,
                    end_idx=end_idx
                )
        except Exception as e:
            logger.debug(f"Error extracting entity relation: {e}")
        
        return None
    
    def _map_verb_to_relation(self, verb: str) -> Optional[str]:
        """Map verb to relation type."""
        verb_mappings = {
            'attack': 'attacks',
            'meet': 'meets_with',
            'support': 'supports',
            'oppose': 'opposes',
            'trade': 'trades_with',
            'invest': 'invests_in',
            'sanction': 'sanctions',
            'ally': 'allied_with',
            'cooperate': 'cooperates_with',
            'border': 'borders'
        }
        return verb_mappings.get(verb)
    
    def _infer_relation_from_entities(self, ent1_type: str, ent2_type: str, context: str) -> Optional[str]:
        """Infer relation type from entity types and context."""
        context_lower = context.lower()
        
        # Geographic relations
        if ent1_type in ['GPE', 'LOC'] and ent2_type in ['GPE', 'LOC']:
            if any(word in context_lower for word in ['border', 'boundary']):
                return 'borders'
            elif any(word in context_lower for word in ['capital', 'city']):
                return 'capital_of'
            elif 'in' in context_lower:
                return 'located_in'
        
        # Political relations
        if ent1_type == 'PERSON' and ent2_type == 'GPE':
            if any(word in context_lower for word in ['president', 'leader', 'minister']):
                return 'leads'
        
        # Conflict relations
        if any(word in context_lower for word in ['attack', 'war', 'conflict']):
            return 'in_conflict_with'
        
        return None
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        if self.nlp:
            doc = self.nlp(text)
            return [sent.text.strip() for sent in doc.sents]
        else:
            # Simple sentence splitting fallback
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if s.strip()]
    
    def _deduplicate_relations(self, relations: List[Relation]) -> List[Relation]:
        """Remove duplicate relations."""
        seen = set()
        unique_relations = []
        
        for relation in relations:
            # Create a key for deduplication
            key = (
                relation.subject.lower(),
                relation.relation_type,
                relation.object.lower()
            )
            
            if key not in seen:
                seen.add(key)
                unique_relations.append(relation)
            else:
                # Keep the one with higher confidence
                for i, existing in enumerate(unique_relations):
                    existing_key = (
                        existing.subject.lower(),
                        existing.relation_type,
                        existing.object.lower()
                    )
                    if existing_key == key and relation.confidence > existing.confidence:
                        unique_relations[i] = relation
                        break
        
        return unique_relations
    
    def get_relation_types(self) -> Dict[str, str]:
        """Get all available relation types and their categories."""
        return self.relation_types.copy()
    
    def get_relation_statistics(self, relations: List[Relation]) -> Dict[str, Any]:
        """Get statistics about extracted relations."""
        stats = {
            'total_relations': len(relations),
            'relation_types': defaultdict(int),
            'relation_categories': defaultdict(int),
            'average_confidence': 0.0,
            'entity_types': defaultdict(int)
        }
        
        if not relations:
            return dict(stats)
        
        for relation in relations:
            stats['relation_types'][relation.relation_type] += 1
            stats['entity_types'][f"{relation.subject_type}-{relation.object_type}"] += 1
            
            # Get category
            category = self.relation_types.get(relation.relation_type, 'other')
            stats['relation_categories'][category] += 1
        
        # Calculate average confidence
        stats['average_confidence'] = sum(r.confidence for r in relations) / len(relations)
        
        # Convert defaultdicts to regular dicts
        stats['relation_types'] = dict(stats['relation_types'])
        stats['relation_categories'] = dict(stats['relation_categories'])
        stats['entity_types'] = dict(stats['entity_types'])
        
        return stats 