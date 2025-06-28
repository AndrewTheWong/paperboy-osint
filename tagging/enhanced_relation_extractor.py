"""
Enhanced Relation Extractor for OSINT Articles
Extracts subject-predicate-object triples with military/geopolitical focus
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc, Token
from spacy.matcher import DependencyMatcher, Matcher
from spacy.util import filter_spans

logger = logging.getLogger(__name__)

@dataclass
class Relation:
    """Structured relation representation"""
    subject: str
    predicate: str
    object: str
    context_sentence: Optional[str] = None
    confidence_score: float = 0.8
    model_used: str = "custom-relation-extractor"

class EnhancedRelationExtractor:
    """Enhanced relation extraction with military/geopolitical focus"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the relation extractor"""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model for relation extraction: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found, downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Initialize matchers
        self.matcher = Matcher(self.nlp.vocab)
        self.dep_matcher = DependencyMatcher(self.nlp.vocab)
        
        # Load relation patterns
        self._load_relation_patterns()
        
        # Military/geopolitical predicates
        self.predicates = {
            'deployed_to': ['deployed', 'sent', 'moved', 'transferred', 'dispatched'],
            'conducted': ['conducted', 'carried out', 'performed', 'executed', 'undertook'],
            'met_with': ['met', 'met with', 'held talks', 'discussed', 'consulted'],
            'threatened': ['threatened', 'warned', 'cautioned', 'admonished', 'intimidated'],
            'violated': ['violated', 'breached', 'crossed', 'entered', 'infringed'],
            'entered': ['entered', 'crossed', 'penetrated', 'invaded', 'accessed'],
            'announced': ['announced', 'declared', 'stated', 'revealed', 'disclosed'],
            'tracked_by': ['tracked', 'monitored', 'followed', 'observed', 'surveilled'],
            'detected_by': ['detected', 'spotted', 'identified', 'found', 'discovered'],
            'exercised_in': ['exercised', 'trained', 'drilled', 'practiced', 'maneuvered'],
            'patrolled': ['patrolled', 'guarded', 'protected', 'secured', 'watched'],
            'fired_at': ['fired', 'shot', 'launched', 'discharged', 'released'],
            'blocked': ['blocked', 'obstructed', 'prevented', 'stopped', 'hindered'],
            'supported': ['supported', 'backed', 'aided', 'assisted', 'helped'],
            'opposed': ['opposed', 'resisted', 'fought', 'challenged', 'defied']
        }
    
    def _load_relation_patterns(self):
        """Load relation extraction patterns"""
        # Pattern 1: Subject + Verb + Object
        pattern1 = [
            {"POS": "PROPN", "OP": "+"},  # Subject (proper noun)
            {"POS": "VERB"},              # Verb
            {"POS": "PROPN", "OP": "+"}   # Object (proper noun)
        ]
        
        # Pattern 2: Organization + Action + Location
        pattern2 = [
            {"LOWER": {"IN": ["pla", "china", "taiwan", "us", "military"]}},
            {"POS": "VERB"},
            {"POS": "PROPN", "OP": "+"}
        ]
        
        # Pattern 3: Military action patterns
        pattern3 = [
            {"LOWER": {"IN": ["deployed", "sent", "moved"]}},
            {"LOWER": "to"},
            {"POS": "PROPN", "OP": "+"}
        ]
        
        pattern4 = [
            {"LOWER": {"IN": ["conducted", "carried", "performed"]}},
            {"LOWER": {"IN": ["exercise", "drill", "training", "operation"]}},
            {"LOWER": "in"},
            {"POS": "PROPN", "OP": "+"}
        ]
        
        # Add patterns to matcher
        self.matcher.add("SUBJECT_VERB_OBJECT", [pattern1])
        self.matcher.add("ORG_ACTION_LOCATION", [pattern2])
        self.matcher.add("DEPLOYMENT", [pattern3])
        self.matcher.add("EXERCISE", [pattern4])
    
    def extract_relations(self, text: str, entities: List = None) -> List[Relation]:
        """Extract relations from text"""
        if not text:
            return []
        
        relations = []
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract relations using multiple methods
            relations.extend(self._extract_pattern_relations(doc))
            relations.extend(self._extract_dependency_relations(doc))
            relations.extend(self._extract_rule_based_relations(doc))
            
            # Filter and validate relations
            relations = self._filter_relations(relations)
            
            # Add context sentences
            relations = self._add_context_sentences(relations, doc)
            
            logger.info(f"Extracted {len(relations)} relations from text")
            
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
        
        return relations
    
    def _extract_pattern_relations(self, doc: Doc) -> List[Relation]:
        """Extract relations using pattern matching"""
        relations = []
        
        matches = self.matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            
            # Extract subject, predicate, object from the span
            relation = self._extract_triple_from_span(span)
            if relation:
                relations.append(relation)
        
        return relations
    
    def _extract_dependency_relations(self, doc: Doc) -> List[Relation]:
        """Extract relations using dependency parsing"""
        relations = []
        
        for token in doc:
            if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp"]:
                # Find subject and object
                subject = self._find_subject(token)
                obj = self._find_object(token)
                
                if subject and obj:
                    predicate = self._normalize_predicate(token.text)
                    relation = Relation(
                        subject=subject,
                        predicate=predicate,
                        object=obj,
                        confidence_score=0.7
                    )
                    relations.append(relation)
        
        return relations
    
    def _extract_rule_based_relations(self, doc: Doc) -> List[Relation]:
        """Extract relations using rule-based patterns"""
        relations = []
        
        # Military deployment patterns
        deployment_patterns = [
            r'(\w+(?:\s+\w+)*)\s+(?:was\s+)?deployed\s+to\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+(?:sent|moved|transferred)\s+(?:to|into)\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+conducted\s+(?:exercise|drill|training)\s+in\s+(\w+(?:\s+\w+)*)'
        ]
        
        text = doc.text
        for pattern in deployment_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                
                # Determine predicate from pattern
                if 'deployed' in match.group(0):
                    predicate = 'deployed_to'
                elif 'conducted' in match.group(0):
                    predicate = 'conducted'
                else:
                    predicate = 'moved_to'
                
                relation = Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence_score=0.8
                )
                relations.append(relation)
        
        # Diplomatic patterns
        diplomatic_patterns = [
            r'(\w+(?:\s+\w+)*)\s+met\s+with\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+held\s+talks\s+with\s+(\w+(?:\s+\w+)*)',
            r'(\w+(?:\s+\w+)*)\s+announced\s+(\w+(?:\s+\w+)*)'
        ]
        
        for pattern in diplomatic_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                subject = match.group(1).strip()
                obj = match.group(2).strip()
                
                if 'met' in match.group(0):
                    predicate = 'met_with'
                elif 'announced' in match.group(0):
                    predicate = 'announced'
                else:
                    predicate = 'held_talks_with'
                
                relation = Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence_score=0.7
                )
                relations.append(relation)
        
        return relations
    
    def _extract_triple_from_span(self, span) -> Optional[Relation]:
        """Extract subject-predicate-object from a matched span"""
        try:
            # Simple extraction: assume first noun is subject, verb is predicate, last noun is object
            tokens = list(span)
            
            subject = None
            predicate = None
            obj = None
            
            for i, token in enumerate(tokens):
                if token.pos_ in ["PROPN", "NOUN"] and not subject:
                    subject = token.text
                elif token.pos_ == "VERB" and not predicate:
                    predicate = self._normalize_predicate(token.text)
                elif token.pos_ in ["PROPN", "NOUN"] and predicate:
                    obj = token.text
            
            if subject and predicate and obj:
                return Relation(
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    confidence_score=0.6
                )
        
        except Exception as e:
            logger.debug(f"Error extracting triple from span: {e}")
        
        return None
    
    def _find_subject(self, verb_token: Token) -> Optional[str]:
        """Find the subject of a verb"""
        for child in verb_token.children:
            if child.dep_ in ["nsubj", "nsubjpass"]:
                return child.text
        return None
    
    def _find_object(self, verb_token: Token) -> Optional[str]:
        """Find the object of a verb"""
        for child in verb_token.children:
            if child.dep_ in ["dobj", "pobj"]:
                return child.text
        return None
    
    def _normalize_predicate(self, verb: str) -> str:
        """Normalize verb to standard predicate"""
        verb_lower = verb.lower()
        
        for predicate, variations in self.predicates.items():
            if verb_lower in variations:
                return predicate
        
        # Default to verb form
        return verb_lower
    
    def _filter_relations(self, relations: List[Relation]) -> List[Relation]:
        """Filter and validate relations"""
        filtered = []
        
        for relation in relations:
            # Basic validation
            if (len(relation.subject) > 2 and 
                len(relation.object) > 2 and 
                relation.subject != relation.object):
                
                # Remove common stop words
                if not self._is_stop_relation(relation):
                    filtered.append(relation)
        
        return filtered
    
    def _is_stop_relation(self, relation: Relation) -> bool:
        """Check if relation should be filtered out"""
        stop_subjects = ['it', 'this', 'that', 'there', 'here']
        stop_objects = ['it', 'this', 'that', 'there', 'here']
        
        return (relation.subject.lower() in stop_subjects or 
                relation.object.lower() in stop_objects)
    
    def _add_context_sentences(self, relations: List[Relation], doc: Doc) -> List[Relation]:
        """Add context sentences to relations"""
        for relation in relations:
            # Find sentences containing both subject and object
            for sent in doc.sents:
                sent_text = sent.text.lower()
                if (relation.subject.lower() in sent_text and 
                    relation.object.lower() in sent_text):
                    relation.context_sentence = sent.text.strip()
                    break
        
        return relations
    
    def get_relations_by_predicate(self, relations: List[Relation], predicate: str) -> List[Relation]:
        """Get relations with a specific predicate"""
        return [r for r in relations if r.predicate == predicate]
    
    def get_relations_involving_entity(self, relations: List[Relation], entity: str) -> List[Relation]:
        """Get relations involving a specific entity"""
        entity_lower = entity.lower()
        return [r for r in relations 
                if entity_lower in r.subject.lower() or entity_lower in r.object.lower()] 