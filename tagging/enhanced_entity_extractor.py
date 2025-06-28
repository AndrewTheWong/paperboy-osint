"""
Enhanced Entity Extractor for OSINT Articles
Extracts and normalizes entities with optional Wikidata linking
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc, Span
import requests
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """Structured entity representation"""
    entity: str
    entity_type: str
    linked_id: Optional[str] = None
    normalized_name: Optional[str] = None
    context_sentence: Optional[str] = None
    confidence_score: float = 0.8
    model_used: str = "spacy-en_core_web_sm"

class EnhancedEntityExtractor:
    """Enhanced entity extraction with comprehensive type support and Wikidata linking"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the entity extractor"""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found, downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Custom entity patterns for military/geopolitical context
        self.custom_patterns = self._load_custom_patterns()
        
        # Wikidata API configuration
        self.wikidata_base_url = "https://www.wikidata.org/w/api.php"
        
    def _load_custom_patterns(self) -> Dict[str, List[str]]:
        """Load custom entity patterns for military/geopolitical context"""
        return {
            'MILITARY_ORGS': [
                r'\b(?:PLA|People\'s Liberation Army|Chinese Military)\b',
                r'\b(?:PLA Navy|PLAN|Chinese Navy)\b',
                r'\b(?:PLA Air Force|PLAAF|Chinese Air Force)\b',
                r'\b(?:Eastern Theater Command|Western Theater Command|Southern Theater Command|Northern Theater Command|Central Theater Command)\b',
                r'\b(?:Taiwan Military|ROC Armed Forces|Taiwan Defense)\b',
                r'\b(?:US Military|US Armed Forces|Pentagon|Department of Defense)\b',
                r'\b(?:NATO|North Atlantic Treaty Organization)\b',
                r'\b(?:UN|United Nations)\b'
            ],
            'FACILITIES': [
                r'\b(?:military base|naval base|air base|missile site|port facility)\b',
                r'\b(?:Taipei|Beijing|Shanghai|Guangzhou|Shenzhen)\s+(?:Airport|Port|Base)\b',
                r'\b(?:Kinmen|Matsu|Penghu)\s+(?:Island|Base|Facility)\b',
                r'\b(?:South China Sea|East China Sea|Yellow Sea)\s+(?:Base|Facility)\b'
            ],
            'AIRSPACE': [
                r'\b(?:ADIZ|Air Defense Identification Zone)\b',
                r'\b(?:Taiwan Strait|median line|exclusive economic zone|EEZ)\b',
                r'\b(?:no-fly zone|restricted airspace|military airspace)\b'
            ],
            'MARITIME_ZONE': [
                r'\b(?:Taiwan Strait|Bashi Channel|Luzon Strait)\b',
                r'\b(?:South China Sea|East China Sea|Yellow Sea)\b',
                r'\b(?:exclusive economic zone|EEZ|territorial waters)\b',
                r'\b(?:continental shelf|fishing zone|patrol zone)\b'
            ],
            'EQUIPMENT': [
                r'\b(?:J-20|J-16|J-15|J-10|Su-35|F-16|F-35)\b',
                r'\b(?:Type 052D|Type 055|Type 054A|Type 056)\s+(?:destroyer|frigate|corvette)\b',
                r'\b(?:YJ-62|YJ-83|YJ-12|YJ-18)\s+(?:missile|anti-ship missile)\b',
                r'\b(?:HIMARS|Patriot|THAAD|Iron Dome)\b',
                r'\b(?:satellite|reconnaissance|surveillance)\s+(?:system|platform)\b'
            ],
            'PLATFORM': [
                r'\b(?:aircraft carrier|destroyer|frigate|submarine|patrol boat)\b',
                r'\b(?:fighter jet|bomber|transport aircraft|helicopter)\b',
                r'\b(?:satellite|drone|UAV|unmanned aerial vehicle)\b'
            ],
            'CYBERTOOL': [
                r'\b(?:malware|virus|trojan|ransomware|cyber attack)\b',
                r'\b(?:hacking|phishing|social engineering|cyber espionage)\b',
                r'\b(?:firewall|antivirus|security software|cyber defense)\b'
            ]
        }
    
    def extract_entities(self, text: str) -> List[Entity]:
        """Extract all entities from text with enhanced classification"""
        if not text:
            return []
        
        entities = []
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract standard NER entities
            entities.extend(self._extract_standard_entities(doc))
            
            # Extract custom pattern entities
            entities.extend(self._extract_custom_entities(text))
            
            # Extract datetime entities
            entities.extend(self._extract_datetime_entities(text))
            
            # Normalize and deduplicate
            entities = self._normalize_entities(entities)
            
            # Optional: Link to Wikidata
            entities = self._link_to_wikidata(entities)
            
            logger.info(f"Extracted {len(entities)} entities from text")
            
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
        
        return entities
    
    def _extract_standard_entities(self, doc: Doc) -> List[Entity]:
        """Extract standard spaCy NER entities"""
        entities = []
        
        for ent in doc.ents:
            entity_type = self._map_spacy_label(ent.label_)
            if entity_type:
                entity = Entity(
                    entity=ent.text.strip(),
                    entity_type=entity_type,
                    context_sentence=self._get_context_sentence(doc, ent),
                    confidence_score=0.8
                )
                entities.append(entity)
        
        return entities
    
    def _map_spacy_label(self, label: str) -> Optional[str]:
        """Map spaCy labels to our entity types"""
        mapping = {
            'PERSON': 'PERSON',
            'ORG': 'ORG',
            'GPE': 'GPE',
            'LOC': 'LOCATION',
            'FAC': 'FACILITY',
            'PRODUCT': 'EQUIPMENT',
            'EVENT': 'EVENT'
        }
        return mapping.get(label)
    
    def _extract_custom_entities(self, text: str) -> List[Entity]:
        """Extract entities using custom patterns"""
        entities = []
        
        for entity_type, patterns in self.custom_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity = Entity(
                        entity=match.group(0),
                        entity_type=entity_type,
                        context_sentence=self._get_context_sentence_from_text(text, match.start(), match.end()),
                        confidence_score=0.9
                    )
                    entities.append(entity)
        
        return entities
    
    def _extract_datetime_entities(self, text: str) -> List[Entity]:
        """Extract datetime entities from text"""
        entities = []
        
        # Common datetime patterns
        datetime_patterns = [
            r'\b(?:today|yesterday|tomorrow|next week|last week)\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',  # MM/DD/YYYY or DD/MM/YYYY
            r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
            r'\b(?:in|on|at)\s+\d{1,2}:\d{2}\s*(?:AM|PM)?\b'  # Time references
        ]
        
        for pattern in datetime_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity = Entity(
                    entity=match.group(0),
                    entity_type='DATETIME',
                    context_sentence=self._get_context_sentence_from_text(text, match.start(), match.end()),
                    confidence_score=0.7
                )
                entities.append(entity)
        
        return entities
    
    def _get_context_sentence(self, doc: Doc, span: Span) -> str:
        """Get the sentence containing the entity"""
        try:
            # Find the sentence containing the entity
            for sent in doc.sents:
                if span.start >= sent.start and span.end <= sent.end:
                    return sent.text.strip()
        except:
            pass
        return ""
    
    def _get_context_sentence_from_text(self, text: str, start: int, end: int) -> str:
        """Get context sentence from text positions"""
        try:
            # Find sentence boundaries
            sentence_start = text.rfind('.', 0, start) + 1
            sentence_end = text.find('.', end)
            if sentence_end == -1:
                sentence_end = len(text)
            
            return text[sentence_start:sentence_end].strip()
        except:
            return ""
    
    def _normalize_entities(self, entities: List[Entity]) -> List[Entity]:
        """Normalize and deduplicate entities"""
        normalized = {}
        
        for entity in entities:
            # Normalize the entity name
            normalized_name = self._normalize_entity_name(entity.entity)
            
            # Create a key for deduplication
            key = (normalized_name.lower(), entity.entity_type)
            
            if key not in normalized:
                entity.normalized_name = normalized_name
                normalized[key] = entity
            else:
                # Keep the one with higher confidence
                if entity.confidence_score > normalized[key].confidence_score:
                    normalized[key] = entity
        
        return list(normalized.values())
    
    def _normalize_entity_name(self, name: str) -> str:
        """Normalize entity name for consistency"""
        # Remove extra whitespace
        name = re.sub(r'\s+', ' ', name.strip())
        
        # Common abbreviations and variations
        abbreviations = {
            'PLA': 'People\'s Liberation Army',
            'PLAN': 'PLA Navy',
            'PLAAF': 'PLA Air Force',
            'ROC': 'Republic of China',
            'PRC': 'People\'s Republic of China',
            'US': 'United States',
            'USA': 'United States',
            'UK': 'United Kingdom',
            'NATO': 'North Atlantic Treaty Organization',
            'UN': 'United Nations'
        }
        
        return abbreviations.get(name.upper(), name)
    
    def _link_to_wikidata(self, entities: List[Entity]) -> List[Entity]:
        """Link entities to Wikidata (optional)"""
        for entity in entities:
            try:
                # Only link certain entity types
                if entity.entity_type in ['PERSON', 'ORG', 'GPE', 'LOCATION']:
                    wikidata_id = self._search_wikidata(entity.normalized_name or entity.entity)
                    if wikidata_id:
                        entity.linked_id = wikidata_id
            except Exception as e:
                logger.debug(f"Wikidata linking failed for {entity.entity}: {e}")
        
        return entities
    
    def _search_wikidata(self, query: str) -> Optional[str]:
        """Search Wikidata for an entity"""
        try:
            params = {
                'action': 'wbsearchentities',
                'format': 'json',
                'language': 'en',
                'type': 'item',
                'search': query,
                'limit': 1
            }
            
            response = requests.get(self.wikidata_base_url, params=params, timeout=5)
            response.raise_for_status()
            
            data = response.json()
            if data.get('search'):
                return data['search'][0]['id']
                
        except Exception as e:
            logger.debug(f"Wikidata search failed: {e}")
        
        return None
    
    def get_entities_by_type(self, entities: List[Entity], entity_type: str) -> List[Entity]:
        """Get entities of a specific type"""
        return [e for e in entities if e.entity_type == entity_type]
    
    def get_entity_names(self, entities: List[Entity]) -> List[str]:
        """Get just the entity names"""
        return [e.entity for e in entities] 