#!/usr/bin/env python3
"""
Event Extraction Module

Extracts events from text including:
1. Event identification and classification
2. Event participants and roles
3. Temporal information
4. Location information
5. Event relations and causality
"""

import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum

try:
    import spacy
    from spacy.tokens import Doc, Token, Span
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False

try:
    import dateutil.parser
    HAS_DATEUTIL = True
except ImportError:
    HAS_DATEUTIL = False

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Event type enumeration."""
    CONFLICT = "conflict"
    DIPLOMATIC = "diplomatic"
    MILITARY = "military"
    ECONOMIC = "economic"
    POLITICAL = "political"
    NATURAL_DISASTER = "natural_disaster"
    CYBER = "cyber"
    TERRORIST = "terrorist"
    PROTEST = "protest"
    ELECTION = "election"
    TRADE = "trade"
    MEETING = "meeting"
    AGREEMENT = "agreement"
    OTHER = "other"

@dataclass
class EventParticipant:
    """Represents a participant in an event."""
    name: str
    role: str
    entity_type: str
    confidence: float

@dataclass
class Event:
    """Represents an extracted event."""
    event_type: EventType
    trigger_word: str
    description: str
    participants: List[EventParticipant]
    location: Optional[str] = None
    time: Optional[str] = None
    confidence: float = 0.0
    sentence_idx: int = 0
    start_idx: int = 0
    end_idx: int = 0
    context: str = ""
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_type': self.event_type.value,
            'trigger_word': self.trigger_word,
            'description': self.description,
            'participants': [
                {
                    'name': p.name,
                    'role': p.role,
                    'entity_type': p.entity_type,
                    'confidence': p.confidence
                } for p in self.participants
            ],
            'location': self.location,
            'time': self.time,
            'confidence': self.confidence,
            'sentence_idx': self.sentence_idx,
            'start_idx': self.start_idx,
            'end_idx': self.end_idx,
            'context': self.context,
            'keywords': self.keywords
        }

class EventExtractor:
    """Comprehensive event extraction system."""
    
    def __init__(self):
        """Initialize the event extractor."""
        self.nlp = None
        
        if HAS_SPACY:
            self._load_spacy_model()
        
        # Initialize event patterns and keywords
        self.event_patterns = self._build_event_patterns()
        self.trigger_words = self._build_trigger_words()
        self.participant_roles = self._build_participant_roles()
        self.temporal_patterns = self._build_temporal_patterns()
        
        logger.info("EventExtractor initialized")
    
    def _load_spacy_model(self):
        """Load spaCy model for NER and dependency parsing."""
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy model for event extraction")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using fallback methods")
            self.nlp = None
    
    def _build_event_patterns(self) -> Dict[EventType, List[Dict]]:
        """Build patterns for different event types."""
        return {
            EventType.CONFLICT: [
                {
                    'triggers': ['attack', 'strike', 'bomb', 'invasion', 'war', 'battle', 'fight', 'clash'],
                    'pattern': r'(?P<actor>\w+(?:\s+\w+)*)\s+(?P<trigger>attacked?|struck|bombed|invaded|fought)\s+(?P<target>\w+(?:\s+\w+)*)',
                    'roles': {'actor': 'attacker', 'target': 'victim'}
                },
                {
                    'triggers': ['ceasefire', 'truce', 'peace'],
                    'pattern': r'(?P<trigger>ceasefire|truce|peace)\s+(?:between|with)\s+(?P<parties>\w+(?:\s+\w+)*)',
                    'roles': {'parties': 'participant'}
                }
            ],
            EventType.DIPLOMATIC: [
                {
                    'triggers': ['meeting', 'summit', 'talks', 'negotiation', 'conference'],
                    'pattern': r'(?P<participants>\w+(?:\s+\w+)*)\s+(?P<trigger>met|meeting|summit|talks|negotiation)\s+(?:with\s+)?(?P<other_participants>\w+(?:\s+\w+)*)?',
                    'roles': {'participants': 'participant', 'other_participants': 'participant'}
                },
                {
                    'triggers': ['treaty', 'agreement', 'deal', 'accord'],
                    'pattern': r'(?P<parties>\w+(?:\s+\w+)*)\s+(?:signed\s+)?(?P<trigger>treaty|agreement|deal|accord)\s+(?:with\s+)?(?P<other_parties>\w+(?:\s+\w+)*)?',
                    'roles': {'parties': 'signatory', 'other_parties': 'signatory'}
                }
            ],
            EventType.MILITARY: [
                {
                    'triggers': ['deployment', 'exercise', 'maneuver', 'drill', 'patrol'],
                    'pattern': r'(?P<force>\w+(?:\s+\w+)*)\s+(?P<trigger>deployed|exercise|maneuver|drill|patrol)\s+(?:in\s+)?(?P<location>\w+(?:\s+\w+)*)?',
                    'roles': {'force': 'military_actor', 'location': 'location'}
                },
                {
                    'triggers': ['missile', 'test', 'launch'],
                    'pattern': r'(?P<actor>\w+(?:\s+\w+)*)\s+(?:launched|tested)\s+(?P<trigger>missile|rocket)\s*(?:in\s+(?P<location>\w+(?:\s+\w+)*))?',
                    'roles': {'actor': 'launcher', 'location': 'location'}
                }
            ],
            EventType.ECONOMIC: [
                {
                    'triggers': ['sanctions', 'embargo', 'tariff', 'trade war'],
                    'pattern': r'(?P<actor>\w+(?:\s+\w+)*)\s+(?:imposed\s+)?(?P<trigger>sanctions|embargo|tariffs?)\s+(?:on\s+)?(?P<target>\w+(?:\s+\w+)*)',
                    'roles': {'actor': 'sanctioner', 'target': 'sanctioned'}
                },
                {
                    'triggers': ['investment', 'trade', 'export', 'import'],
                    'pattern': r'(?P<actor>\w+(?:\s+\w+)*)\s+(?P<trigger>invested?|traded?|exported?|imported?)\s+(?:to\s+|from\s+|with\s+)?(?P<partner>\w+(?:\s+\w+)*)',
                    'roles': {'actor': 'trader', 'partner': 'trade_partner'}
                }
            ],
            EventType.POLITICAL: [
                {
                    'triggers': ['election', 'vote', 'referendum', 'campaign'],
                    'pattern': r'(?P<trigger>election|vote|referendum|campaign)\s+(?:in\s+)?(?P<location>\w+(?:\s+\w+)*)',
                    'roles': {'location': 'location'}
                },
                {
                    'triggers': ['resign', 'appointment', 'elected'],
                    'pattern': r'(?P<person>\w+(?:\s+\w+)*)\s+(?P<trigger>resigned?|appointed|elected)\s+(?:as\s+)?(?P<position>\w+(?:\s+\w+)*)?',
                    'roles': {'person': 'official', 'position': 'position'}
                }
            ],
            EventType.PROTEST: [
                {
                    'triggers': ['protest', 'demonstration', 'riot', 'unrest', 'rally'],
                    'pattern': r'(?P<trigger>protests?|demonstrations?|riots?|unrest|rallies?)\s+(?:in\s+)?(?P<location>\w+(?:\s+\w+)*)',
                    'roles': {'location': 'location'}
                }
            ],
            EventType.CYBER: [
                {
                    'triggers': ['cyberattack', 'hack', 'breach', 'malware'],
                    'pattern': r'(?P<actor>\w+(?:\s+\w+)*)\s+(?P<trigger>hacked|breached|cyberattack)\s+(?P<target>\w+(?:\s+\w+)*)',
                    'roles': {'actor': 'attacker', 'target': 'victim'}
                }
            ]
        }
    
    def _build_trigger_words(self) -> Dict[EventType, Set[str]]:
        """Build trigger words for each event type."""
        triggers = defaultdict(set)
        
        for event_type, patterns in self.event_patterns.items():
            for pattern_dict in patterns:
                triggers[event_type].update(pattern_dict['triggers'])
        
        return dict(triggers)
    
    def _build_participant_roles(self) -> Dict[str, str]:
        """Build mapping of participant roles."""
        return {
            'attacker': 'aggressive actor',
            'victim': 'target of action',
            'participant': 'general participant',
            'signatory': 'agreement signer',
            'military_actor': 'military force',
            'launcher': 'missile/weapon launcher',
            'sanctioner': 'imposing sanctions',
            'sanctioned': 'receiving sanctions',
            'trader': 'trade participant',
            'trade_partner': 'trade counterpart',
            'official': 'government official',
            'position': 'official position',
            'location': 'event location'
        }
    
    def _build_temporal_patterns(self) -> List[str]:
        """Build patterns for extracting temporal information."""
        return [
            r'\b(?:today|yesterday|tomorrow)\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b(?:last|this|next)\s+(?:week|month|year)\b',
            r'\b\d{1,2}\s+(?:days?|weeks?|months?|years?)\s+ago\b',
            r'\bin\s+\d{4}\b',
            r'\bsince\s+\d{4}\b'
        ]
    
    def extract_events(self, text: str, entities: Dict[str, List[str]] = None) -> List[Event]:
        """
        Extract events from text.
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            
        Returns:
            List of extracted events
        """
        events = []
        
        try:
            # Method 1: Pattern-based event extraction
            pattern_events = self._extract_pattern_based_events(text)
            events.extend(pattern_events)
            
            # Method 2: Trigger word based extraction
            trigger_events = self._extract_trigger_based_events(text)
            events.extend(trigger_events)
            
            # Method 3: NER-based event extraction (if spaCy available)
            if self.nlp:
                ner_events = self._extract_ner_based_events(text, entities)
                events.extend(ner_events)
            
            # Enhance events with temporal and location information
            events = self._enhance_events_with_context(events, text)
            
            # Remove duplicates and rank by confidence
            events = self._deduplicate_events(events)
            events.sort(key=lambda x: x.confidence, reverse=True)
            
        except Exception as e:
            logger.error(f"Error extracting events: {e}")
        
        return events
    
    def _extract_pattern_based_events(self, text: str) -> List[Event]:
        """Extract events using regex patterns."""
        events = []
        sentences = self._split_sentences(text)
        
        for sent_idx, sentence in enumerate(sentences):
            for event_type, patterns in self.event_patterns.items():
                for pattern_dict in patterns:
                    pattern = pattern_dict['pattern']
                    triggers = pattern_dict['triggers']
                    roles = pattern_dict['roles']
                    
                    # Check if any trigger words are present
                    if any(trigger in sentence.lower() for trigger in triggers):
                        matches = re.finditer(pattern, sentence, re.IGNORECASE)
                        for match in matches:
                            try:
                                event = self._create_event_from_match(
                                    match, event_type, sentence, sent_idx, roles
                                )
                                if event:
                                    events.append(event)
                            except Exception as e:
                                logger.debug(f"Error processing pattern match: {e}")
        
        return events
    
    def _extract_trigger_based_events(self, text: str) -> List[Event]:
        """Extract events based on trigger words."""
        events = []
        sentences = self._split_sentences(text)
        
        for sent_idx, sentence in enumerate(sentences):
            words = sentence.lower().split()
            
            for event_type, trigger_words in self.trigger_words.items():
                for trigger in trigger_words:
                    if trigger in words:
                        # Found a trigger, create a basic event
                        trigger_idx = sentence.lower().find(trigger)
                        event = Event(
                            event_type=event_type,
                            trigger_word=trigger,
                            description=sentence.strip(),
                            participants=[],
                            confidence=0.5,
                            sentence_idx=sent_idx,
                            start_idx=trigger_idx,
                            end_idx=trigger_idx + len(trigger),
                            context=sentence,
                            keywords=[trigger]
                        )
                        events.append(event)
        
        return events
    
    def _extract_ner_based_events(self, text: str, entities: Dict[str, List[str]] = None) -> List[Event]:
        """Extract events using named entity recognition."""
        events = []
        
        if not self.nlp:
            return events
        
        try:
            doc = self.nlp(text)
            
            for sent in doc.sents:
                # Look for event patterns in dependency parse
                for token in sent:
                    if token.pos_ == "VERB" and token.lemma_ in self._get_all_triggers():
                        event_type = self._get_event_type_from_trigger(token.lemma_)
                        if event_type:
                            participants = self._extract_participants_from_parse(token, sent)
                            
                            event = Event(
                                event_type=event_type,
                                trigger_word=token.text,
                                description=str(sent),
                                participants=participants,
                                confidence=0.7,
                                sentence_idx=0,
                                start_idx=token.idx,
                                end_idx=token.idx + len(token.text),
                                context=str(sent),
                                keywords=[token.lemma_]
                            )
                            events.append(event)
        
        except Exception as e:
            logger.error(f"Error in NER-based event extraction: {e}")
        
        return events
    
    def _create_event_from_match(self, match: re.Match, event_type: EventType, 
                                sentence: str, sent_idx: int, roles: Dict[str, str]) -> Optional[Event]:
        """Create an event from a regex match."""
        try:
            participants = []
            trigger_word = ""
            
            # Extract participants based on named groups
            for group_name, text in match.groupdict().items():
                if text and group_name in roles:
                    role = roles[group_name]
                    if group_name == 'trigger':
                        trigger_word = text
                    else:
                        participant = EventParticipant(
                            name=text.strip(),
                            role=role,
                            entity_type='ENTITY',
                            confidence=0.8
                        )
                        participants.append(participant)
            
            if not trigger_word:
                # Find trigger word in the match
                for trigger in self.trigger_words.get(event_type, []):
                    if trigger in match.group().lower():
                        trigger_word = trigger
                        break
            
            return Event(
                event_type=event_type,
                trigger_word=trigger_word,
                description=match.group(),
                participants=participants,
                confidence=0.8,
                sentence_idx=sent_idx,
                start_idx=match.start(),
                end_idx=match.end(),
                context=sentence,
                keywords=[trigger_word] if trigger_word else []
            )
        
        except Exception as e:
            logger.debug(f"Error creating event from match: {e}")
            return None
    
    def _extract_participants_from_parse(self, verb_token: Token, sentence: Span) -> List[EventParticipant]:
        """Extract participants from dependency parse."""
        participants = []
        
        try:
            # Find subjects
            for child in verb_token.children:
                if child.dep_ in ["nsubj", "nsubjpass"]:
                    participant = EventParticipant(
                        name=child.text,
                        role="subject",
                        entity_type=child.ent_type_ or "ENTITY",
                        confidence=0.7
                    )
                    participants.append(participant)
                
                # Find objects
                elif child.dep_ in ["dobj", "pobj"]:
                    participant = EventParticipant(
                        name=child.text,
                        role="object",
                        entity_type=child.ent_type_ or "ENTITY",
                        confidence=0.7
                    )
                    participants.append(participant)
        
        except Exception as e:
            logger.debug(f"Error extracting participants: {e}")
        
        return participants
    
    def _enhance_events_with_context(self, events: List[Event], text: str) -> List[Event]:
        """Enhance events with temporal and location information."""
        sentences = self._split_sentences(text)
        
        for event in events:
            try:
                # Extract temporal information
                if event.sentence_idx < len(sentences):
                    sentence = sentences[event.sentence_idx]
                    event.time = self._extract_temporal_info(sentence)
                    event.location = self._extract_location_info(sentence)
            except Exception as e:
                logger.debug(f"Error enhancing event context: {e}")
        
        return events
    
    def _extract_temporal_info(self, text: str) -> Optional[str]:
        """Extract temporal information from text."""
        for pattern in self.temporal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()
        
        return None
    
    def _extract_location_info(self, text: str) -> Optional[str]:
        """Extract location information from text."""
        if self.nlp:
            try:
                doc = self.nlp(text)
                for ent in doc.ents:
                    if ent.label_ in ['GPE', 'LOC']:
                        return ent.text
            except Exception as e:
                logger.debug(f"Error extracting location: {e}")
        
        return None
    
    def _get_all_triggers(self) -> Set[str]:
        """Get all trigger words."""
        all_triggers = set()
        for triggers in self.trigger_words.values():
            all_triggers.update(triggers)
        return all_triggers
    
    def _get_event_type_from_trigger(self, trigger: str) -> Optional[EventType]:
        """Get event type from trigger word."""
        for event_type, triggers in self.trigger_words.items():
            if trigger in triggers:
                return event_type
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
    
    def _deduplicate_events(self, events: List[Event]) -> List[Event]:
        """Remove duplicate events."""
        seen = set()
        unique_events = []
        
        for event in events:
            # Create a key for deduplication
            key = (
                event.event_type,
                event.trigger_word.lower(),
                event.sentence_idx
            )
            
            if key not in seen:
                seen.add(key)
                unique_events.append(event)
            else:
                # Keep the one with higher confidence
                for i, existing in enumerate(unique_events):
                    existing_key = (
                        existing.event_type,
                        existing.trigger_word.lower(),
                        existing.sentence_idx
                    )
                    if existing_key == key and event.confidence > existing.confidence:
                        unique_events[i] = event
                        break
        
        return unique_events
    
    def get_event_statistics(self, events: List[Event]) -> Dict[str, Any]:
        """Get statistics about extracted events."""
        stats = {
            'total_events': len(events),
            'event_types': defaultdict(int),
            'trigger_words': defaultdict(int),
            'participant_roles': defaultdict(int),
            'average_confidence': 0.0,
            'events_with_time': 0,
            'events_with_location': 0,
            'events_with_participants': 0
        }
        
        if not events:
            return dict(stats)
        
        for event in events:
            stats['event_types'][event.event_type.value] += 1
            stats['trigger_words'][event.trigger_word] += 1
            
            if event.time:
                stats['events_with_time'] += 1
            if event.location:
                stats['events_with_location'] += 1
            if event.participants:
                stats['events_with_participants'] += 1
                
                for participant in event.participants:
                    stats['participant_roles'][participant.role] += 1
        
        # Calculate average confidence
        stats['average_confidence'] = sum(e.confidence for e in events) / len(events)
        
        # Convert defaultdicts to regular dicts
        stats['event_types'] = dict(stats['event_types'])
        stats['trigger_words'] = dict(stats['trigger_words'])
        stats['participant_roles'] = dict(stats['participant_roles'])
        
        return stats
    
    def extract_event_relations(self, events: List[Event]) -> List[Tuple[Event, Event, str]]:
        """Extract relations between events."""
        relations = []
        
        for i, event1 in enumerate(events):
            for j, event2 in enumerate(events):
                if i != j:
                    relation_type = self._infer_event_relation(event1, event2)
                    if relation_type:
                        relations.append((event1, event2, relation_type))
        
        return relations
    
    def _infer_event_relation(self, event1: Event, event2: Event) -> Optional[str]:
        """Infer relation between two events."""
        # Temporal relations
        if event1.time and event2.time:
            # This would need more sophisticated temporal parsing
            if event1.sentence_idx < event2.sentence_idx:
                return "precedes"
            elif event1.sentence_idx > event2.sentence_idx:
                return "follows"
        
        # Causal relations
        if event1.event_type == EventType.CONFLICT and event2.event_type == EventType.DIPLOMATIC:
            return "causes"
        
        # Co-occurrence
        if event1.sentence_idx == event2.sentence_idx:
            return "co_occurs"
        
        return None 