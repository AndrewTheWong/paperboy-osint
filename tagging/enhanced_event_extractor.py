"""
Enhanced Event Extractor for OSINT Articles
Classifies events according to military/geopolitical schema
"""

import re
import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import spacy
from spacy.tokens import Doc
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class Event:
    """Structured event representation with comprehensive details"""
    event_type: str
    participants: List[str]
    description: str = ""  # Detailed description of the event
    escalation_score: float = 0.0  # Escalation score from 0-1
    escalation_analysis: str = ""  # Analysis of escalation factors
    location: Optional[str] = None
    datetime: Optional[str] = None
    severity_rating: str = "medium"
    confidence_score: float = 0.8
    model_used: str = "enhanced-event-classifier"
    context_sentence: Optional[str] = None
    keywords: List[str] = None
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        
        # Generate description if not provided
        if not self.description:
            self.description = self._generate_description()
        
        # Calculate escalation score if not provided
        if self.escalation_score == 0.0:
            self.escalation_score = self._calculate_escalation_score()
        
        # Generate escalation analysis if not provided
        if not self.escalation_analysis:
            self.escalation_analysis = self._generate_escalation_analysis()
    
    def _generate_description(self) -> str:
        """Generate a detailed description of the event"""
        description_parts = []
        
        # Start with event type
        event_type_desc = {
            'military_movement': 'Military forces deployment or repositioning',
            'live_fire_exercise': 'Live ammunition military training exercise',
            'gray_zone_operation': 'Non-military coercive operation',
            'airspace_intrusion': 'Unauthorized airspace entry or violation',
            'naval_patrol': 'Naval vessel patrol or transit operation',
            'diplomatic_statement': 'Official diplomatic communication or statement',
            'summit_or_meeting': 'High-level diplomatic meeting or summit',
            'arms_sale': 'Military equipment or weapons transaction',
            'cyber_attack': 'Digital attack or cyber operation',
            'economic_signal': 'Economic pressure or trade action',
            'legislative_action': 'Legal or regulatory government action',
            'disinfo_campaign': 'Information warfare or propaganda operation',
            'incident_reported': 'Reported incident or unexpected event'
        }
        
        base_desc = event_type_desc.get(self.event_type, f"{self.event_type} event")
        description_parts.append(base_desc)
        
        # Add participants
        if self.participants:
            if len(self.participants) == 1:
                description_parts.append(f"involving {self.participants[0]}")
            else:
                description_parts.append(f"involving {', '.join(self.participants[:-1])} and {self.participants[-1]}")
        
        # Add location if available
        if self.location:
            description_parts.append(f"in {self.location}")
        
        # Add context from sentence
        if self.context_sentence:
            # Extract key details from context
            context_lower = self.context_sentence.lower()
            if any(word in context_lower for word in ['announced', 'reported', 'confirmed']):
                description_parts.append("as officially reported")
            elif any(word in context_lower for word in ['alleged', 'claimed', 'suspected']):
                description_parts.append("according to unconfirmed reports")
        
        return ". ".join(description_parts) + "."
    
    def _calculate_escalation_score(self) -> float:
        """Calculate escalation score based on event characteristics"""
        base_scores = {
            'military_movement': 0.6,
            'live_fire_exercise': 0.7,
            'gray_zone_operation': 0.5,
            'airspace_intrusion': 0.8,
            'naval_patrol': 0.4,
            'diplomatic_statement': 0.3,
            'summit_or_meeting': 0.2,
            'arms_sale': 0.5,
            'cyber_attack': 0.7,
            'economic_signal': 0.4,
            'legislative_action': 0.3,
            'disinfo_campaign': 0.4,
            'incident_reported': 0.6
        }
        
        score = base_scores.get(self.event_type, 0.5)
        
        # Adjust based on severity
        severity_multipliers = {
            'low': 0.7,
            'medium': 1.0,
            'high': 1.3,
            'critical': 1.6
        }
        score *= severity_multipliers.get(self.severity_rating, 1.0)
        
        # Adjust based on context keywords
        if self.context_sentence:
            context_lower = self.context_sentence.lower()
            
            # Escalating keywords
            escalating_words = ['threat', 'aggressive', 'hostile', 'provocative', 'dangerous', 'escalate']
            for word in escalating_words:
                if word in context_lower:
                    score += 0.1
            
            # De-escalating keywords
            deescalating_words = ['peaceful', 'routine', 'defensive', 'cooperation', 'dialogue']
            for word in deescalating_words:
                if word in context_lower:
                    score -= 0.1
        
        # Adjust based on participants
        high_risk_actors = ['military', 'army', 'navy', 'air force', 'missile', 'nuclear']
        if self.participants:
            for participant in self.participants:
                participant_lower = participant.lower()
                for risk_actor in high_risk_actors:
                    if risk_actor in participant_lower:
                        score += 0.05
        
        return max(0.0, min(1.0, score))
    
    def _generate_escalation_analysis(self) -> str:
        """Generate detailed escalation analysis"""
        analysis_parts = []
        
        # Score interpretation
        if self.escalation_score >= 0.8:
            analysis_parts.append("HIGH ESCALATION RISK: This event represents a significant escalatory action")
        elif self.escalation_score >= 0.6:
            analysis_parts.append("MODERATE-HIGH ESCALATION: This event has notable escalatory potential")
        elif self.escalation_score >= 0.4:
            analysis_parts.append("MODERATE ESCALATION: This event shows some escalatory characteristics")
        elif self.escalation_score >= 0.2:
            analysis_parts.append("LOW-MODERATE ESCALATION: This event has limited escalatory impact")
        else:
            analysis_parts.append("LOW ESCALATION: This event is unlikely to escalate tensions")
        
        # Event type specific analysis
        type_analysis = {
            'military_movement': "Military deployments can signal preparation for larger operations",
            'live_fire_exercise': "Live fire exercises demonstrate military capability and readiness",
            'gray_zone_operation': "Gray zone activities operate below the threshold of armed conflict",
            'airspace_intrusion': "Airspace violations represent direct sovereignty challenges",
            'naval_patrol': "Naval activities can control sea lanes and project power",
            'diplomatic_statement': "Diplomatic statements can signal policy changes or warnings",
            'arms_sale': "Arms transfers can alter regional military balance",
            'cyber_attack': "Cyber operations can disrupt critical infrastructure",
            'economic_signal': "Economic measures can impose costs and signal resolve"
        }
        
        if self.event_type in type_analysis:
            analysis_parts.append(type_analysis[self.event_type])
        
        # Severity-based analysis
        if self.severity_rating == 'critical':
            analysis_parts.append("The critical severity indicates immediate attention required")
        elif self.severity_rating == 'high':
            analysis_parts.append("High severity suggests significant regional impact")
        
        # Location-based analysis
        if self.location:
            location_lower = self.location.lower()
            if any(area in location_lower for area in ['strait', 'taiwan', 'south china sea']):
                analysis_parts.append("Location in strategic waterway increases geopolitical significance")
            elif any(area in location_lower for area in ['border', 'boundary', 'zone']):
                analysis_parts.append("Border area activity heightens territorial tensions")
        
        return ". ".join(analysis_parts) + "."
    
    def to_dict(self) -> Dict[str, any]:
        """Convert Event to dictionary format for JSON serialization"""
        return {
            'event_type': self.event_type,
            'participants': self.participants,
            'description': self.description,
            'escalation_score': self.escalation_score,
            'escalation_analysis': self.escalation_analysis,
            'location': self.location,
            'datetime': self.datetime,
            'severity_rating': self.severity_rating,
            'confidence_score': self.confidence_score,
            'model_used': self.model_used,
            'context_sentence': self.context_sentence,
            'keywords': self.keywords
        }

class EnhancedEventExtractor:
    """Enhanced event extraction with military/geopolitical classification"""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Initialize the event extractor"""
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model for event extraction: {model_name}")
        except OSError:
            logger.warning(f"Model {model_name} not found, downloading...")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Event classification patterns
        self.event_patterns = self._load_event_patterns()
        
        # Severity indicators
        self.severity_indicators = {
            'low': ['routine', 'normal', 'regular', 'standard', 'usual', 'scheduled'],
            'medium': ['significant', 'notable', 'important', 'concerning', 'unusual'],
            'high': ['serious', 'critical', 'urgent', 'alarming', 'threatening', 'major'],
            'critical': ['extreme', 'severe', 'dangerous', 'hostile', 'aggressive', 'unprecedented']
        }
        
        # Escalation keywords
        self.escalation_keywords = {
            'high': ['war', 'attack', 'invasion', 'strike', 'bomb', 'missile', 'nuclear', 'threat'],
            'medium': ['military', 'exercise', 'patrol', 'deployment', 'tension', 'dispute', 'warning'],
            'low': ['meeting', 'talks', 'cooperation', 'agreement', 'dialogue', 'peaceful']
        }
    
    def _load_event_patterns(self) -> Dict[str, List[str]]:
        """Load event classification patterns with enhanced descriptions"""
        return {
            'military_movement': [
                r'\b(?:deployed|sent|moved|transferred|dispatched)\s+(?:to|into|toward)\b',
                r'\b(?:troop|soldier|military|army|navy|air force)\s+(?:movement|deployment|transfer)\b',
                r'\b(?:military|defense|security)\s+(?:build-up|buildup|reinforcement)\b',
                r'\b(?:withdrew|pulled back|retreated|evacuated)\s+(?:forces|troops|military)\b',
                r'\b(?:positioning|repositioning|mobilizing)\s+(?:forces|units|troops)\b',
                # Added simpler patterns
                r'\bmilitary\s+(?:exercise|exercises|training|drill|drills)\b',
                r'\b(?:army|navy|air force|forces)\s+(?:exercise|exercises|training)\b'
            ],
            'live_fire_exercise': [
                r'\b(?:live fire|live-fire|live ammunition)\s+(?:exercise|drill|training)\b',
                r'\b(?:fired|launched|discharged|shot)\s+(?:missile|rocket|artillery|weapon)\b',
                r'\b(?:weapon|missile|artillery)\s+(?:test|firing|launch)\b',
                r'\b(?:military|naval|air)\s+(?:exercise|drill|training)\s+(?:with|using)\s+(?:live|real)\b',
                r'\b(?:combat|war)\s+(?:exercise|simulation|drill)\b',
                # Added simpler patterns
                r'\bconducted\s+(?:live.fire|military)\s+exercise',
                r'\b(?:exercise|exercises|training)\s+(?:involving|with)\s+(?:naval|military|aircraft)',
                r'\bmilitary\s+exercises?\s+(?:in|near|involving)'
            ],
            'gray_zone_operation': [
                r'\b(?:coast guard|fishing fleet|fishing vessel)\s+(?:operation|patrol|activity)\b',
                r'\b(?:civilian|commercial)\s+(?:vessel|ship|aircraft)\s+(?:in|near)\s+(?:disputed|contested)\b',
                r'\b(?:psychological|information|propaganda)\s+(?:operation|campaign|warfare)\b',
                r'\b(?:non-military|non-combat)\s+(?:pressure|coercion|intimidation)\b',
                r'\b(?:fishing|mining|survey)\s+(?:operation|activity)\s+(?:in|near)\s+(?:disputed)\b',
                r'\b(?:paramilitary|militia)\s+(?:activity|operation|deployment)\b'
            ],
            'airspace_intrusion': [
                r'\b(?:entered|crossed|violated|breached)\s+(?:ADIZ|air defense identification zone)\b',
                r'\b(?:aircraft|plane|jet|drone)\s+(?:entered|crossed|violated)\s+(?:airspace|zone)\b',
                r'\b(?:median line|boundary|border)\s+(?:crossing|violation|breach)\b',
                r'\b(?:no-fly zone|restricted airspace)\s+(?:violation|breach)\b',
                r'\b(?:fighter|bomber|reconnaissance)\s+(?:aircraft|plane)\s+(?:approached|entered)\b',
                # Added simpler patterns
                r'\baircraft\s+crossing\s+(?:the\s+)?median\s+line',
                r'\btracking\s+(?:multiple\s+)?(?:Chinese|military)\s+aircraft'
            ],
            'naval_patrol': [
                r'\b(?:naval|navy|ship|vessel)\s+(?:patrol|patrolling|sailing|transit)\b',
                r'\b(?:strait|channel|waterway)\s+(?:transit|passage|crossing)\b',
                r'\b(?:warship|destroyer|frigate|corvette)\s+(?:patrol|sail|transit)\b',
                r'\b(?:maritime|sea)\s+(?:patrol|surveillance|monitoring)\b',
                r'\b(?:freedom of navigation|FONOP)\s+(?:operation|mission)\b',
                # Added simpler patterns
                r'\bnaval\s+vessels?\s+(?:and|in|near)',
                r'\bvessel\s+(?:patrol|transit|operation)'
            ],
            'diplomatic_statement': [
                r'\b(?:diplomatic|foreign|international)\s+(?:statement|declaration|announcement)\b',
                r'\b(?:minister|official|spokesperson)\s+(?:said|stated|announced|declared)\b',
                r'\b(?:government|ministry|department)\s+(?:statement|response|comment)\b',
                r'\b(?:diplomatic|official)\s+(?:protest|complaint|objection)\b',
                r'\b(?:condemned|criticized|denounced|warned)\b',
                # Added simpler patterns
                r'\b(?:State Department|Ministry)\s+(?:issued|released)\s+(?:a\s+)?statement',
                r'\bcalling\s+for\s+(?:restraint|peaceful|resolution)',
                r'\bissued\s+a\s+statement\s+calling'
            ],
            'summit_or_meeting': [
                r'\b(?:summit|meeting|conference|talks|negotiation)\b',
                r'\b(?:diplomatic|bilateral|multilateral)\s+(?:meeting|talks|discussion)\b',
                r'\b(?:leaders|officials|representatives)\s+(?:met|held|attended)\b',
                r'\b(?:president|prime minister|minister)\s+(?:met|visited|held talks)\b',
                r'\b(?:high-level|senior)\s+(?:meeting|talks|consultation)\b'
            ],
            'arms_sale': [
                r'\b(?:arms|weapon|military)\s+(?:sale|deal|agreement|contract)\b',
                r'\b(?:sold|purchased|acquired|procured)\s+(?:weapon|missile|system|equipment)\b',
                r'\b(?:defense|military)\s+(?:contract|deal|agreement)\b',
                r'\b(?:weapon|missile|system)\s+(?:delivery|transfer|shipment)\b',
                r'\b(?:military aid|defense assistance)\b'
            ],
            'cyber_attack': [
                r'\b(?:cyber|digital|computer)\s+(?:attack|hack|breach|intrusion)\b',
                r'\b(?:malware|virus|trojan|ransomware)\s+(?:attack|infection)\b',
                r'\b(?:hacking|phishing|social engineering)\s+(?:attack|campaign)\b',
                r'\b(?:cyber|digital)\s+(?:espionage|surveillance|monitoring)\b',
                r'\b(?:data breach|system compromise)\b'
            ],
            'economic_signal': [
                r'\b(?:economic|trade|commercial)\s+(?:sanction|ban|restriction)\b',
                r'\b(?:import|export)\s+(?:ban|restriction|control)\b',
                r'\b(?:investment|trade|economic)\s+(?:agreement|deal|partnership)\b',
                r'\b(?:tariff|duty|tax)\s+(?:increase|decrease|change)\b',
                r'\b(?:economic pressure|trade war)\b'
            ],
            'legislative_action': [
                r'\b(?:law|legislation|bill|act)\s+(?:passed|enacted|approved)\b',
                r'\b(?:parliament|congress|legislature)\s+(?:vote|decision|resolution)\b',
                r'\b(?:legal|regulatory)\s+(?:change|reform|amendment)\b',
                r'\b(?:constitutional|legal)\s+(?:amendment|reform|change)\b',
                r'\b(?:policy|regulation)\s+(?:change|update|revision)\b'
            ],
            'disinfo_campaign': [
                r'\b(?:disinformation|misinformation|propaganda)\s+(?:campaign|operation)\b',
                r'\b(?:fake|false|misleading)\s+(?:news|information|claim)\b',
                r'\b(?:information|media)\s+(?:warfare|manipulation|campaign)\b',
                r'\b(?:social media|online)\s+(?:campaign|operation|manipulation)\b',
                r'\b(?:narrative|messaging)\s+(?:campaign|operation)\b'
            ],
            'incident_reported': [
                r'\b(?:incident|accident|crash|collision)\s+(?:reported|occurred|happened)\b',
                r'\b(?:UAV|drone|aircraft|ship)\s+(?:crash|accident|incident)\b',
                r'\b(?:radar|system|equipment)\s+(?:failure|malfunction|incident)\b',
                r'\b(?:unusual|unexpected|strange)\s+(?:event|incident|occurrence)\b',
                r'\b(?:emergency|alert|alarm)\s+(?:situation|condition)\b'
            ]
        }
    
    def extract_events(self, text: str, entities: List = None) -> List[Event]:
        """Extract and classify events from text with comprehensive analysis"""
        if not text:
            return []
        
        events = []
        
        try:
            # Process with spaCy
            doc = self.nlp(text)
            
            # Extract events using pattern matching
            pattern_events = self._extract_pattern_events(doc)
            events.extend(pattern_events)
            
            # Extract events using rule-based approach
            rule_events = self._extract_rule_based_events(doc)
            events.extend(rule_events)
            
            # Extract events using NER and dependency parsing
            if entities:
                ner_events = self._extract_ner_events(doc, entities)
                events.extend(ner_events)
            
            # Filter and validate events
            events = self._filter_events(events)
            
            # Add comprehensive metadata
            events = self._add_comprehensive_metadata(events, doc)
            
            # Ensure all events have descriptions and escalation analysis
            events = self._ensure_complete_events(events)
            
            logger.info(f"Extracted {len(events)} events with complete analysis from text")
            
        except Exception as e:
            logger.error(f"Error extracting events: {e}")
        
        return events
    
    def _extract_pattern_events(self, doc: Doc) -> List[Event]:
        """Extract events using enhanced pattern matching"""
        events = []
        text = doc.text
        
        for event_type, patterns in self.event_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    # Extract context sentence
                    context = self._get_context_sentence(doc, match.start(), match.end())
                    
                    # Extract participants and location
                    participants = self._extract_participants(context)
                    location = self._extract_location(context)
                    
                    # Determine severity
                    severity = self._determine_severity(context)
                    
                    # Extract keywords from match
                    keywords = self._extract_keywords_from_context(context)
                    
                    event = Event(
                        event_type=event_type,
                        participants=participants,
                        location=location,
                        severity_rating=severity,
                        context_sentence=context,
                        confidence_score=0.8,
                        keywords=keywords
                    )
                    events.append(event)
        
        return events
    
    def _extract_rule_based_events(self, doc: Doc) -> List[Event]:
        """Extract events using enhanced rule-based approach"""
        events = []
        
        # Look for verb-based events
        for sent in doc.sents:
            for token in sent:
                if token.pos_ == "VERB" and token.dep_ == "ROOT":
                    # Check if this verb indicates an event
                    verb_lemma = token.lemma_.lower()
                    
                    # Military/political action verbs
                    action_verbs = {
                        'attack': 'military_movement',
                        'deploy': 'military_movement', 
                        'launch': 'live_fire_exercise',
                        'fire': 'live_fire_exercise',
                        'patrol': 'naval_patrol',
                        'meet': 'summit_or_meeting',
                        'announce': 'diplomatic_statement',
                        'hack': 'cyber_attack',
                        'sanction': 'economic_signal'
                    }
                    
                    if verb_lemma in action_verbs:
                        event_type = action_verbs[verb_lemma]
                        
                        # Extract participants from subject and object
                        participants = []
                        for child in token.children:
                            if child.dep_ in ["nsubj", "dobj", "pobj"]:
                                participants.append(child.text)
                        
                        # Get sentence context
                        context = sent.text
                        
                        event = Event(
                            event_type=event_type,
                            participants=participants,
                            context_sentence=context,
                            confidence_score=0.7
                        )
                        events.append(event)
        
        return events
    
    def _extract_ner_events(self, doc: Doc, entities: List) -> List[Event]:
        """Extract events based on named entities"""
        events = []
        
        # Look for sentences with military/political entities
        for sent in doc.sents:
            sent_entities = [ent for ent in doc.ents if ent.start >= sent.start and ent.end <= sent.end]
            
            # Check for combinations that suggest events
            has_military = any(ent.label_ in ["ORG"] and any(mil in ent.text.lower() for mil in ["military", "army", "navy", "force"]) for ent in sent_entities)
            has_location = any(ent.label_ in ["GPE", "LOC"] for ent in sent_entities)
            has_action = any(token.pos_ == "VERB" for token in sent)
            
            if has_military and has_location and has_action:
                participants = [ent.text for ent in sent_entities if ent.label_ == "ORG"]
                location = next((ent.text for ent in sent_entities if ent.label_ in ["GPE", "LOC"]), None)
                
                event = Event(
                    event_type="military_movement",  # Default type
                    participants=participants,
                    location=location,
                    context_sentence=sent.text,
                    confidence_score=0.6
                )
                events.append(event)
        
        return events

    def _extract_participants(self, text: str) -> List[str]:
        """Enhanced participant extraction"""
        if not text:
            return []
        
        participants = []
        
        # Use spaCy to find organizations and people
        doc = self.nlp(text)
        for ent in doc.ents:
            if ent.label_ in ["ORG", "PERSON", "GPE"]:
                participants.append(ent.text)
        
        # Pattern-based extraction for military units
        military_patterns = [
            r'\b(?:PLA|People\'s Liberation Army)\b',
            r'\b(?:US|United States)\s+(?:Navy|Army|Air Force|Marines)\b',
            r'\b(?:Taiwan|ROC)\s+(?:military|forces|navy|army)\b',
            r'\b(?:\w+)\s+(?:fleet|squadron|division|regiment)\b'
        ]
        
        for pattern in military_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                participants.append(match.group())
        
        # Remove duplicates and clean
        participants = list(set(participants))
        return participants[:5]  # Limit to top 5

    def _extract_location(self, text: str) -> Optional[str]:
        """Enhanced location extraction"""
        if not text:
            return None
        
        # Use spaCy for location entities
        doc = self.nlp(text)
        locations = [ent.text for ent in doc.ents if ent.label_ in ["GPE", "LOC"]]
        
        # Pattern-based extraction for strategic locations
        strategic_patterns = [
            r'\bTaiwan Strait\b',
            r'\bSouth China Sea\b',
            r'\bEast China Sea\b',
            r'\bFirst Island Chain\b',
            r'\bMalacca Strait\b',
            r'\bLuzon Strait\b'
        ]
        
        for pattern in strategic_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()
        
        return locations[0] if locations else None

    def _determine_severity(self, text: str) -> str:
        """Enhanced severity determination"""
        if not text:
            return "medium"
        
        text_lower = text.lower()
        
        # Count severity indicators
        severity_scores = {}
        for severity, indicators in self.severity_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            severity_scores[severity] = score
        
        # Return highest scoring severity
        if severity_scores['critical'] > 0:
            return 'critical'
        elif severity_scores['high'] > 0:
            return 'high'
        elif severity_scores['medium'] > 0:
            return 'medium'
        else:
            return 'low'

    def _extract_keywords_from_context(self, context: str) -> List[str]:
        """Extract relevant keywords from context"""
        if not context:
            return []
        
        keywords = []
        context_lower = context.lower()
        
        # Extract escalation-related keywords
        all_escalation_words = []
        for level, words in self.escalation_keywords.items():
            all_escalation_words.extend(words)
        
        for word in all_escalation_words:
            if word in context_lower:
                keywords.append(word)
        
        # Extract military/political terms
        military_terms = ['military', 'naval', 'army', 'navy', 'air force', 'missile', 'weapon', 'defense']
        political_terms = ['diplomatic', 'government', 'minister', 'official', 'policy', 'statement']
        
        for term in military_terms + political_terms:
            if term in context_lower:
                keywords.append(term)
        
        return list(set(keywords))[:10]  # Limit and deduplicate

    def _get_context_sentence(self, doc: Doc, start: int, end: int) -> str:
        """Get the sentence containing the match"""
        for sent in doc.sents:
            if sent.start_char <= start <= sent.end_char:
                return sent.text
        return ""

    def _filter_events(self, events: List[Event]) -> List[Event]:
        """Filter and deduplicate events"""
        if not events:
            return events
        
        # Remove duplicates based on event type and context similarity
        filtered_events = []
        seen_contexts = set()
        
        for event in events:
            # Create a signature for the event
            signature = f"{event.event_type}_{event.context_sentence[:50]}"
            
            if signature not in seen_contexts:
                seen_contexts.add(signature)
                filtered_events.append(event)
        
        return filtered_events

    def _add_comprehensive_metadata(self, events: List[Event], doc: Doc) -> List[Event]:
        """Add comprehensive metadata to events"""
        for event in events:
            # Extract datetime information
            if event.context_sentence:
                event.datetime = self._extract_datetime(event.context_sentence)
            
            # Set model used
            event.model_used = "enhanced-event-classifier-v2"
            
            # Enhance confidence based on multiple factors
            confidence_factors = []
            
            # Pattern match confidence
            if any(pattern in event.context_sentence.lower() for patterns in self.event_patterns.values() for pattern in patterns):
                confidence_factors.append(0.8)
            
            # Entity presence confidence
            if event.participants:
                confidence_factors.append(0.7)
            
            # Location presence confidence
            if event.location:
                confidence_factors.append(0.6)
            
            # Calculate average confidence
            if confidence_factors:
                event.confidence_score = sum(confidence_factors) / len(confidence_factors)
            
        return events

    def _ensure_complete_events(self, events: List[Event]) -> List[Event]:
        """Ensure all events have complete descriptions and escalation analysis"""
        for event in events:
            # Trigger the post_init methods to ensure description and analysis are generated
            if not event.description:
                event.description = event._generate_description()
            
            if event.escalation_score == 0.0:
                event.escalation_score = event._calculate_escalation_score()
            
            if not event.escalation_analysis:
                event.escalation_analysis = event._generate_escalation_analysis()
        
        return events

    def _extract_datetime(self, text: str) -> Optional[str]:
        """Enhanced datetime extraction"""
        temporal_patterns = [
            r'\b(?:today|yesterday|tomorrow)\b',
            r'\b(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)\b',
            r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2}(?:,\s*\d{4})?\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b',
            r'\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b',
            r'\b(?:last|this|next)\s+(?:week|month|year)\b',
            r'\b\d{1,2}\s+(?:days?|weeks?|months?|years?)\s+ago\b'
        ]
        
        for pattern in temporal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group()
        
        return None

    def get_events_by_type(self, events: List[Event], event_type: str) -> List[Event]:
        """Get events filtered by type"""
        return [event for event in events if event.event_type == event_type]

    def get_events_by_severity(self, events: List[Event], severity: str) -> List[Event]:
        """Get events filtered by severity"""
        return [event for event in events if event.severity_rating == severity]

    def get_events_involving_participant(self, events: List[Event], participant: str) -> List[Event]:
        """Get events involving a specific participant"""
        return [event for event in events if any(participant.lower() in p.lower() for p in event.participants)]
    
    def get_high_escalation_events(self, events: List[Event], threshold: float = 0.6) -> List[Event]:
        """Get events with high escalation scores"""
        return [event for event in events if event.escalation_score >= threshold]
    
    def get_event_statistics(self, events: List[Event]) -> Dict[str, any]:
        """Get comprehensive statistics about extracted events"""
        if not events:
            return {}
        
        stats = {
            'total_events': len(events),
            'by_type': {},
            'by_severity': {},
            'escalation_stats': {
                'average_score': sum(e.escalation_score for e in events) / len(events),
                'max_score': max(e.escalation_score for e in events),
                'min_score': min(e.escalation_score for e in events),
                'high_escalation_count': len([e for e in events if e.escalation_score >= 0.6])
            },
            'participants': {},
            'locations': set()
        }
        
        # Count by type
        for event in events:
            stats['by_type'][event.event_type] = stats['by_type'].get(event.event_type, 0) + 1
            stats['by_severity'][event.severity_rating] = stats['by_severity'].get(event.severity_rating, 0) + 1
            
            # Count participants
            for participant in event.participants:
                stats['participants'][participant] = stats['participants'].get(participant, 0) + 1
            
            # Collect locations
            if event.location:
                stats['locations'].add(event.location)
        
        stats['locations'] = list(stats['locations'])
        
        return stats 