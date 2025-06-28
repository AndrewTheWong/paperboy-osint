#!/usr/bin/env python3
"""
Enhanced Tagging Layer with Relation and Event Extraction

Integrates multiple NLP capabilities:
1. Named Entity Recognition (NER)
2. Relation Extraction between entities
3. Event Extraction and classification
4. Geographic and semantic tagging
5. Sentiment analysis
6. Escalation prediction
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Core imports
from analytics.ner.relation_extraction import RelationExtractor, Relation
from analytics.ner.named_entity_recognizer import IntelligenceNER

# Enhanced event extraction
from tagging.enhanced_event_extractor import EnhancedEventExtractor, Event

# Try to import existing tagging components
try:
    from tagging.comprehensive_tagging_pipeline import ComprehensiveTaggingPipeline
    HAS_COMPREHENSIVE_TAGGING = True
except ImportError:
    HAS_COMPREHENSIVE_TAGGING = False

try:
    from pipelines.Ingest.NewsArticles.processors.article_tagger import UnifiedArticleTagger
    HAS_ARTICLE_TAGGER = True
except ImportError:
    HAS_ARTICLE_TAGGER = False

logger = logging.getLogger(__name__)

class EnhancedTaggingLayer:
    """
    Enhanced tagging layer that combines multiple NLP approaches.
    
    Features:
    - Named Entity Recognition
    - Relation Extraction
    - Event Extraction
    - Geographic Tagging
    - Sentiment Analysis
    - Escalation Scoring
    """
    
    def __init__(self):
        """Initialize the enhanced tagging layer."""
        self.relation_extractor = RelationExtractor()
        self.event_extractor = EnhancedEventExtractor()
        self.ner = IntelligenceNER()
        
        # Initialize existing taggers if available
        self.comprehensive_tagger = None
        self.article_tagger = None
        
        if HAS_COMPREHENSIVE_TAGGING:
            try:
                self.comprehensive_tagger = ComprehensiveTaggingPipeline()
                logger.info("✅ Loaded comprehensive tagging pipeline")
            except Exception as e:
                logger.warning(f"Failed to load comprehensive tagger: {e}")
        
        if HAS_ARTICLE_TAGGER:
            try:
                self.article_tagger = UnifiedArticleTagger()
                logger.info("✅ Loaded unified article tagger")
            except Exception as e:
                logger.warning(f"Failed to load article tagger: {e}")
        
        logger.info("Enhanced tagging layer initialized")
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete article with all tagging capabilities.
        
        Args:
            article: Article dictionary with 'title' and 'content' fields
            
        Returns:
            Enhanced article with all extracted information
        """
        title = article.get('title', '')
        content = article.get('content', '')
        combined_text = f"{title}. {content}"
        
        logger.info(f"Processing article: {title[:100]}...")
        
        # Start with the original article
        enhanced_article = article.copy()
        
        try:
            # Step 1: Named Entity Recognition
            entities = self.extract_entities(combined_text)
            enhanced_article['entities'] = entities
            
            # Step 2: Relation Extraction
            relations = self.extract_relations(combined_text, entities)
            enhanced_article['relations'] = [r.to_dict() for r in relations]
            
            # Step 3: Event Extraction
            events = self.extract_events(combined_text, entities)
            enhanced_article['events'] = [e.to_dict() for e in events]
            
            # Step 4: Event Relations
            event_relations = self.extract_event_relations(events)
            enhanced_article['event_relations'] = self._format_event_relations(event_relations)
            
            # Step 5: Enhanced Analytics
            analytics = self.extract_analytics(combined_text, entities, relations, events)
            enhanced_article.update(analytics)
            
            # Step 6: Integration with existing taggers
            if self.comprehensive_tagger:
                comprehensive_tags = self.comprehensive_tagger.process_article(article)
                enhanced_article = self._merge_comprehensive_tags(enhanced_article, comprehensive_tags)
            
            if self.article_tagger:
                article_tags = self.article_tagger.tag_article(title, content)
                enhanced_article = self._merge_article_tags(enhanced_article, article_tags)
            
            # Step 7: Generate summary insights
            insights = self.generate_insights(enhanced_article)
            enhanced_article['tagging_insights'] = insights
            
            logger.info(f"✅ Successfully processed article with {len(relations)} relations and {len(events)} events")
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            enhanced_article['tagging_error'] = str(e)
        
        return enhanced_article
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text."""
        try:
            ner_analysis = self.ner.extract_entities(text)
            # Convert to the expected format
            entities = {
                'PERSON': ner_analysis.people,
                'ORG': ner_analysis.organizations,
                'LOC': ner_analysis.locations,
                'GPE': ner_analysis.geopolitical_entities,
                'MILITARY': ner_analysis.military_terms
            }
            logger.debug(f"Extracted {sum(len(v) for v in entities.values())} entities")
            return entities
        except Exception as e:
            logger.error(f"Error extracting entities: {e}")
            return {}
    
    def extract_relations(self, text: str, entities: Dict[str, List[str]] = None) -> List[Relation]:
        """Extract relations between entities."""
        try:
            relations = self.relation_extractor.extract_relations(text, entities)
            logger.debug(f"Extracted {len(relations)} relations")
            return relations
        except Exception as e:
            logger.error(f"Error extracting relations: {e}")
            return []
    
    def extract_events(self, text: str, entities: Dict[str, List[str]] = None) -> List[Event]:
        """Extract events from text."""
        try:
            events = self.event_extractor.extract_events(text, entities)
            logger.debug(f"Extracted {len(events)} events")
            return events
        except Exception as e:
            logger.error(f"Error extracting events: {e}")
            return []
    
    def extract_event_relations(self, events: List[Event]) -> List[Tuple[Event, Event, str]]:
        """Extract relations between events."""
        try:
            # Check if the event extractor has extract_event_relations method
            if hasattr(self.event_extractor, 'extract_event_relations'):
                event_relations = self.event_extractor.extract_event_relations(events)
                logger.debug(f"Extracted {len(event_relations)} event relations")
                return event_relations
            else:
                # Simple fallback: look for temporal or causal relations
                event_relations = []
                for i, event1 in enumerate(events):
                    for j, event2 in enumerate(events[i+1:], i+1):
                        # Simple heuristic: if events are related by location or type
                        if (event1.location and event2.location and event1.location == event2.location):
                            event_relations.append((event1, event2, "co_located"))
                        elif event1.event_type == event2.event_type:
                            event_relations.append((event1, event2, "similar_type"))
                
                logger.debug(f"Generated {len(event_relations)} simple event relations")
                return event_relations
        except Exception as e:
            logger.error(f"Error extracting event relations: {e}")
            return []
    
    def extract_analytics(self, text: str, entities: Dict[str, List[str]], 
                         relations: List[Relation], events: List[Event]) -> Dict[str, Any]:
        """Extract analytical insights from the extracted information."""
        analytics = {}
        
        try:
            # Relation statistics
            if relations:
                relation_stats = self.relation_extractor.get_relation_statistics(relations)
                analytics['relation_statistics'] = relation_stats
            
            # Event statistics
            if events:
                event_stats = self.event_extractor.get_event_statistics(events)
                analytics['event_statistics'] = event_stats
            
            # Conflict and escalation indicators
            analytics['escalation_indicators'] = self._analyze_escalation_indicators(
                relations, events, entities
            )
            
            # Key actor analysis
            analytics['key_actors'] = self._analyze_key_actors(relations, events, entities)
            
            # Geographic focus
            analytics['geographic_focus'] = self._analyze_geographic_focus(
                relations, events, entities
            )
            
            # Temporal patterns
            analytics['temporal_patterns'] = self._analyze_temporal_patterns(events)
            
        except Exception as e:
            logger.error(f"Error extracting analytics: {e}")
            analytics['analytics_error'] = str(e)
        
        return analytics
    
    def _analyze_escalation_indicators(self, relations: List[Relation], 
                                     events: List[Event], 
                                     entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze indicators of conflict escalation."""
        indicators = {
            'conflict_events': 0,
            'attack_relations': 0,
            'tension_relations': 0,
            'diplomatic_events': 0,
            'military_events': 0,
            'escalation_score': 0.0,
            'escalation_factors': []
        }
        
        # Count conflict-related events
        for event in events:
            event_type = event.event_type.lower() if hasattr(event, 'event_type') else ''
            
            if any(conflict_type in event_type for conflict_type in ['military', 'conflict', 'attack', 'live_fire']):
                indicators['conflict_events'] += 1
                indicators['escalation_factors'].append(f"Conflict event: {event_type}")
            elif 'diplomatic' in event_type:
                indicators['diplomatic_events'] += 1
            elif 'military' in event_type:
                indicators['military_events'] += 1
        
        # Count hostile relations
        for relation in relations:
            if relation.relation_type in ['attacks', 'opposes', 'sanctions', 'in_conflict_with']:
                indicators['attack_relations'] += 1
                indicators['escalation_factors'].append(f"Hostile relation: {relation.relation_type}")
            elif relation.relation_type == 'has_tensions_with':
                indicators['tension_relations'] += 1
                indicators['escalation_factors'].append(f"Tension: {relation.subject} - {relation.object}")
        
        # Calculate escalation score
        escalation_score = 0
        escalation_score += indicators['conflict_events'] * 0.3
        escalation_score += indicators['attack_relations'] * 0.25
        escalation_score += indicators['tension_relations'] * 0.15
        escalation_score += indicators['military_events'] * 0.2
        escalation_score -= indicators['diplomatic_events'] * 0.1  # Diplomatic events reduce escalation
        
        indicators['escalation_score'] = min(1.0, max(0.0, escalation_score))
        
        return indicators
    
    def _analyze_key_actors(self, relations: List[Relation], 
                           events: List[Event], 
                           entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze key actors based on their involvement in relations and events."""
        actor_involvement = {}
        
        # Count involvement in relations
        for relation in relations:
            for actor in [relation.subject, relation.object]:
                if actor not in actor_involvement:
                    actor_involvement[actor] = {
                        'relation_count': 0,
                        'event_count': 0,
                        'relation_types': set(),
                        'event_types': set(),
                        'roles': set()
                    }
                actor_involvement[actor]['relation_count'] += 1
                actor_involvement[actor]['relation_types'].add(relation.relation_type)
        
        # Count involvement in events
        for event in events:
            # Handle enhanced Event class with participants as list of strings
            participants = event.participants if hasattr(event, 'participants') else []
            for participant in participants:
                actor = participant if isinstance(participant, str) else str(participant)
                if actor not in actor_involvement:
                    actor_involvement[actor] = {
                        'relation_count': 0,
                        'event_count': 0,
                        'relation_types': set(),
                        'event_types': set(),
                        'roles': set()
                    }
                actor_involvement[actor]['event_count'] += 1
                actor_involvement[actor]['event_types'].add(event.event_type)
                actor_involvement[actor]['roles'].add('participant')  # Default role for enhanced events
        
        # Convert sets to lists for JSON serialization
        for actor in actor_involvement:
            actor_involvement[actor]['relation_types'] = list(actor_involvement[actor]['relation_types'])
            actor_involvement[actor]['event_types'] = list(actor_involvement[actor]['event_types'])
            actor_involvement[actor]['roles'] = list(actor_involvement[actor]['roles'])
        
        # Sort by total involvement
        sorted_actors = sorted(
            actor_involvement.items(),
            key=lambda x: x[1]['relation_count'] + x[1]['event_count'],
            reverse=True
        )
        
        return {
            'actor_involvement': dict(sorted_actors[:10]),  # Top 10 actors
            'total_actors': len(actor_involvement)
        }
    
    def _analyze_geographic_focus(self, relations: List[Relation], 
                                 events: List[Event], 
                                 entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Analyze geographic focus of the article."""
        locations = set()
        
        # Extract locations from entities
        for entity_type in ['GPE', 'LOC', 'countries', 'locations']:
            if entity_type in entities:
                locations.update(entities[entity_type])
        
        # Extract locations from relations
        for relation in relations:
            if relation.subject_type in ['GPE', 'LOC'] or 'location' in relation.subject_type.lower():
                locations.add(relation.subject)
            if relation.object_type in ['GPE', 'LOC'] or 'location' in relation.object_type.lower():
                locations.add(relation.object)
        
        # Extract locations from events
        for event in events:
            if hasattr(event, 'location') and event.location:
                locations.add(event.location)
            # Handle enhanced Event class with participants as list of strings
            participants = event.participants if hasattr(event, 'participants') else []
            for participant in participants:
                participant_str = participant if isinstance(participant, str) else str(participant)
                # Simple heuristic: if participant contains location-like words
                if any(loc_word in participant_str.lower() for loc_word in ['taiwan', 'china', 'strait', 'sea', 'island']):
                    locations.add(participant_str)
        
        return {
            'primary_locations': list(locations)[:10],
            'location_count': len(locations)
        }
    
    def _analyze_temporal_patterns(self, events: List[Event]) -> Dict[str, Any]:
        """Analyze temporal patterns in events."""
        temporal_info = {
            'events_with_time': 0,
            'temporal_indicators': [],
            'event_sequence': []
        }
        
        for event in events:
            event_time = getattr(event, 'datetime', None) or getattr(event, 'time', None)
            if event_time:
                temporal_info['events_with_time'] += 1
                temporal_info['temporal_indicators'].append(event_time)
            
            temporal_info['event_sequence'].append({
                'event_type': event.event_type,
                'description': getattr(event, 'description', ''),
                'time': event_time,
                'location': getattr(event, 'location', None)
            })
        
        return temporal_info
    
    def _format_event_relations(self, event_relations: List[Tuple[Event, Event, str]]) -> List[Dict[str, Any]]:
        """Format event relations for JSON serialization."""
        formatted_relations = []
        
        for event1, event2, relation_type in event_relations:
            formatted_relations.append({
                'event1': {
                    'type': event1.event_type,
                    'description': getattr(event1, 'description', ''),
                    'location': getattr(event1, 'location', None)
                },
                'event2': {
                    'type': event2.event_type,
                    'description': getattr(event2, 'description', ''),
                    'location': getattr(event2, 'location', None)
                },
                'relation_type': relation_type
            })
        
        return formatted_relations
    
    def _merge_comprehensive_tags(self, enhanced_article: Dict[str, Any], 
                                comprehensive_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from comprehensive tagging pipeline."""
        # Add comprehensive tagging results
        if 'comprehensive_tags' not in enhanced_article:
            enhanced_article['comprehensive_tags'] = {}
        
        # Merge specific fields
        merge_fields = [
            'embedding', 'keywords', 'semantic_tags', 'geographic_tags',
            'sentiment', 'escalation_score', 'geographic_info'
        ]
        
        for field in merge_fields:
            if field in comprehensive_tags:
                enhanced_article['comprehensive_tags'][field] = comprehensive_tags[field]
        
        return enhanced_article
    
    def _merge_article_tags(self, enhanced_article: Dict[str, Any], 
                           article_tags: Dict[str, Any]) -> Dict[str, Any]:
        """Merge results from article tagger."""
        # Add article tagging results
        if 'article_tags' not in enhanced_article:
            enhanced_article['article_tags'] = {}
        
        # Merge specific fields
        merge_fields = [
            'keywords', 'sentiment', 'escalation_score', 
            'locations', 'countries', 'regions'
        ]
        
        for field in merge_fields:
            if field in article_tags:
                enhanced_article['article_tags'][field] = article_tags[field]
        
        return enhanced_article
    
    def generate_insights(self, enhanced_article: Dict[str, Any]) -> Dict[str, Any]:
        """Generate high-level insights from all extracted information."""
        insights = {
            'summary': {},
            'key_findings': [],
            'complexity_score': 0.0,
            'information_density': 0.0
        }
        
        try:
            # Count extracted information
            entity_count = sum(len(v) for v in enhanced_article.get('entities', {}).values())
            relation_count = len(enhanced_article.get('relations', []))
            event_count = len(enhanced_article.get('events', []))
            
            insights['summary'] = {
                'entities_extracted': entity_count,
                'relations_extracted': relation_count,
                'events_extracted': event_count,
                'has_escalation_indicators': enhanced_article.get('escalation_indicators', {}).get('escalation_score', 0) > 0.3
            }
            
            # Generate key findings
            if relation_count > 5:
                insights['key_findings'].append("High relational complexity detected")
            
            if event_count > 3:
                insights['key_findings'].append("Multiple events identified")
            
            escalation_score = enhanced_article.get('escalation_indicators', {}).get('escalation_score', 0)
            if escalation_score > 0.5:
                insights['key_findings'].append(f"Significant escalation indicators (score: {escalation_score:.2f})")
            
            # Calculate complexity score
            text_length = len(enhanced_article.get('content', ''))
            if text_length > 0:
                insights['complexity_score'] = min(1.0, (entity_count + relation_count * 2 + event_count * 3) / text_length * 1000)
                insights['information_density'] = (entity_count + relation_count + event_count) / (text_length / 100)
            
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def get_tagging_statistics(self) -> Dict[str, Any]:
        """Get statistics about the tagging layer capabilities."""
        stats = {
            'components_available': {
                'relation_extractor': True,
                'event_extractor': True,
                'ner': True,
                'comprehensive_tagger': self.comprehensive_tagger is not None,
                'article_tagger': self.article_tagger is not None
            },
            'relation_types': list(self.relation_extractor.get_relation_types().keys()),
            'event_types': [et.value for et in EventType],
            'capabilities': [
                'Named Entity Recognition',
                'Relation Extraction',
                'Event Extraction',
                'Event Relation Analysis',
                'Escalation Indicator Analysis',
                'Key Actor Analysis',
                'Geographic Focus Analysis',
                'Temporal Pattern Analysis'
            ]
        }
        
        return stats

# Convenience function for easy integration
def process_article_with_enhanced_tagging(article: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to process an article with enhanced tagging.
    
    Args:
        article: Article dictionary with 'title' and 'content'
        
    Returns:
        Enhanced article with all extracted information
    """
    tagging_layer = EnhancedTaggingLayer()
    return tagging_layer.process_article(article)

# Example usage
if __name__ == "__main__":
    # Test the enhanced tagging layer
    sample_article = {
        'title': 'China and Taiwan Tensions Escalate as Military Exercises Begin',
        'content': '''China launched military exercises near Taiwan yesterday, raising tensions in the region. 
        The exercises involve naval and air force units conducting drills in the Taiwan Strait. 
        Taiwan's defense ministry condemned the actions, calling them provocative. 
        The United States expressed concern and called for restraint from both sides. 
        This follows recent diplomatic talks between Beijing and Washington aimed at reducing tensions.'''
    }
    
    print("Testing Enhanced Tagging Layer...")
    enhanced_article = process_article_with_enhanced_tagging(sample_article)
    
    print(f"Extracted {len(enhanced_article.get('relations', []))} relations")
    print(f"Extracted {len(enhanced_article.get('events', []))} events")
    print(f"Escalation score: {enhanced_article.get('escalation_indicators', {}).get('escalation_score', 0):.2f}")
    
    # Pretty print results
    import json
    print("\nFull Results:")
    print(json.dumps(enhanced_article, indent=2, default=str)) 