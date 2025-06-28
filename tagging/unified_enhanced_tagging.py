"""
Unified Enhanced Tagging Pipeline
Combines all enhanced extractors and stores results in Supabase
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

from .enhanced_entity_extractor import EnhancedEntityExtractor, Entity
from .enhanced_relation_extractor import EnhancedRelationExtractor, Relation
from .enhanced_event_extractor import EnhancedEventExtractor, Event
from .enhanced_sentiment_analyzer import EnhancedSentimentAnalyzer, SentimentAnalysis

logger = logging.getLogger(__name__)

class UnifiedEnhancedTagging:
    """Unified enhanced tagging pipeline for OSINT articles"""
    
    def __init__(self):
        """Initialize the unified tagging pipeline"""
        logger.info("Initializing Unified Enhanced Tagging Pipeline")
        
        # Initialize all extractors
        self.entity_extractor = EnhancedEntityExtractor()
        self.relation_extractor = EnhancedRelationExtractor()
        self.event_extractor = EnhancedEventExtractor()
        self.sentiment_analyzer = EnhancedSentimentAnalyzer()
        
        logger.info("✅ All extractors initialized successfully")
    
    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a complete article with all enhanced tagging capabilities
        
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
            # Step 1: Extract entities
            entities = self.entity_extractor.extract_entities(combined_text)
            enhanced_article['entities'] = [self._entity_to_dict(e) for e in entities]
            
            # Step 2: Extract relations
            relations = self.relation_extractor.extract_relations(combined_text, entities)
            enhanced_article['relations'] = [self._relation_to_dict(r) for r in relations]
            
            # Step 3: Extract events
            events = self.event_extractor.extract_events(combined_text, entities)
            enhanced_article['events'] = [self._event_to_dict(e) for e in events]
            
            # Step 4: Analyze sentiment
            sentiment = self.sentiment_analyzer.analyze_sentiment(combined_text)
            enhanced_article['sentiment'] = self._sentiment_to_dict(sentiment)
            
            # Step 5: Generate summary insights
            insights = self._generate_insights(entities, relations, events, sentiment)
            enhanced_article['tagging_insights'] = insights
            
            # Step 6: Add metadata
            enhanced_article['processed_at'] = datetime.now().isoformat()
            enhanced_article['tagging_version'] = 'enhanced-v1.0'
            
            logger.info(f"✅ Successfully processed article with {len(entities)} entities, {len(relations)} relations, {len(events)} events")
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            enhanced_article['tagging_error'] = str(e)
        
        return enhanced_article
    
    def _entity_to_dict(self, entity: Entity) -> Dict[str, Any]:
        """Convert Entity to dictionary"""
        return {
            'entity': entity.entity,
            'entity_type': entity.entity_type,
            'linked_id': entity.linked_id,
            'normalized_name': entity.normalized_name,
            'context_sentence': entity.context_sentence,
            'confidence_score': entity.confidence_score,
            'model_used': entity.model_used
        }
    
    def _relation_to_dict(self, relation: Relation) -> Dict[str, Any]:
        """Convert Relation to dictionary"""
        return {
            'subject': relation.subject,
            'predicate': relation.predicate,
            'object': relation.object,
            'context_sentence': relation.context_sentence,
            'confidence_score': relation.confidence_score,
            'model_used': relation.model_used
        }
    
    def _event_to_dict(self, event: Event) -> Dict[str, Any]:
        """Convert Event to dictionary"""
        return {
            'event_type': event.event_type,
            'participants': event.participants,
            'location': event.location,
            'datetime': event.datetime,
            'severity_rating': event.severity_rating,
            'confidence_score': event.confidence_score,
            'model_used': event.model_used,
            'context_sentence': event.context_sentence
        }
    
    def _sentiment_to_dict(self, sentiment: SentimentAnalysis) -> Dict[str, Any]:
        """Convert SentimentAnalysis to dictionary"""
        return {
            'sentiment_toward_Taiwan': sentiment.sentiment_toward_Taiwan,
            'escalation_level': sentiment.escalation_level,
            'intent_signal': sentiment.intent_signal,
            'strategic_posture_change': sentiment.strategic_posture_change,
            'info_warfare_detected': sentiment.info_warfare_detected,
            'confidence_score': sentiment.confidence_score,
            'model_used': sentiment.model_used
        }
    
    def _generate_insights(self, entities: List[Entity], relations: List[Relation], 
                          events: List[Event], sentiment: SentimentAnalysis) -> Dict[str, Any]:
        """Generate summary insights from extracted information"""
        insights = {
            'summary': '',
            'key_entities': [],
            'key_relations': [],
            'key_events': [],
            'risk_assessment': '',
            'recommendations': []
        }
        
        # Key entities (high confidence and important types)
        key_entities = [e for e in entities if e.confidence_score > 0.8 and 
                       e.entity_type in ['PERSON', 'ORG', 'GPE', 'FACILITY']]
        insights['key_entities'] = [e.entity for e in key_entities[:5]]
        
        # Key relations (high confidence)
        key_relations = [r for r in relations if r.confidence_score > 0.7]
        insights['key_relations'] = [f"{r.subject} {r.predicate} {r.object}" for r in key_relations[:3]]
        
        # Key events (high severity or confidence)
        key_events = [e for e in events if e.severity_rating in ['high', 'critical'] or e.confidence_score > 0.8]
        insights['key_events'] = [e.event_type for e in key_events[:3]]
        
        # Risk assessment based on sentiment and events
        risk_level = 'low'
        if sentiment.escalation_level == 'high' or sentiment.sentiment_toward_Taiwan == 'hostile':
            risk_level = 'high'
        elif sentiment.escalation_level == 'medium' or any(e.severity_rating == 'high' for e in events):
            risk_level = 'medium'
        
        insights['risk_assessment'] = f"Risk level: {risk_level}"
        
        # Generate summary
        summary_parts = []
        if entities:
            summary_parts.append(f"Identified {len(entities)} entities")
        if relations:
            summary_parts.append(f"Extracted {len(relations)} relations")
        if events:
            summary_parts.append(f"Detected {len(events)} events")
        
        summary_parts.append(f"Sentiment: {sentiment.sentiment_toward_Taiwan}")
        summary_parts.append(f"Escalation: {sentiment.escalation_level}")
        
        insights['summary'] = "; ".join(summary_parts)
        
        # Recommendations
        recommendations = []
        if sentiment.escalation_level == 'high':
            recommendations.append("Monitor for further escalation")
        if sentiment.info_warfare_detected:
            recommendations.append("Verify information sources")
        if any(e.event_type == 'military_movement' for e in events):
            recommendations.append("Track military movements")
        
        insights['recommendations'] = recommendations
        
        return insights
    
    def store_to_supabase(self, enhanced_article: Dict[str, Any], supabase_client) -> bool:
        """
        Store enhanced article data to Supabase using the new schema
        
        Args:
            enhanced_article: Enhanced article with all extracted data
            supabase_client: Supabase client instance
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Insert main event record
            event_data = {
                'title': enhanced_article.get('title', ''),
                'content': enhanced_article.get('content', ''),
                'summary': enhanced_article.get('summary', ''),
                'url': enhanced_article.get('url', ''),
                'source': enhanced_article.get('source', ''),
                'source_name': enhanced_article.get('source_name', ''),
                'language_original': enhanced_article.get('language_original', 'en'),
                'primary_country': enhanced_article.get('primary_country'),
                'primary_city': enhanced_article.get('primary_city'),
                'primary_region': enhanced_article.get('primary_region'),
                'all_locations': enhanced_article.get('all_locations', []),
                'geographic_confidence': enhanced_article.get('geographic_confidence'),
                'published_at': enhanced_article.get('published_at'),
                'scraped_at': enhanced_article.get('scraped_at'),
                'word_count': enhanced_article.get('word_count', 0),
                'processed_at': enhanced_article.get('processed_at')
            }
            
            # Insert event and get ID
            result = supabase_client.table('events').insert(event_data).execute()
            if not result.data:
                logger.error("Failed to insert event record")
                return False
            
            event_id = result.data[0]['id']
            logger.info(f"Inserted event with ID: {event_id}")
            
            # Insert entities
            entities = enhanced_article.get('entities', [])
            for entity in entities:
                entity_data = {
                    'article_id': event_id,
                    'entity': entity['entity'],
                    'entity_type': entity['entity_type'],
                    'linked_id': entity.get('linked_id'),
                    'normalized_name': entity.get('normalized_name'),
                    'context_sentence': entity.get('context_sentence'),
                    'confidence_score': entity.get('confidence_score', 0.8),
                    'model_used': entity.get('model_used', 'spacy-en_core_web_sm')
                }
                supabase_client.table('entities').insert(entity_data).execute()
            
            # Insert relations
            relations = enhanced_article.get('relations', [])
            for relation in relations:
                relation_data = {
                    'article_id': event_id,
                    'subject': relation['subject'],
                    'predicate': relation['predicate'],
                    'object': relation['object'],
                    'context_sentence': relation.get('context_sentence'),
                    'confidence_score': relation.get('confidence_score', 0.8),
                    'model_used': relation.get('model_used', 'custom-relation-extractor')
                }
                supabase_client.table('relations').insert(relation_data).execute()
            
            # Insert event tags
            events = enhanced_article.get('events', [])
            for event in events:
                event_data = {
                    'article_id': event_id,
                    'event_type': event['event_type'],
                    'participants': event.get('participants', []),
                    'location': event.get('location'),
                    'datetime': event.get('datetime'),
                    'severity_rating': event.get('severity_rating', 'medium'),
                    'confidence_score': event.get('confidence_score', 0.8),
                    'model_used': event.get('model_used', 'custom-event-classifier')
                }
                supabase_client.table('event_tags').insert(event_data).execute()
            
            # Insert sentiment
            sentiment = enhanced_article.get('sentiment', {})
            if sentiment:
                sentiment_data = {
                    'article_id': event_id,
                    'sentiment_toward_Taiwan': sentiment.get('sentiment_toward_Taiwan', 'neutral'),
                    'escalation_level': sentiment.get('escalation_level', 'low'),
                    'intent_signal': sentiment.get('intent_signal', 'symbolic'),
                    'strategic_posture_change': sentiment.get('strategic_posture_change', False),
                    'info_warfare_detected': sentiment.get('info_warfare_detected', False),
                    'confidence_score': sentiment.get('confidence_score', 0.8),
                    'model_used': sentiment.get('model_used', 'custom-sentiment-analyzer')
                }
                supabase_client.table('article_sentiment').insert(sentiment_data).execute()
            
            # Insert article tags for quick reference
            self._insert_article_tags(event_id, enhanced_article, supabase_client)
            
            logger.info(f"✅ Successfully stored all data to Supabase for event ID: {event_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error storing to Supabase: {e}")
            return False
    
    def _insert_article_tags(self, event_id: int, enhanced_article: Dict[str, Any], supabase_client):
        """Insert article tags for quick reference"""
        tags = []
        
        # Entity tags
        entities = enhanced_article.get('entities', [])
        for entity in entities:
            tags.append({
                'article_id': event_id,
                'tag_type': 'entity',
                'tag_value': entity['entity'],
                'tag_metadata': {
                    'entity_type': entity['entity_type'],
                    'confidence': entity.get('confidence_score', 0.8)
                }
            })
        
        # Relation tags
        relations = enhanced_article.get('relations', [])
        for relation in relations:
            tags.append({
                'article_id': event_id,
                'tag_type': 'relation',
                'tag_value': f"{relation['subject']} {relation['predicate']} {relation['object']}",
                'tag_metadata': {
                    'predicate': relation['predicate'],
                    'confidence': relation.get('confidence_score', 0.8)
                }
            })
        
        # Event tags
        events = enhanced_article.get('events', [])
        for event in events:
            tags.append({
                'article_id': event_id,
                'tag_type': 'event',
                'tag_value': event['event_type'],
                'tag_metadata': {
                    'severity': event.get('severity_rating', 'medium'),
                    'confidence': event.get('confidence_score', 0.8)
                }
            })
        
        # Sentiment tags
        sentiment = enhanced_article.get('sentiment', {})
        if sentiment:
            tags.append({
                'article_id': event_id,
                'tag_type': 'sentiment',
                'tag_value': sentiment.get('sentiment_toward_Taiwan', 'neutral'),
                'tag_metadata': {
                    'escalation': sentiment.get('escalation_level', 'low'),
                    'confidence': sentiment.get('confidence_score', 0.8)
                }
            })
        
        # Insert all tags
        if tags:
            supabase_client.table('article_tags').insert(tags).execute()
    
    def get_processing_summary(self, enhanced_article: Dict[str, Any]) -> str:
        """Get a human-readable summary of the processing results"""
        entities = enhanced_article.get('entities', [])
        relations = enhanced_article.get('relations', [])
        events = enhanced_article.get('events', [])
        sentiment = enhanced_article.get('sentiment', {})
        
        summary = f"""
Enhanced Tagging Results:
- Entities extracted: {len(entities)}
- Relations extracted: {len(relations)}
- Events detected: {len(events)}
- Sentiment toward Taiwan: {sentiment.get('sentiment_toward_Taiwan', 'unknown')}
- Escalation level: {sentiment.get('escalation_level', 'unknown')}
- Intent signal: {sentiment.get('intent_signal', 'unknown')}
- Strategic posture change: {sentiment.get('strategic_posture_change', False)}
- Information warfare detected: {sentiment.get('info_warfare_detected', False)}
        """.strip()
        
        return summary 