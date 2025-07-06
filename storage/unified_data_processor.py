#!/usr/bin/env python3
"""
Unified Data Processor for GDELT and OSINT Articles
Ensures consistent tagging, embedding, and storage for both data types
"""

import logging
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ProcessedData:
    """Standardized data structure for both GDELT and OSINT"""
    title: str
    content: str
    source: str
    url: Optional[str] = None
    primary_country: Optional[str] = None
    primary_city: Optional[str] = None
    primary_region: Optional[str] = None
    coordinates: Optional[tuple] = None
    keyword_tags: List[str] = field(default_factory=list)
    semantic_tags: List[str] = field(default_factory=list)
    geographic_tags: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    significance_score: Optional[float] = None
    title_embedding: Optional[List[float]] = None
    content_embedding: Optional[List[float]] = None
    language: str = 'en'
    processed_at: datetime = field(default_factory=datetime.utcnow)

class UnifiedDataProcessor:
    """Unified processor for both GDELT and OSINT data"""
    
    def __init__(self):
        self.supabase = None
        self.embedding_model = None
        self.sentiment_analyzer = None
        self.tagging_pipeline = None
        self.geo_extractor = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize all components"""
        logger.info("Initializing unified data processing components...")
        
        try:
            from utils.supabase_client import get_supabase_client
            self.supabase = get_supabase_client()
            logger.info("Supabase client initialized")
            
            try:
                from sentence_transformers import SentenceTransformer
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Embedding model loaded")
            except Exception as e:
                logger.warning(f"Could not load embedding model: {e}")
            
            try:
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis", 
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    return_all_scores=True
                )
                logger.info("Sentiment analyzer loaded")
            except Exception as e:
                logger.warning(f"Could not load sentiment analyzer: {e}")
            
            try:
                from pipelines.tagging.enhanced_article_tagging import EnhancedArticleTaggingPipeline
                self.tagging_pipeline = EnhancedArticleTaggingPipeline()
                logger.info("Tagging pipeline loaded")
            except Exception as e:
                logger.warning(f"Could not load tagging pipeline: {e}")
            
            try:
                from pipelines.tagging.geographic_extractor import GeographicExtractor
                self.geo_extractor = GeographicExtractor()
                logger.info("Geographic extractor loaded")
            except Exception as e:
                logger.warning(f"Could not load geographic extractor: {e}")
                
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise
    
    def process_article(self, article_data: Dict[str, Any], data_type: str = 'osint') -> ProcessedData:
        """Process a single article with unified pipeline"""
        try:
            title = article_data.get('title', '')
            content = article_data.get('content', article_data.get('summary', ''))
            source = article_data.get('source', 'Unknown')
            url = article_data.get('url', article_data.get('source_url'))
            
            processed = ProcessedData(
                title=title,
                content=content,
                source=source,
                url=url
            )
            
            if title:
                processed.title_embedding = self._generate_embedding(title)
            if content:
                processed.content_embedding = self._generate_embedding(content)
            
            processed.keyword_tags = self._extract_keyword_tags(title, content)
            processed.semantic_tags = self._extract_semantic_tags(title, content)
            
            geo_info = self._extract_geographic_info(title, content)
            processed.primary_country = geo_info.get('primary_country')
            processed.primary_city = geo_info.get('primary_city')
            processed.primary_region = geo_info.get('primary_region')
            processed.geographic_tags = geo_info.get('locations', [])
            processed.coordinates = geo_info.get('coordinates')
            
            processed.sentiment_score = self._analyze_sentiment(title, content)
            processed.significance_score = self._calculate_significance_score(processed)
            
            return processed
            
        except Exception as e:
            logger.error(f"Failed to process article: {e}")
            return ProcessedData(
                title=article_data.get('title', ''),
                content=article_data.get('content', ''),
                source=article_data.get('source', 'Unknown'),
                url=article_data.get('url')
            )
    
    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for text"""
        try:
            if not text or not self.embedding_model:
                return []
            
            text = text.strip()[:512]
            if not text:
                return []
            
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
            
        except Exception as e:
            logger.warning(f"Failed to generate embedding: {e}")
            return []
    
    def _extract_keyword_tags(self, title: str, content: str) -> List[str]:
        """Extract keyword tags from text"""
        try:
            if not self.tagging_pipeline:
                return self._basic_keyword_extraction(title, content)
            
            article_data = {'title': title, 'content': content}
            tagged = self.tagging_pipeline.tag_single_article(article_data)
            return tagged.get('tags', [])
            
        except Exception as e:
            logger.warning(f"Failed to extract keyword tags: {e}")
            return self._basic_keyword_extraction(title, content)
    
    def _basic_keyword_extraction(self, title: str, content: str) -> List[str]:
        """Basic keyword extraction fallback"""
        text = f"{title} {content}".lower()
        keywords = []
        
        keyword_patterns = {
            'china': ['china', 'chinese', 'beijing'],
            'taiwan': ['taiwan', 'taiwanese', 'taipei'],
            'usa': ['usa', 'america', 'united states'],
            'japan': ['japan', 'japanese', 'tokyo'],
            'military': ['military', 'army', 'navy', 'defense'],
            'economic': ['trade', 'economy', 'business'],
            'diplomatic': ['diplomat', 'embassy', 'foreign']
        }
        
        for keyword, patterns in keyword_patterns.items():
            if any(pattern in text for pattern in patterns):
                keywords.append(keyword)
        
        return keywords
    
    def _extract_semantic_tags(self, title: str, content: str) -> List[str]:
        """Extract semantic tags using NLP"""
        try:
            text = f"{title} {content}".lower()
            semantic_tags = []
            
            if any(word in text for word in ['military', 'defense', 'armed forces', 'navy', 'army', 'air force']):
                semantic_tags.append('military')
            
            if any(word in text for word in ['diplomatic', 'embassy', 'foreign minister', 'ambassador']):
                semantic_tags.append('diplomacy')
            
            if any(word in text for word in ['trade', 'economic', 'market', 'business', 'investment']):
                semantic_tags.append('economic')
            
            if any(word in text for word in ['technology', 'cyber', 'tech', 'digital', 'ai', 'semiconductor']):
                semantic_tags.append('technology')
            
            if any(word in text for word in ['security', 'threat', 'risk', 'warning', 'alert']):
                semantic_tags.append('security')
            
            return semantic_tags
            
        except Exception as e:
            logger.warning(f"Failed to extract semantic tags: {e}")
            return []
    
    def _extract_geographic_info(self, title: str, content: str) -> Dict[str, Any]:
        """Extract geographic information"""
        try:
            if not self.geo_extractor:
                return self._basic_geographic_extraction(title, content)
            
            text = f"{title} {content}"
            geo_info = self.geo_extractor.extract_geographic_info(text)
            
            return {
                'primary_country': geo_info.get('country'),
                'primary_city': geo_info.get('city'),
                'primary_region': geo_info.get('region'),
                'locations': geo_info.get('locations', []),
                'coordinates': geo_info.get('coordinates')
            }
            
        except Exception as e:
            logger.warning(f"Failed to extract geographic info: {e}")
            return self._basic_geographic_extraction(title, content)
    
    def _basic_geographic_extraction(self, title: str, content: str) -> Dict[str, Any]:
        """Basic geographic extraction fallback"""
        text = f"{title} {content}".lower()
        
        countries = {
            'china': ['china', 'chinese', 'beijing'],
            'taiwan': ['taiwan', 'taiwanese', 'taipei'],
            'usa': ['usa', 'america', 'united states'],
            'japan': ['japan', 'japanese', 'tokyo'],
            'south korea': ['korea', 'korean', 'seoul']
        }
        
        detected_country = None
        for country, patterns in countries.items():
            if any(pattern in text for pattern in patterns):
                detected_country = country
                break
        
        return {
            'primary_country': detected_country,
            'primary_city': None,
            'primary_region': None,
            'locations': [detected_country] if detected_country else [],
            'coordinates': None
        }
    
    def _analyze_sentiment(self, title: str, content: str) -> float:
        """Analyze sentiment and return normalized score"""
        try:
            if not self.sentiment_analyzer:
                return 0.0
            
            text = f"{title} {content}".strip()[:512]
            if not text:
                return 0.0
            
            results = self.sentiment_analyzer(text)
            
            for result in results[0]:
                if result['label'] == 'LABEL_0':
                    return -result['score']
                elif result['label'] == 'LABEL_2':
                    return result['score']
            
            return 0.0
            
        except Exception as e:
            logger.warning(f"Failed to analyze sentiment: {e}")
            return 0.0
    
    def _calculate_significance_score(self, processed: ProcessedData) -> float:
        """Calculate significance score based on multiple factors"""
        try:
            score = 0.0
            
            tag_richness = len(processed.keyword_tags) / 10.0
            score += min(tag_richness, 0.3)
            
            if processed.primary_country:
                score += 0.1
            if processed.primary_city:
                score += 0.1
            
            return min(score, 1.0)
            
        except Exception as e:
            logger.warning(f"Failed to calculate significance score: {e}")
            return 0.0
    
    def store_osint_articles(self, processed_articles: List[ProcessedData]) -> Dict[str, int]:
        """Store processed OSINT articles in database"""
        logger.info(f"Storing {len(processed_articles)} OSINT articles...")
        
        success_count = 0
        error_count = 0
        
        for article in processed_articles:
            try:
                article_data = {
                    'title': article.title,
                    'content': article.content,
                    'url': article.url,
                    'source': article.source,
                    'primary_country': article.primary_country,
                    'primary_city': article.primary_city,
                    'primary_region': article.primary_region,
                    'all_locations': article.geographic_tags,
                    'keyword_tags': article.keyword_tags,
                    'semantic_tags': article.semantic_tags,
                    'geographic_tags': article.geographic_tags,
                    'sentiment_score': article.sentiment_score,
                    'significance_score': article.significance_score,
                    'language': article.language,
                    'scraped_at': datetime.utcnow().isoformat(),
                    'processed_at': article.processed_at.isoformat()
                }
                
                if article.title_embedding:
                    article_data['title_embedding'] = json.dumps(article.title_embedding)
                if article.content_embedding:
                    article_data['content_embedding'] = json.dumps(article.content_embedding)
                
                result = self.supabase.table('osint_articles').insert(article_data).execute()
                
                if result.data:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error storing article {article.title[:50]}: {e}")
        
        logger.info(f"Stored {success_count} articles, {error_count} errors")
        return {'success': success_count, 'errors': error_count}
    
    def store_gdelt_events(self, processed_events: List[ProcessedData]) -> Dict[str, int]:
        """Store processed GDELT events in database"""
        logger.info(f"Storing {len(processed_events)} GDELT events...")
        
        success_count = 0
        error_count = 0
        
        for event in processed_events:
            try:
                event_data = {
                    'title': event.title,
                    'content': event.content,
                    'source_url': event.url,
                    'source': 'GDELT',
                    'primary_country': event.primary_country,
                    'primary_city': event.primary_city,
                    'primary_region': event.primary_region,
                    'keyword_tags': event.keyword_tags,
                    'semantic_tags': event.semantic_tags,
                    'geographic_tags': event.geographic_tags,
                    'sentiment_score': event.sentiment_score,
                    'significance_score': event.significance_score,
                    'language': event.language,
                    'processed_at': event.processed_at.isoformat()
                }
                
                if event.title_embedding:
                    event_data['title_embedding'] = json.dumps(event.title_embedding)
                if event.content_embedding:
                    event_data['content_embedding'] = json.dumps(event.content_embedding)
                
                result = self.supabase.table('gdelt_events').insert(event_data).execute()
                
                if result.data:
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                error_count += 1
                logger.error(f"Error storing event {event.title[:50]}: {e}")
        
        logger.info(f"Stored {success_count} events, {error_count} errors")
        return {'success': success_count, 'errors': error_count}
    
    def process_and_store_batch(self, data_batch: List[Dict[str, Any]], data_type: str) -> Dict[str, Any]:
        """Process and store a batch of data"""
        logger.info(f"Processing batch of {len(data_batch)} {data_type} items...")
        
        try:
            processed_items = []
            for item in data_batch:
                processed = self.process_article(item, data_type)
                processed_items.append(processed)
            
            if data_type == 'osint':
                storage_result = self.store_osint_articles(processed_items)
            elif data_type == 'gdelt':
                storage_result = self.store_gdelt_events(processed_items)
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            return {
                'processed': len(processed_items),
                'stored': storage_result['success'],
                'errors': storage_result['errors'],
                'data_type': data_type
            }
            
        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            raise 