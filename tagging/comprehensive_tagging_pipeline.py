#!/usr/bin/env python3
"""
Comprehensive Tagging and Embedding Pipeline
Processes both GDELT and OSINT data with complete tagging and embedding before Supabase upload
"""

import os
import sys
import json
import logging
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import re
from dataclasses import dataclass

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Core ML imports
try:
    from sentence_transformers import SentenceTransformer
    import spacy
    from transformers import pipeline
    import torch
except ImportError as e:
    logging.warning(f"Some ML libraries not available: {e}")

# Project imports
from supabase import create_client

# Try to import geographic extractor, fallback if not available
try:
    from pipelines.tagging.geographic_extractor import extract_geographic_info
except ImportError:
    # Fallback function if geographic extractor not available
    def extract_geographic_info(text: str) -> Dict[str, Any]:
        """Extract geographic information from text."""
        result = {
            'primary_country': None,
            'primary_city': None,
            'primary_region': None,
            'all_locations': [],
            'geographic_confidence': 0.0,
            'coordinates': None
        }
        
        try:
            import spacy
            
            # Load spaCy model if available
            try:
                nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not available for geographic extraction")
                return result
            
            # Process text
            doc = nlp(text[:2000])  # Limit text length for performance
            
            # Country mappings
            country_mappings = {
                'united states': 'United States',
                'usa': 'United States', 
                'us': 'United States',
                'america': 'United States',
                'china': 'China',
                'taiwan': 'Taiwan',
                'japan': 'Japan',
                'south korea': 'South Korea',
                'korea': 'South Korea',
                'north korea': 'North Korea',
                'russia': 'Russia',
                'ukraine': 'Ukraine',
                'philippines': 'Philippines',
                'singapore': 'Singapore',
                'malaysia': 'Malaysia',
                'indonesia': 'Indonesia',
                'thailand': 'Thailand',
                'vietnam': 'Vietnam',
                'india': 'India',
                'australia': 'Australia',
                'united kingdom': 'United Kingdom',
                'uk': 'United Kingdom',
                'britain': 'United Kingdom',
                'france': 'France',
                'germany': 'Germany',
                'israel': 'Israel',
                'iran': 'Iran',
                'iraq': 'Iraq',
                'afghanistan': 'Afghanistan',
                'pakistan': 'Pakistan'
            }
            
            # Region mappings
            region_mappings = {
                'asia pacific': 'Asia-Pacific',
                'asia-pacific': 'Asia-Pacific',
                'east asia': 'East Asia',
                'southeast asia': 'Southeast Asia',
                'south asia': 'South Asia',
                'middle east': 'Middle East',
                'europe': 'Europe',
                'north america': 'North America',
                'south america': 'South America',
                'africa': 'Africa',
                'oceania': 'Oceania',
                'indo-pacific': 'Indo-Pacific'
            }
            
            # City coordinates (major cities)
            city_coordinates = {
                'beijing': (39.9042, 116.4074),
                'shanghai': (31.2304, 121.4737),
                'taipei': (25.0330, 121.5654),
                'tokyo': (35.6762, 139.6503),
                'seoul': (37.5665, 126.9780),
                'pyongyang': (39.0392, 125.7625),
                'manila': (14.5995, 120.9842),
                'singapore': (1.3521, 103.8198),
                'kuala lumpur': (3.1390, 101.6869),
                'jakarta': (6.2088, 106.8456),
                'bangkok': (13.7563, 100.5018),
                'hanoi': (21.0285, 105.8542),
                'new delhi': (28.6139, 77.2090),
                'mumbai': (19.0760, 72.8777),
                'sydney': (33.8688, 151.2093),
                'melbourne': (37.8136, 144.9631),
                'washington': (38.9072, 77.0369),
                'new york': (40.7128, 74.0060),
                'los angeles': (34.0522, 118.2437),
                'london': (51.5074, 0.1278),
                'paris': (48.8566, 2.3522),
                'berlin': (52.5200, 13.4050),
                'moscow': (55.7558, 37.6173),
                'kyiv': (50.4501, 30.5234),
                'tehran': (35.6892, 51.3890),
                'baghdad': (33.3152, 44.3661),
                'kabul': (34.5553, 69.2075),
                'islamabad': (33.7294, 73.0931)
            }
            
            # Extract entities
            locations = []
            countries = []
            cities = []
            regions = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entities and locations
                    location_text = ent.text.lower().strip()
                    
                    # Check if it's a country
                    if location_text in country_mappings:
                        countries.append(country_mappings[location_text])
                        locations.append({
                            'name': country_mappings[location_text],
                            'type': 'country',
                            'confidence': 0.8
                        })
                    
                    # Check if it's a city
                    elif location_text in city_coordinates:
                        cities.append(location_text.title())
                        locations.append({
                            'name': location_text.title(),
                            'type': 'city',
                            'confidence': 0.7,
                            'coordinates': city_coordinates[location_text]
                        })
            
            # Check for regions in text
            text_lower = text.lower()
            for region_key, region_name in region_mappings.items():
                if region_key in text_lower:
                    regions.append(region_name)
                    locations.append({
                        'name': region_name,
                        'type': 'region',
                        'confidence': 0.6
                    })
            
            # Determine primary locations
            if countries:
                result['primary_country'] = countries[0]  # Most mentioned or first found
            
            if cities:
                result['primary_city'] = cities[0]
                # Get coordinates for primary city
                primary_city_lower = cities[0].lower()
                if primary_city_lower in city_coordinates:
                    result['coordinates'] = city_coordinates[primary_city_lower]
            
            if regions:
                result['primary_region'] = regions[0]
            
            # Set all locations and confidence
            result['all_locations'] = locations
            result['geographic_confidence'] = min(len(locations) * 0.2, 1.0)  # Max 1.0
            
            # If no specific region found but we have a country, infer region
            if not result['primary_region'] and result['primary_country']:
                country = result['primary_country']
                if country in ['China', 'Japan', 'South Korea', 'North Korea', 'Taiwan']:
                    result['primary_region'] = 'East Asia'
                elif country in ['United States', 'Canada', 'Mexico']:
                    result['primary_region'] = 'North America'
                elif country in ['Singapore', 'Malaysia', 'Indonesia', 'Thailand', 'Vietnam', 'Philippines']:
                    result['primary_region'] = 'Southeast Asia'
                elif country in ['India', 'Pakistan', 'Afghanistan']:
                    result['primary_region'] = 'South Asia'
                elif country in ['Australia', 'New Zealand']:
                    result['primary_region'] = 'Oceania'
                elif country in ['United Kingdom', 'France', 'Germany', 'Russia', 'Ukraine']:
                    result['primary_region'] = 'Europe'
                elif country in ['Israel', 'Iran', 'Iraq']:
                    result['primary_region'] = 'Middle East'
            
        except Exception as e:
            logger.error(f"Error in geographic extraction: {e}")
        
        return result

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TaggingConfig:
    """Configuration for tagging pipeline."""
    sbert_model: str = "all-MiniLM-L6-v2"
    spacy_model: str = "en_core_web_sm"
    sentiment_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    embedding_dim: int = 384
    max_text_length: int = 512
    batch_size: int = 32
    enable_gpu: bool = True

class ComprehensiveTaggingPipeline:
    """Comprehensive tagging and embedding pipeline for GDELT and OSINT data."""
    
    def __init__(self, config: Optional[TaggingConfig] = None):
        """Initialize the tagging pipeline."""
        self.config = config or TaggingConfig()
        
        # Initialize models
        self.sbert_model = None
        self.nlp = None
        self.sentiment_analyzer = None
        self.device = "cuda" if torch.cuda.is_available() and self.config.enable_gpu else "cpu"
        
        # Initialize Supabase
        self.supabase_url = os.getenv('SUPABASE_URL', 'https://kyoowijhqqqnlmqceuka.supabase.co')
        self.supabase_key = os.getenv('SUPABASE_KEY', 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Imt5b283aWpocXFxbmxtcWNldWthIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NDc3MDU4MjAsImV4cCI6MjA2MzI4MTgyMH0.f7GRfEOmaSdZtvhhJFue_bhgMTILzum_ePZ-os7f_WE')
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        
        # Load models
        self._load_models()
        
        # Geographic mappings
        self.country_mappings = self._load_country_mappings()
        
    def _load_models(self):
        """Load all required models."""
        logger.info("Loading ML models...")
        
        try:
            # Load SBERT model
            self.sbert_model = SentenceTransformer(self.config.sbert_model, device=self.device)
            logger.info(f"✅ Loaded SBERT model: {self.config.sbert_model}")
            
            # Load spaCy model
            self.nlp = spacy.load(self.config.spacy_model)
            logger.info(f"✅ Loaded spaCy model: {self.config.spacy_model}")
            
            # Load sentiment analyzer
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=self.config.sentiment_model,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"✅ Loaded sentiment model: {self.config.sentiment_model}")
            
        except Exception as e:
            logger.error(f"❌ Error loading models: {e}")
            logger.warning("⚠️ Some models may not be available - continuing with limited functionality")
    
    def _load_country_mappings(self) -> Dict[str, str]:
        """Load country code mappings."""
        return {
            "USA": "United States", "CHN": "China", "TWN": "Taiwan", 
            "JPN": "Japan", "KOR": "South Korea", "PRK": "North Korea",
            "RUS": "Russia", "PHL": "Philippines", "IND": "India",
            "AUS": "Australia", "UKR": "Ukraine", "VNM": "Vietnam",
            "THA": "Thailand", "SGP": "Singapore", "MYS": "Malaysia",
            "IDN": "Indonesia", "AFG": "Afghanistan", "IRQ": "Iraq",
            "IRN": "Iran", "ISR": "Israel", "PAK": "Pakistan"
        }
    
    def process_gdelt_data(self, gdelt_records: List[Dict]) -> List[Dict]:
        """Process GDELT data with comprehensive tagging and embedding."""
        logger.info(f"🔄 Processing {len(gdelt_records)} GDELT records...")
        
        processed_records = []
        
        for i, record in enumerate(gdelt_records):
            try:
                # Create comprehensive text for analysis
                title = record.get('title', f"GDELT Event {record.get('event_code', '010')}")
                content = record.get('content', f"GDELT event between {record.get('actor1_country', 'UNK')} and {record.get('actor2_country', 'UNK')}")
                full_text = f"{title}. {content}"
                
                # Extract tags and embeddings
                tagging_result = self._comprehensive_tagging(full_text, title, content)
                
                # Enhance GDELT record
                enhanced_record = record.copy()
                enhanced_record.update({
                    # Required GDELT fields
                    'keyword_tags': tagging_result['keyword_tags'],
                    'semantic_tags': tagging_result['semantic_tags'],
                    'geographic_tags': tagging_result['geographic_tags'],
                    'title_embedding': tagging_result['title_embedding'],
                    'content_embedding': tagging_result['content_embedding'],
                    'coordinates': tagging_result['coordinates'],
                    
                    # Enhanced metadata
                    'sentiment_score': tagging_result['sentiment_score'],
                    'significance_score': tagging_result['significance_score'],
                    'word_count': len(full_text.split()),
                    'processed_at': datetime.now().isoformat()
                })
                
                processed_records.append(enhanced_record)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(gdelt_records)} GDELT records")
                
            except Exception as e:
                logger.error(f"❌ Error processing GDELT record {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully processed {len(processed_records)}/{len(gdelt_records)} GDELT records")
        return processed_records
    
    def process_osint_data(self, osint_records: List[Dict]) -> List[Dict]:
        """Process OSINT data with comprehensive tagging and embedding."""
        logger.info(f"🔄 Processing {len(osint_records)} OSINT records...")
        
        processed_records = []
        
        for i, record in enumerate(osint_records):
            try:
                # Extract text fields
                title = record.get('title', '')
                content = record.get('content', '')
                
                if not content and not title:
                    logger.warning(f"⚠️ Skipping OSINT record {i}: No content or title")
                    continue
                
                # Create comprehensive text for analysis
                full_text = f"{title}. {content}".strip()
                
                # Extract tags and embeddings
                tagging_result = self._comprehensive_tagging(full_text, title, content)
                
                # Create summary if not provided
                summary = record.get('summary') or self._create_summary(content, title)
                
                # Enhance OSINT record
                enhanced_record = record.copy()
                enhanced_record.update({
                    # Required OSINT fields
                    'content': content,
                    'summary': summary,
                    'primary_country': tagging_result['primary_country'],
                    'primary_city': tagging_result['primary_city'],
                    'primary_region': tagging_result['primary_region'],
                    'all_locations': tagging_result['all_locations'],
                    'geographic_confidence': tagging_result['geographic_confidence'],
                    'coordinates': tagging_result['coordinates'],
                    'keyword_tags': tagging_result['keyword_tags'],
                    'semantic_tags': tagging_result['semantic_tags'],
                    'geographic_tags': tagging_result['geographic_tags'],
                    'sentiment_score': tagging_result['sentiment_score'],
                    'significance_score': tagging_result['significance_score'],
                    'title_embedding': tagging_result['title_embedding'],
                    'content_embedding': tagging_result['content_embedding'],
                    'published_at': record.get('published_at') or record.get('scraped_at') or datetime.now().isoformat(),
                    'word_count': len(full_text.split()),
                    'processed_at': datetime.now().isoformat()
                })
                
                processed_records.append(enhanced_record)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"📊 Processed {i + 1}/{len(osint_records)} OSINT records")
                
            except Exception as e:
                logger.error(f"❌ Error processing OSINT record {i}: {e}")
                continue
        
        logger.info(f"✅ Successfully processed {len(processed_records)}/{len(osint_records)} OSINT records")
        return processed_records
    
    def _comprehensive_tagging(self, full_text: str, title: str = "", content: str = "") -> Dict[str, Any]:
        """Apply comprehensive tagging to text."""
        result = {
            'keyword_tags': [],
            'semantic_tags': [],
            'sentiment_score': 0.0,
            'significance_score': 0.5,
            'title_embedding': None,
            'content_embedding': None,
            'summary': '',
            'word_count': 0
        }
        
        try:
            # Use content if available, otherwise use full_text
            text_to_analyze = content if content and len(content.strip()) > 0 else full_text
            
            # Calculate word count properly
            if text_to_analyze:
                # Remove extra whitespace and count words
                cleaned_text = ' '.join(text_to_analyze.split())
                result['word_count'] = len(cleaned_text.split()) if cleaned_text else 0
            
            # Create English summary
            result['summary'] = self._create_summary(text_to_analyze, title)
            
            # 1. Extract keywords using spaCy
            if self.nlp:
                result['keyword_tags'] = self._extract_keywords(text_to_analyze)
                result['semantic_tags'] = self._extract_semantic_tags(text_to_analyze)
            
            # 2. Sentiment analysis - fix the scoring issue
            if self.sentiment_analyzer and text_to_analyze:
                result['sentiment_score'] = self._analyze_sentiment(text_to_analyze)
            
            # 3. Create embeddings
            if self.sbert_model:
                if title and len(title.strip()) > 0:
                    result['title_embedding'] = self._create_embedding(title)
                if text_to_analyze and len(text_to_analyze.strip()) > 0:
                    result['content_embedding'] = self._create_embedding(text_to_analyze)
            
            # 4. Calculate significance score
            result['significance_score'] = self._calculate_significance(text_to_analyze, result)
            
        except Exception as e:
            logger.error(f"Error in comprehensive tagging: {e}")
        
        return result
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract keywords using spaCy NER and POS tagging."""
        if not self.nlp:
            return []
        
        try:
            doc = self.nlp(text[:self.config.max_text_length])
            keywords = []
            
            # Extract named entities
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'GPE', 'EVENT', 'PRODUCT']:
                    keywords.append(ent.text.lower())
            
            # Extract important nouns and adjectives
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN'] and 
                    len(token.text) > 3 and 
                    not token.is_stop and 
                    token.is_alpha):
                    keywords.append(token.lemma_.lower())
            
            # Remove duplicates and limit
            keywords = list(set(keywords))[:20]
            return keywords
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            return []
    
    def _extract_semantic_tags(self, text: str) -> List[str]:
        """Extract semantic tags based on content analysis."""
        semantic_tags = []
        text_lower = text.lower()
        
        # Military/Security tags
        military_keywords = ['military', 'defense', 'security', 'army', 'navy', 'air force', 'weapon', 'missile', 'nuclear']
        if any(keyword in text_lower for keyword in military_keywords):
            semantic_tags.append('military')
        
        # Economic tags
        economic_keywords = ['economy', 'trade', 'economic', 'financial', 'market', 'investment', 'gdp', 'inflation']
        if any(keyword in text_lower for keyword in economic_keywords):
            semantic_tags.append('economic')
        
        # Political tags
        political_keywords = ['government', 'political', 'policy', 'election', 'democracy', 'president', 'minister']
        if any(keyword in text_lower for keyword in political_keywords):
            semantic_tags.append('political')
        
        # Diplomatic tags
        diplomatic_keywords = ['diplomatic', 'embassy', 'ambassador', 'treaty', 'negotiation', 'summit', 'meeting']
        if any(keyword in text_lower for keyword in diplomatic_keywords):
            semantic_tags.append('diplomatic')
        
        # Conflict tags
        conflict_keywords = ['conflict', 'war', 'attack', 'violence', 'terrorism', 'protest', 'crisis']
        if any(keyword in text_lower for keyword in conflict_keywords):
            semantic_tags.append('conflict')
        
        # Technology tags
        tech_keywords = ['technology', 'cyber', 'digital', 'artificial intelligence', 'ai', 'computer', 'internet']
        if any(keyword in text_lower for keyword in tech_keywords):
            semantic_tags.append('technology')
        
        return semantic_tags
    
    def _extract_geographic_tags(self, geo_info: Dict[str, Any]) -> List[str]:
        """Extract geographic tags from geographic information."""
        tags = []
        
        # Add country tags
        if geo_info.get('primary_country'):
            tags.append(f"country:{geo_info['primary_country'].lower()}")
        
        # Add region tags
        if geo_info.get('primary_region'):
            tags.append(f"region:{geo_info['primary_region'].lower()}")
        
        # Add city tags
        if geo_info.get('primary_city'):
            tags.append(f"city:{geo_info['primary_city'].lower()}")
        
        # Add location type tags
        for location in geo_info.get('all_locations', []):
            if isinstance(location, dict) and location.get('type'):
                tags.append(f"location_type:{location['type']}")
        
        return tags
    
    def _analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment and return score between -1 and 1."""
        try:
            # Truncate text if too long
            text = text[:self.config.max_text_length]
            
            if not text or len(text.strip()) < 5:
                return 0.0
            
            result = self.sentiment_analyzer(text)
            
            # The model returns different label formats, handle both
            label = result[0]['label'].upper()
            score = result[0]['score']
            
            # Map labels to scores
            if label in ['POSITIVE', 'LABEL_2']:
                return score
            elif label in ['NEGATIVE', 'LABEL_0']:
                return -score
            else:  # NEUTRAL, LABEL_1, or unknown
                return 0.0
                
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return 0.0
    
    def _create_embedding(self, text: str) -> Optional[List[float]]:
        """Create SBERT embedding for text."""
        if not self.sbert_model:
            return None
        
        try:
            # Truncate text if too long
            text = text[:self.config.max_text_length]
            
            # Generate embedding
            embedding = self.sbert_model.encode(text, convert_to_numpy=True)
            
            # Convert to list for JSON serialization
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Error creating embedding: {e}")
            return None
    
    def _format_coordinates(self, coordinates: Optional[Tuple[float, float]]) -> Optional[str]:
        """Format coordinates for PostgreSQL POINT type."""
        if coordinates and len(coordinates) == 2:
            lat, lon = coordinates
            return f"POINT({lon} {lat})"  # PostgreSQL expects (longitude, latitude)
        return None
    
    def _create_summary(self, content: str, title: str = "") -> str:
        """Create an English summary of the content."""
        if not content or len(content.strip()) < 10:
            return title or "No content available"
        
        # Clean the content
        content = content.strip()
        
        # If we have a title, use it as the base
        if title and len(title.strip()) > 0:
            title = title.strip()
            # If content is very short, just return title + content
            if len(content) < 100:
                combined = f"{title}: {content}"
                return combined[:200] + ('...' if len(combined) > 200 else '')
        
        # Extract meaningful sentences (skip very short ones)
        sentences = re.split(r'[.!?]+', content)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            # Keep sentences that are substantial (>20 chars) and in Latin script
            if (len(sentence) > 20 and 
                any(c.isalpha() and ord(c) < 256 for c in sentence)):
                meaningful_sentences.append(sentence)
                if len(meaningful_sentences) >= 2:
                    break
        
        if meaningful_sentences:
            summary = '. '.join(meaningful_sentences)
            # Ensure it ends with a period
            if not summary.endswith('.'):
                summary += '.'
            return summary[:300] + ('...' if len(summary) > 300 else '')
        
        # Fallback: use first 200 chars of content
        fallback = content[:200].strip()
        if not fallback.endswith('.'):
            fallback += '...'
        return fallback
    
    def _calculate_significance(self, text: str, tagging_result: Dict[str, Any]) -> float:
        """Calculate significance score based on various factors."""
        score = 0.5  # Base score
        
        # Geographic significance
        geo_confidence = tagging_result.get('geographic_confidence', 0.0)
        score += geo_confidence * 0.2
        
        # Keyword significance
        keyword_count = len(tagging_result.get('keyword_tags', []))
        score += min(keyword_count / 20.0, 0.2)
        
        # Semantic tag significance
        semantic_count = len(tagging_result.get('semantic_tags', []))
        score += min(semantic_count / 5.0, 0.2)
        
        # Text length significance
        word_count = len(text.split())
        if word_count > 100:
            score += 0.1
        
        # Sentiment extremity
        sentiment = abs(tagging_result.get('sentiment_score', 0.0))
        score += sentiment * 0.1
        
        return min(score, 1.0)
    
    def upload_gdelt_to_supabase(self, processed_records: List[Dict]) -> bool:
        """Upload processed GDELT records to Supabase."""
        logger.info(f"📤 Uploading {len(processed_records)} GDELT records to Supabase...")
        
        batch_size = 50
        uploaded = 0
        errors = 0
        
        for i in range(0, len(processed_records), batch_size):
            batch = processed_records[i:i + batch_size]
            
            try:
                response = self.supabase.table('gdelt_events').insert(batch).execute()
                
                if hasattr(response, 'data') and response.data:
                    uploaded += len(response.data)
                else:
                    errors += len(batch)
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ GDELT batch upload error: {e}")
                errors += len(batch)
                continue
        
        success_rate = (uploaded / len(processed_records)) * 100 if processed_records else 0
        logger.info(f"✅ GDELT upload complete: {uploaded:,} successful ({success_rate:.1f}%), {errors:,} errors")
        
        return success_rate > 80
    
    def upload_osint_to_supabase(self, processed_records: List[Dict]) -> bool:
        """Upload processed OSINT records to Supabase."""
        logger.info(f"📤 Uploading {len(processed_records)} OSINT records to Supabase...")
        
        batch_size = 50
        uploaded = 0
        errors = 0
        
        for i in range(0, len(processed_records), batch_size):
            batch = processed_records[i:i + batch_size]
            
            try:
                response = self.supabase.table('osint_articles').insert(batch).execute()
                
                if hasattr(response, 'data') and response.data:
                    uploaded += len(response.data)
                else:
                    errors += len(batch)
                
                time.sleep(0.2)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ OSINT batch upload error: {e}")
                errors += len(batch)
                continue
        
        success_rate = (uploaded / len(processed_records)) * 100 if processed_records else 0
        logger.info(f"✅ OSINT upload complete: {uploaded:,} successful ({success_rate:.1f}%), {errors:,} errors")
        
        return success_rate > 80

    def process_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single article with comprehensive tagging."""
        try:
            # Extract text content
            title = article.get('title', '')
            content = article.get('content', '')
            full_text = f"{title} {content}".strip()
            
            if not full_text:
                logger.warning("Article has no content to process")
                return article
            
            # Apply comprehensive tagging
            tagging_result = self._comprehensive_tagging(full_text, title, content)
            
            # Create enhanced article with all fields
            enhanced_article = article.copy()
            enhanced_article.update({
                # Core fields
                'content': content,
                'summary': tagging_result['summary'],
                'word_count': tagging_result['word_count'],
                'processed_at': datetime.now().isoformat(),
                
                # Tagging fields
                'keyword_tags': tagging_result['keyword_tags'],
                'semantic_tags': tagging_result['semantic_tags'],
                
                # ML fields
                'sentiment_score': tagging_result['sentiment_score'],
                'significance_score': tagging_result['significance_score'],
                'title_embedding': tagging_result['title_embedding'],
                'content_embedding': tagging_result['content_embedding'],
                
                # Ensure published_at exists
                'published_at': article.get('published_at') or article.get('scraped_at') or datetime.now().isoformat()
            })
            
            return enhanced_article
            
        except Exception as e:
            logger.error(f"Error processing article: {e}")
            # Return original article with error info
            article['processing_error'] = str(e)
            return article

def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Tagging Pipeline')
    parser.add_argument('--data-type', choices=['gdelt', 'osint'], required=True, help='Type of data to process')
    parser.add_argument('--input-file', type=str, help='Input JSON file with data')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ComprehensiveTaggingPipeline()
    
    if args.test:
        # Test with sample data
        if args.data_type == 'gdelt':
            sample_data = [{
                'event_date': '2024-01-01',
                'actor1_country': 'USA',
                'actor2_country': 'CHN',
                'event_code': '010',
                'title': 'US-China Trade Meeting',
                'content': 'Officials from the United States and China met to discuss trade relations and economic cooperation.',
                'goldstein_score': 1.5
            }]
            processed = pipeline.process_gdelt_data(sample_data)
            print(f"Processed GDELT sample: {json.dumps(processed[0], indent=2)}")
        else:
            sample_data = [{
                'title': 'Taiwan Military Exercise',
                'content': 'Taiwan conducted a military exercise in the Taiwan Strait amid rising tensions with China.',
                'url': 'https://example.com/news',
                'source': 'Test News'
            }]
            processed = pipeline.process_osint_data(sample_data)
            print(f"Processed OSINT sample: {json.dumps(processed[0], indent=2)}")
    
    elif args.input_file:
        # Process file
        with open(args.input_file, 'r') as f:
            data = json.load(f)
        
        if args.data_type == 'gdelt':
            processed = pipeline.process_gdelt_data(data)
            pipeline.upload_gdelt_to_supabase(processed)
        else:
            processed = pipeline.process_osint_data(data)
            pipeline.upload_osint_to_supabase(processed)
    
    else:
        logger.error("❌ Please provide --input-file or --test")

if __name__ == '__main__':
    main()