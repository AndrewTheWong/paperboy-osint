#!/usr/bin/env python3
"""
Unified Geographic Tagger
Comprehensive geographic information extraction from article text
"""

import logging
import re
import json
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)

try:
    import spacy
    HAS_SPACY = True
except ImportError:
    logger.warning("spaCy not available. Geographic extraction will be limited.")
    HAS_SPACY = False

@dataclass
class GeographicLocation:
    """Represents a geographic location with metadata"""
    name: str
    location_type: str  # country, city, region, landmark, sea
    confidence: float
    coordinates: Optional[Tuple[float, float]] = None
    country_code: Optional[str] = None
    parent_location: Optional[str] = None

class UnifiedGeographicTagger:
    """Comprehensive geographic information extraction"""
    
    def __init__(self):
        """Initialize the geographic tagger"""
        self.nlp = self._load_spacy_model()
        self.locations_db = self._build_locations_database()
        
    def _load_spacy_model(self):
        """Load spaCy model for NER"""
        if not HAS_SPACY:
            return None
        
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model 'en_core_web_sm' not found. Geographic extraction will be limited.")
            return None
    
    def _build_locations_database(self) -> Dict[str, Dict[str, Any]]:
        """Build comprehensive locations database"""
        return {
            # Major Countries
            "china": {
                "official_name": "China",
                "type": "country",
                "aliases": ["people's republic of china", "prc", "mainland china"],
                "coordinates": (35.8617, 104.1954),
                "code": "CN"
            },
            "united states": {
                "official_name": "United States",
                "type": "country", 
                "aliases": ["usa", "us", "america", "united states of america"],
                "coordinates": (37.0902, -95.7129),
                "code": "US"
            },
            "taiwan": {
                "official_name": "Taiwan",
                "type": "country",
                "aliases": ["republic of china", "roc", "formosa"],
                "coordinates": (23.6978, 120.9605),
                "code": "TW"
            },
            "japan": {
                "official_name": "Japan",
                "type": "country",
                "aliases": ["nippon", "nihon"],
                "coordinates": (36.2048, 138.2529),
                "code": "JP"
            },
            "south korea": {
                "official_name": "South Korea",
                "type": "country",
                "aliases": ["republic of korea", "rok", "korea"],
                "coordinates": (35.9078, 127.7669),
                "code": "KR"
            },
            "north korea": {
                "official_name": "North Korea",
                "type": "country",
                "aliases": ["democratic people's republic of korea", "dprk"],
                "coordinates": (40.3399, 127.5101),
                "code": "KP"
            },
            "russia": {
                "official_name": "Russia",
                "type": "country",
                "aliases": ["russian federation"],
                "coordinates": (61.5240, 105.3188),
                "code": "RU"
            },
            "ukraine": {
                "official_name": "Ukraine",
                "type": "country",
                "aliases": [],
                "coordinates": (48.3794, 31.1656),
                "code": "UA"
            },
            "philippines": {
                "official_name": "Philippines",
                "type": "country",
                "aliases": [],
                "coordinates": (12.8797, 121.7740),
                "code": "PH"
            },
            "vietnam": {
                "official_name": "Vietnam",
                "type": "country",
                "aliases": ["viet nam"],
                "coordinates": (14.0583, 108.2772),
                "code": "VN"
            },
            "singapore": {
                "official_name": "Singapore",
                "type": "country",
                "aliases": [],
                "coordinates": (1.3521, 103.8198),
                "code": "SG"
            },
            "malaysia": {
                "official_name": "Malaysia",
                "type": "country",
                "aliases": [],
                "coordinates": (4.2105, 101.9758),
                "code": "MY"
            },
            "thailand": {
                "official_name": "Thailand",
                "type": "country",
                "aliases": [],
                "coordinates": (15.8700, 100.9925),
                "code": "TH"
            },
            "indonesia": {
                "official_name": "Indonesia",
                "type": "country",
                "aliases": [],
                "coordinates": (-0.7893, 113.9213),
                "code": "ID"
            },
            "india": {
                "official_name": "India",
                "type": "country",
                "aliases": [],
                "coordinates": (20.5937, 78.9629),
                "code": "IN"
            },
            "pakistan": {
                "official_name": "Pakistan",
                "type": "country",
                "aliases": [],
                "coordinates": (30.3753, 69.3451),
                "code": "PK"
            },
            "iran": {
                "official_name": "Iran",
                "type": "country",
                "aliases": ["islamic republic of iran"],
                "coordinates": (32.4279, 53.6880),
                "code": "IR"
            },
            "israel": {
                "official_name": "Israel",
                "type": "country",
                "aliases": [],
                "coordinates": (31.0461, 34.8516),
                "code": "IL"
            },
            
            # Major Cities
            "beijing": {
                "official_name": "Beijing",
                "type": "city",
                "country": "China",
                "aliases": ["peking"],
                "coordinates": (39.9042, 116.4074)
            },
            "shanghai": {
                "official_name": "Shanghai",
                "type": "city",
                "country": "China",
                "aliases": [],
                "coordinates": (31.2304, 121.4737)
            },
            "taipei": {
                "official_name": "Taipei",
                "type": "city",
                "country": "Taiwan",
                "aliases": [],
                "coordinates": (25.0330, 121.5654)
            },
            "tokyo": {
                "official_name": "Tokyo",
                "type": "city",
                "country": "Japan",
                "aliases": [],
                "coordinates": (35.6762, 139.6503)
            },
            "seoul": {
                "official_name": "Seoul",
                "type": "city",
                "country": "South Korea",
                "aliases": [],
                "coordinates": (37.5665, 126.9780)
            },
            "pyongyang": {
                "official_name": "Pyongyang",
                "type": "city",
                "country": "North Korea",
                "aliases": [],
                "coordinates": (39.0392, 125.7625)
            },
            "washington": {
                "official_name": "Washington D.C.",
                "type": "city",
                "country": "United States",
                "aliases": ["washington d.c.", "washington dc", "d.c."],
                "coordinates": (38.9072, -77.0369)
            },
            "new york": {
                "official_name": "New York",
                "type": "city",
                "country": "United States",
                "aliases": ["nyc", "new york city"],
                "coordinates": (40.7128, -74.0060)
            },
            "moscow": {
                "official_name": "Moscow",
                "type": "city",
                "country": "Russia",
                "aliases": [],
                "coordinates": (55.7558, 37.6176)
            },
            "kyiv": {
                "official_name": "Kyiv",
                "type": "city",
                "country": "Ukraine",
                "aliases": ["kiev"],
                "coordinates": (50.4501, 30.5234)
            },
            "manila": {
                "official_name": "Manila",
                "type": "city",
                "country": "Philippines",
                "aliases": [],
                "coordinates": (14.5995, 120.9842)
            },
            "hong kong": {
                "official_name": "Hong Kong",
                "type": "city",
                "country": "China",
                "aliases": [],
                "coordinates": (22.3193, 114.1694)
            },
            
            # Geographic Features & Regions
            "south china sea": {
                "official_name": "South China Sea",
                "type": "sea",
                "countries": ["China", "Taiwan", "Philippines", "Vietnam", "Malaysia"],
                "coordinates": (16.0, 114.0),
                "aliases": []
            },
            "east china sea": {
                "official_name": "East China Sea",
                "type": "sea",
                "countries": ["China", "Japan", "Taiwan", "South Korea"],
                "coordinates": (29.0, 125.0),
                "aliases": []
            },
            "taiwan strait": {
                "official_name": "Taiwan Strait",
                "type": "strait",
                "countries": ["China", "Taiwan"],
                "coordinates": (24.0, 119.5),
                "aliases": ["formosa strait"]
            },
            "korean peninsula": {
                "official_name": "Korean Peninsula",
                "type": "peninsula",
                "countries": ["South Korea", "North Korea"],
                "coordinates": (38.0, 127.5),
                "aliases": []
            },
            "spratly islands": {
                "official_name": "Spratly Islands",
                "type": "islands",
                "countries": ["China", "Taiwan", "Philippines", "Vietnam", "Malaysia"],
                "coordinates": (8.5, 111.5),
                "aliases": ["spratlys"]
            },
            "paracel islands": {
                "official_name": "Paracel Islands",
                "type": "islands",
                "countries": ["China", "Taiwan", "Vietnam"],
                "coordinates": (16.5, 112.0),
                "aliases": ["paracels"]
            },
            "east asia": {
                "official_name": "East Asia",
                "type": "region",
                "countries": ["China", "Japan", "South Korea", "North Korea", "Taiwan"],
                "coordinates": (35.0, 120.0),
                "aliases": []
            },
            "southeast asia": {
                "official_name": "Southeast Asia",
                "type": "region",
                "countries": ["Philippines", "Vietnam", "Thailand", "Indonesia", "Malaysia", "Singapore"],
                "coordinates": (10.0, 110.0),
                "aliases": ["south east asia"]
            },
            "indo-pacific": {
                "official_name": "Indo-Pacific",
                "type": "region",
                "countries": ["India", "China", "Japan", "Australia", "Indonesia", "Philippines"],
                "coordinates": (0.0, 120.0),
                "aliases": ["indo pacific"]
            },
            "middle east": {
                "official_name": "Middle East",
                "type": "region",
                "countries": ["Iran", "Iraq", "Syria", "Israel", "Saudi Arabia", "Turkey"],
                "coordinates": (29.0, 47.0),
                "aliases": []
            }
        }
    
    def extract_geographic_info(self, text: str) -> Dict[str, Any]:
        """Extract comprehensive geographic information from text"""
        if not text or len(text.strip()) < 10:
            return {
                'primary_country': None,
                'primary_city': None,
                'primary_region': None,
                'all_locations': [],
                'coordinates': None,
                'geographic_confidence': 0.0
            }
        
        # Clean and normalize text
        text = self._clean_text(text)
        
        # Extract locations using multiple methods
        locations = []
        
        # Method 1: spaCy NER (if available)
        if self.nlp:
            locations.extend(self._extract_with_spacy(text))
        
        # Method 2: Database matching
        locations.extend(self._extract_with_database(text))
        
        # Method 3: Pattern matching
        locations.extend(self._extract_with_patterns(text))
        
        # Deduplicate and rank locations
        locations = self._deduplicate_locations(locations)
        locations = self._rank_locations(locations, text)
        
        # Build final result
        result = self._build_result(locations)
        
        return result
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for processing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Limit text length for performance
        if len(text) > 5000:
            text = text[:5000]
        
        return text
    
    def _extract_with_spacy(self, text: str) -> List[GeographicLocation]:
        """Extract locations using spaCy NER"""
        locations = []
        
        try:
            doc = self.nlp(text)
            
            for ent in doc.ents:
                if ent.label_ in ['GPE', 'LOC']:  # Geopolitical entities and locations
                    name = ent.text.lower().strip()
                    
                    # Check if we have this location in our database
                    if name in self.locations_db:
                        loc_data = self.locations_db[name]
                        location = GeographicLocation(
                            name=loc_data['official_name'],
                            location_type=loc_data['type'],
                            confidence=0.8,  # High confidence for spaCy + database match
                            coordinates=loc_data.get('coordinates'),
                            country_code=loc_data.get('code'),
                            parent_location=loc_data.get('country')
                        )
                        locations.append(location)
                    else:
                        # Unknown location from spaCy
                        location = GeographicLocation(
                            name=ent.text.strip(),
                            location_type='unknown',
                            confidence=0.5  # Lower confidence for unknown
                        )
                        locations.append(location)
        
        except Exception as e:
            logger.debug(f"spaCy extraction failed: {e}")
        
        return locations
    
    def _extract_with_database(self, text: str) -> List[GeographicLocation]:
        """Extract locations using database matching"""
        locations = []
        text_lower = text.lower()
        
        for key, loc_data in self.locations_db.items():
            # Check main name
            if key in text_lower:
                location = GeographicLocation(
                    name=loc_data['official_name'],
                    location_type=loc_data['type'],
                    confidence=0.7,  # Good confidence for exact match
                    coordinates=loc_data.get('coordinates'),
                    country_code=loc_data.get('code'),
                    parent_location=loc_data.get('country')
                )
                locations.append(location)
                continue
            
            # Check aliases
            for alias in loc_data.get('aliases', []):
                if alias.lower() in text_lower:
                    location = GeographicLocation(
                        name=loc_data['official_name'],
                        location_type=loc_data['type'],
                        confidence=0.6,  # Slightly lower for alias match
                        coordinates=loc_data.get('coordinates'),
                        country_code=loc_data.get('code'),
                        parent_location=loc_data.get('country')
                    )
                    locations.append(location)
                    break
        
        return locations
    
    def _extract_with_patterns(self, text: str) -> List[GeographicLocation]:
        """Extract locations using regex patterns"""
        locations = []
        
        # Pattern for "City, Country" format
        city_country_pattern = r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b'
        matches = re.findall(city_country_pattern, text)
        
        for city, country in matches:
            city_lower = city.lower()
            country_lower = country.lower()
            
            # Check if we know these locations
            if city_lower in self.locations_db and country_lower in self.locations_db:
                city_data = self.locations_db[city_lower]
                if city_data.get('country', '').lower() == country_lower:
                    location = GeographicLocation(
                        name=city_data['official_name'],
                        location_type='city',
                        confidence=0.9,  # High confidence for pattern + database match
                        coordinates=city_data.get('coordinates'),
                        parent_location=country
                    )
                    locations.append(location)
        
        return locations
    
    def _deduplicate_locations(self, locations: List[GeographicLocation]) -> List[GeographicLocation]:
        """Remove duplicate locations"""
        seen = set()
        deduplicated = []
        
        for location in locations:
            key = (location.name.lower(), location.location_type)
            if key not in seen:
                seen.add(key)
                deduplicated.append(location)
        
        return deduplicated
    
    def _rank_locations(self, locations: List[GeographicLocation], text: str) -> List[GeographicLocation]:
        """Rank locations by relevance and confidence"""
        text_lower = text.lower()
        
        for location in locations:
            # Boost confidence based on frequency in text
            count = text_lower.count(location.name.lower())
            if count > 1:
                location.confidence = min(1.0, location.confidence + (count - 1) * 0.1)
            
            # Boost confidence for locations mentioned early in text
            first_pos = text_lower.find(location.name.lower())
            if first_pos != -1 and first_pos < len(text) * 0.3:  # First 30% of text
                location.confidence = min(1.0, location.confidence + 0.1)
        
        # Sort by confidence
        locations.sort(key=lambda x: x.confidence, reverse=True)
        
        return locations
    
    def _build_result(self, locations: List[GeographicLocation]) -> Dict[str, Any]:
        """Build final geographic extraction result"""
        if not locations:
            return {
                'primary_country': None,
                'primary_city': None,
                'primary_region': None,
                'all_locations': [],
                'coordinates': None,
                'geographic_confidence': 0.0
            }
        
        # Find primary country, city, and region
        primary_country = None
        primary_city = None
        primary_region = None
        
        for location in locations:
            if location.location_type == 'country' and not primary_country:
                primary_country = location.name
            elif location.location_type == 'city' and not primary_city:
                primary_city = location.name
            elif location.location_type in ['region', 'sea', 'strait'] and not primary_region:
                primary_region = location.name
        
        # Get coordinates from highest confidence location with coordinates
        coordinates = None
        for location in locations:
            if location.coordinates:
                coordinates = list(location.coordinates)
                break
        
        # Calculate overall confidence
        if locations:
            geographic_confidence = max(locations[0].confidence, 0.1)
            # Boost confidence based on number of locations found
            geographic_confidence = min(1.0, geographic_confidence + len(locations) * 0.05)
        else:
            geographic_confidence = 0.0
        
        # Build all locations list
        all_locations = [location.name for location in locations[:10]]  # Limit to top 10
        
        return {
            'primary_country': primary_country,
            'primary_city': primary_city,
            'primary_region': primary_region,
            'all_locations': all_locations,
            'coordinates': coordinates,
            'geographic_confidence': round(geographic_confidence, 3)
        }
    
    def tag_article(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Tag a single article with geographic information"""
        title = article.get('title', '')
        content = article.get('content', '')
        
        # Combine title and content for analysis
        combined_text = f"{title}. {content}"
        
        # Extract geographic information
        geo_info = self.extract_geographic_info(combined_text)
        
        # Add to article
        result = article.copy()
        result.update(geo_info)
        
        return result 