#!/usr/bin/env python3
"""
Unified OSINT Article Clustering
Combines advanced clustering with NER integration for optimal geopolitical intelligence analysis.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import re
import spacy

# Core ML libraries
try:
    from sentence_transformers import SentenceTransformer
    import hdbscan
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import StandardScaler
    HAS_CLUSTERING = True
except ImportError:
    HAS_CLUSTERING = False
    logging.warning("Clustering libraries not available")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedOSINTClustering:
    """
    Unified clustering pipeline that combines:
    - SBERT embeddings for semantic similarity
    - Named Entity Recognition for actor/location grouping
    - Event-key clustering for temporal grouping
    - HDBSCAN for optimal cluster detection
    """
    
    def __init__(self):
        """Initialize the unified clustering pipeline."""
        self.text_encoder = None
        self.nlp = None
        self.entity_normalizer = EntityNormalizer()
        
        # Load models
        self._load_models()
        
    def _load_models(self):
        """Load required models for clustering."""
        try:
            self.text_encoder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("✅ Loaded SentenceTransformer for embeddings")
        except Exception as e:
            logger.error(f"Failed to load text encoder: {e}")
        
        try:
            self.nlp = spacy.load("en_core_web_sm")
            logger.info("✅ Loaded spaCy NER model")
        except OSError:
            logger.warning("⚠️ spaCy model not found, using fallback NER")
    
    def cluster_articles(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Main clustering pipeline that produces optimal clusters for intelligence analysis.
        
        Args:
            articles: List of article dictionaries with title, content, etc.
            
        Returns:
            Comprehensive clustering results with metadata
        """
        logger.info(f"🚀 Starting Unified OSINT Clustering for {len(articles)} articles")
        
        if not articles:
            return self._empty_clustering_result()
        
        # Step 1: Extract features (embeddings + NER + metadata)
        enriched_articles = self._extract_comprehensive_features(articles)
        
        # Step 2: Create semantic embeddings
        embeddings = self._create_semantic_embeddings(enriched_articles)
        
        # Step 3: Perform multi-layer clustering
        clusters = self._perform_multi_layer_clustering(enriched_articles, embeddings)
        
        # Step 4: Post-process and validate clusters
        validated_clusters = self._validate_and_enhance_clusters(clusters, enriched_articles)
        
        # Step 5: Generate comprehensive results
        results = self._generate_clustering_results(validated_clusters, enriched_articles, articles)
        
        logger.info(f"✅ Clustering complete: {len(results['clusters'])} clusters identified")
        return results
    
    def _extract_comprehensive_features(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract comprehensive features including NER, metadata, and text processing."""
        logger.info("📊 Extracting comprehensive features...")
        
        enriched_articles = []
        
        for i, article in enumerate(articles):
            if i % 25 == 0:
                logger.info(f"   Processing article {i+1}/{len(articles)}")
            
            enriched = article.copy()
            
            # Extract and combine text
            title = article.get('title', '')
            content = article.get('content', article.get('summary', ''))
            combined_text = f"{title}. {content}"
            
            # Extract named entities with NER
            entities = self._extract_named_entities(combined_text)
            normalized_entities = self.entity_normalizer.normalize_entities(entities)
            
            # Extract temporal features
            timestamps = self._extract_temporal_features(article)
            
            # Extract geopolitical features
            geopolitical_features = self._extract_geopolitical_features(combined_text, normalized_entities)
            
            # Calculate text metrics
            text_metrics = self._calculate_text_metrics(combined_text)
            
            # Add all features to article
            enriched.update({
                'combined_text': combined_text,
                'entities': normalized_entities,
                'timestamps': timestamps,
                'geopolitical_features': geopolitical_features,
                'text_metrics': text_metrics,
                'article_id': article.get('id', f"art_{i}")
            })
            
            enriched_articles.append(enriched)
        
        logger.info(f"✅ Feature extraction complete: {len(enriched_articles)} articles enriched")
        return enriched_articles
    
    def _extract_named_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities using spaCy with fallback."""
        entities = {
            'organizations': [],
            'persons': [],
            'countries': [],
            'locations': [],
            'events': [],
            'dates': []
        }
        
        if self.nlp and text:
            try:
                # Limit text length for performance
                doc = self.nlp(text[:5000])
                
                for ent in doc.ents:
                    entity_text = ent.text.strip()
                    if len(entity_text) < 2:
                        continue
                    
                    if ent.label_ in ['ORG']:
                        entities['organizations'].append(entity_text)
                    elif ent.label_ in ['PERSON']:
                        entities['persons'].append(entity_text)
                    elif ent.label_ in ['GPE']:
                        entities['countries'].append(entity_text)
                    elif ent.label_ in ['LOC', 'FAC']:
                        entities['locations'].append(entity_text)
                    elif ent.label_ in ['EVENT']:
                        entities['events'].append(entity_text)
                    elif ent.label_ in ['DATE', 'TIME']:
                        entities['dates'].append(entity_text)
            
            except Exception as e:
                logger.warning(f"NER extraction failed: {e}")
        
        # Fallback entity extraction
        if not any(entities.values()):
            entities = self._extract_entities_fallback(text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(dict.fromkeys(entities[key]))
        
        return entities
    
    def _extract_entities_fallback(self, text: str) -> Dict[str, List[str]]:
        """Fallback entity extraction using keyword patterns."""
        entities = {
            'organizations': [],
            'persons': [],
            'countries': [],
            'locations': [],
            'events': [],
            'dates': []
        }
        
        # Key countries for geopolitical analysis
        countries = [
            'United States', 'China', 'Taiwan', 'Russia', 'Iran', 'Israel', 
            'Japan', 'South Korea', 'North Korea', 'India', 'Pakistan', 
            'Ukraine', 'Turkey', 'Syria', 'Iraq', 'Afghanistan'
        ]
        
        # Key organizations
        organizations = [
            'Pentagon', 'State Department', 'CIA', 'FBI', 'NATO', 'UN', 
            'European Union', 'ASEAN', 'QUAD', 'AUKUS'
        ]
        
        text_lower = text.lower()
        
        for country in countries:
            if country.lower() in text_lower:
                entities['countries'].append(country)
        
        for org in organizations:
            if org.lower() in text_lower:
                entities['organizations'].append(org)
        
        return entities
    
    def _extract_temporal_features(self, article: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features from article."""
        timestamps = {}
        
        # Published date
        if 'published_at' in article:
            timestamps['published_at'] = article['published_at']
        elif 'date' in article:
            timestamps['published_at'] = article['date']
        
        # Scraped date
        if 'scraped_at' in article:
            timestamps['scraped_at'] = article['scraped_at']
        
        # Calculate recency score
        if timestamps.get('published_at'):
            try:
                if isinstance(timestamps['published_at'], str):
                    pub_date = pd.to_datetime(timestamps['published_at'])
                else:
                    pub_date = timestamps['published_at']
                
                now = datetime.now()
                if pub_date.tzinfo is None:
                    pub_date = pub_date.replace(tzinfo=now.tzinfo)
                
                days_old = (now - pub_date).days
                timestamps['days_old'] = days_old
                timestamps['recency_score'] = max(0, 1 - (days_old / 30))  # Decay over 30 days
                
            except Exception:
                timestamps['days_old'] = 999
                timestamps['recency_score'] = 0.0
        else:
            timestamps['days_old'] = 999
            timestamps['recency_score'] = 0.0
        
        return timestamps
    
    def _extract_geopolitical_features(self, text: str, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """Extract geopolitical features for intelligence analysis."""
        features = {
            'primary_region': None,
            'conflict_indicators': [],
            'tension_level': 0.0,
            'actor_count': 0,
            'location_count': 0
        }
        
        # Determine primary region
        asia_pacific_countries = ['China', 'Taiwan', 'Japan', 'South Korea', 'North Korea', 'Philippines', 'Vietnam']
        middle_east_countries = ['Iran', 'Israel', 'Syria', 'Iraq', 'Saudi Arabia', 'Turkey']
        europe_countries = ['Russia', 'Ukraine', 'Germany', 'France', 'UK', 'Poland']
        
        countries = entities.get('countries', [])
        
        if any(country in countries for country in asia_pacific_countries):
            features['primary_region'] = 'Asia-Pacific'
        elif any(country in countries for country in middle_east_countries):
            features['primary_region'] = 'Middle East'
        elif any(country in countries for country in europe_countries):
            features['primary_region'] = 'Europe'
        else:
            features['primary_region'] = 'Global'
        
        # Detect conflict indicators
        conflict_keywords = [
            'military', 'war', 'conflict', 'tension', 'crisis', 'threat', 'attack',
            'missile', 'nuclear', 'sanctions', 'diplomatic', 'escalation', 'dispute'
        ]
        
        text_lower = text.lower()
        for keyword in conflict_keywords:
            if keyword in text_lower:
                features['conflict_indicators'].append(keyword)
        
        # Calculate tension level
        features['tension_level'] = min(1.0, len(features['conflict_indicators']) / 5.0)
        
        # Count actors and locations
        features['actor_count'] = len(entities.get('organizations', [])) + len(entities.get('persons', []))
        features['location_count'] = len(entities.get('countries', [])) + len(entities.get('locations', []))
        
        return features
    
    def _calculate_text_metrics(self, text: str) -> Dict[str, Any]:
        """Calculate text-based metrics."""
        return {
            'length': len(text),
            'word_count': len(text.split()),
            'sentence_count': len([s for s in text.split('.') if s.strip()]),
            'complexity_score': min(1.0, len(text.split()) / 500)  # Normalize to 0-1
        }
    
    def _create_semantic_embeddings(self, articles: List[Dict[str, Any]]) -> np.ndarray:
        """Create semantic embeddings for articles."""
        logger.info("🔤 Creating semantic embeddings...")
        
        if not self.text_encoder:
            logger.warning("No text encoder available, using random embeddings")
            return np.random.randn(len(articles), 384)
        
        texts = []
        for article in articles:
            # Combine title and key content for embedding
            title = article.get('title', '')
            content = article.get('combined_text', '')
            
            # Use first 500 words for embedding to avoid token limits
            embedding_text = f"{title}. {content}"[:2000]
            texts.append(embedding_text)
        
        try:
            embeddings = self.text_encoder.encode(texts, show_progress_bar=True)
            logger.info(f"✅ Created embeddings: {embeddings.shape}")
            return embeddings
        except Exception as e:
            logger.error(f"Embedding creation failed: {e}")
            return np.random.randn(len(articles), 384)
    
    def _perform_multi_layer_clustering(self, articles: List[Dict[str, Any]], embeddings: np.ndarray) -> Dict[str, Any]:
        """Perform multi-layer clustering using semantic + entity-based approaches."""
        logger.info("🔄 Performing multi-layer clustering...")
        
        results = {}
        
        # Layer 1: Semantic clustering with HDBSCAN
        semantic_clusters = self._semantic_clustering(embeddings)
        results['semantic'] = semantic_clusters
        
        # Layer 2: Entity-based clustering
        entity_clusters = self._entity_based_clustering(articles)
        results['entity'] = entity_clusters
        
        # Layer 3: Temporal clustering
        temporal_clusters = self._temporal_clustering(articles)
        results['temporal'] = temporal_clusters
        
        # Layer 4: Combine and optimize clusters
        combined_clusters = self._combine_clustering_layers(results, articles)
        results['combined'] = combined_clusters
        
        return results
    
    def _semantic_clustering(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """Perform semantic clustering using HDBSCAN."""
        if not HAS_CLUSTERING:
            return {'labels': np.zeros(len(embeddings)), 'n_clusters': 1}
        
        try:
            # Use HDBSCAN for optimal cluster detection
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=2,
                min_samples=1,
                metric='cosine',
                cluster_selection_epsilon=0.3
            )
            
            cluster_labels = clusterer.fit_predict(embeddings)
            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            
            logger.info(f"   Semantic clustering: {n_clusters} clusters, {np.sum(cluster_labels == -1)} noise points")
            
            return {
                'labels': cluster_labels,
                'n_clusters': n_clusters,
                'noise_points': np.sum(cluster_labels == -1),
                'probabilities': getattr(clusterer, 'probabilities_', None)
            }
        
        except Exception as e:
            logger.warning(f"Semantic clustering failed: {e}")
            return {'labels': np.zeros(len(embeddings)), 'n_clusters': 1}
    
    def _entity_based_clustering(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster articles based on shared entities (actors, locations)."""
        logger.info("   Entity-based clustering...")
        
        # Create entity signatures for each article
        entity_signatures = []
        for article in articles:
            entities = article.get('entities', {})
            
            # Create signature from key entities
            countries = set(entities.get('countries', []))
            orgs = set(entities.get('organizations', []))
            locations = set(entities.get('locations', []))
            
            signature = {
                'countries': countries,
                'organizations': orgs,
                'locations': locations,
                'signature_hash': hash(tuple(sorted(countries | orgs | locations)))
            }
            entity_signatures.append(signature)
        
        # Group articles with similar entity signatures
        signature_groups = defaultdict(list)
        for i, sig in enumerate(entity_signatures):
            signature_groups[sig['signature_hash']].append(i)
        
        # Create cluster labels
        labels = np.full(len(articles), -1)
        cluster_id = 0
        
        for indices in signature_groups.values():
            if len(indices) >= 2:  # Minimum cluster size
                for idx in indices:
                    labels[idx] = cluster_id
                cluster_id += 1
        
        n_clusters = len([g for g in signature_groups.values() if len(g) >= 2])
        
        logger.info(f"   Entity clustering: {n_clusters} clusters")
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'signatures': entity_signatures
        }
    
    def _temporal_clustering(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Cluster articles based on temporal proximity."""
        logger.info("   Temporal clustering...")
        
        # Extract publication dates
        dates = []
        for article in articles:
            timestamps = article.get('timestamps', {})
            pub_date = timestamps.get('published_at')
            
            if pub_date:
                try:
                    if isinstance(pub_date, str):
                        dates.append(pd.to_datetime(pub_date))
                    else:
                        dates.append(pub_date)
                except:
                    dates.append(None)
            else:
                dates.append(None)
        
        # Group by time windows (24-hour periods)
        labels = np.full(len(articles), -1)
        cluster_id = 0
        
        for i, date1 in enumerate(dates):
            if date1 is None or labels[i] != -1:
                continue
            
            # Find articles within 24 hours
            cluster_indices = [i]
            for j, date2 in enumerate(dates):
                if i != j and date2 is not None and labels[j] == -1:
                    if abs((date1 - date2).total_seconds()) <= 86400:  # 24 hours
                        cluster_indices.append(j)
            
            if len(cluster_indices) >= 2:
                for idx in cluster_indices:
                    labels[idx] = cluster_id
                cluster_id += 1
        
        n_clusters = cluster_id
        
        logger.info(f"   Temporal clustering: {n_clusters} clusters")
        
        return {
            'labels': labels,
            'n_clusters': n_clusters,
            'dates': dates
        }
    
    def _combine_clustering_layers(self, layer_results: Dict[str, Any], articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple clustering layers to create optimal final clusters."""
        logger.info("   Combining clustering layers...")
        
        n_articles = len(articles)
        
        # Get labels from each layer
        semantic_labels = layer_results['semantic']['labels']
        entity_labels = layer_results['entity']['labels']
        temporal_labels = layer_results['temporal']['labels']
        
        # Create composite clustering by finding consensus
        final_labels = np.full(n_articles, -1)
        cluster_id = 0
        
        # Group articles that cluster together in multiple layers
        for i in range(n_articles):
            if final_labels[i] != -1:
                continue
            
            # Find articles that should be in same cluster
            cluster_candidates = set([i])
            
            # Check semantic similarity
            if semantic_labels[i] != -1:
                semantic_cluster = np.where(semantic_labels == semantic_labels[i])[0]
                cluster_candidates.update(semantic_cluster)
            
            # Check entity similarity
            if entity_labels[i] != -1:
                entity_cluster = np.where(entity_labels == entity_labels[i])[0]
                cluster_candidates.update(entity_cluster)
            
            # Check temporal similarity
            if temporal_labels[i] != -1:
                temporal_cluster = np.where(temporal_labels == temporal_labels[i])[0]
                cluster_candidates.update(temporal_cluster)
            
            # Only create cluster if we have multiple articles
            cluster_candidates = [idx for idx in cluster_candidates if final_labels[idx] == -1]
            
            if len(cluster_candidates) >= 2:
                for idx in cluster_candidates:
                    final_labels[idx] = cluster_id
                cluster_id += 1
        
        n_clusters = cluster_id
        
        logger.info(f"   Combined clustering: {n_clusters} final clusters")
        
        return {
            'labels': final_labels,
            'n_clusters': n_clusters,
            'layer_agreement': self._calculate_layer_agreement(semantic_labels, entity_labels, temporal_labels)
        }
    
    def _calculate_layer_agreement(self, *label_arrays) -> float:
        """Calculate agreement between clustering layers."""
        if len(label_arrays) < 2:
            return 1.0
        
        agreements = []
        for i in range(len(label_arrays)):
            for j in range(i + 1, len(label_arrays)):
                # Calculate adjusted rand index or similar metric
                # For simplicity, use basic agreement
                labels1, labels2 = label_arrays[i], label_arrays[j]
                agreement = np.mean([(labels1[k] != -1 and labels2[k] != -1 and 
                                    ((labels1[k] == labels1[m]) == (labels2[k] == labels2[m])))
                                   for k in range(len(labels1)) for m in range(k + 1, len(labels1))])
                agreements.append(agreement)
        
        return np.mean(agreements) if agreements else 0.0
    
    def _validate_and_enhance_clusters(self, cluster_results: Dict[str, Any], articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate and enhance cluster quality."""
        logger.info("✅ Validating and enhancing clusters...")
        
        combined_results = cluster_results['combined']
        labels = combined_results['labels']
        
        # Group articles by cluster
        clusters = defaultdict(list)
        for i, label in enumerate(labels):
            if label != -1:
                clusters[label].append(i)
        
        # Enhance each cluster with metadata
        enhanced_clusters = {}
        for cluster_id, article_indices in clusters.items():
            cluster_articles = [articles[i] for i in article_indices]
            
            enhanced_clusters[cluster_id] = {
                'article_indices': article_indices,
                'article_count': len(article_indices),
                'articles': cluster_articles,
                'metadata': self._generate_cluster_metadata(cluster_articles),
                'quality_score': self._calculate_cluster_quality(cluster_articles)
            }
        
        return {
            'clusters': enhanced_clusters,
            'n_clusters': len(enhanced_clusters),
            'total_clustered': sum(len(indices) for indices in clusters.values()),
            'noise_points': np.sum(labels == -1)
        }
    
    def _generate_cluster_metadata(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive metadata for a cluster."""
        metadata = {
            'primary_actors': [],
            'primary_locations': [],
            'primary_countries': [],
            'event_type': 'Unknown',
            'time_range': {},
            'escalation_level': 0.0,
            'significance_score': 0.0
        }
        
        # Aggregate entities
        all_countries = []
        all_orgs = []
        all_locations = []
        all_dates = []
        escalation_scores = []
        
        for article in articles:
            entities = article.get('entities', {})
            all_countries.extend(entities.get('countries', []))
            all_orgs.extend(entities.get('organizations', []))
            all_locations.extend(entities.get('locations', []))
            
            # Collect dates
            timestamps = article.get('timestamps', {})
            if timestamps.get('published_at'):
                all_dates.append(timestamps['published_at'])
            
            # Collect escalation scores
            if 'escalation_score' in article:
                escalation_scores.append(article['escalation_score'])
        
        # Find most common entities
        if all_countries:
            metadata['primary_countries'] = [item for item, count in Counter(all_countries).most_common(3)]
        if all_orgs:
            metadata['primary_actors'] = [item for item, count in Counter(all_orgs).most_common(3)]
        if all_locations:
            metadata['primary_locations'] = [item for item, count in Counter(all_locations).most_common(3)]
        
        # Calculate time range
        if all_dates:
            try:
                dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in all_dates]
                metadata['time_range'] = {
                    'start': min(dates).isoformat(),
                    'end': max(dates).isoformat(),
                    'span_days': (max(dates) - min(dates)).days
                }
            except:
                pass
        
        # Calculate average escalation
        if escalation_scores:
            metadata['escalation_level'] = np.mean(escalation_scores)
        
        # Infer event type
        metadata['event_type'] = self._infer_event_type(articles)
        
        # Calculate significance
        metadata['significance_score'] = self._calculate_significance_score(articles, metadata)
        
        return metadata
    
    def _infer_event_type(self, articles: List[Dict[str, Any]]) -> str:
        """Infer the type of event from cluster articles."""
        all_text = ' '.join([article.get('combined_text', '') for article in articles]).lower()
        
        event_patterns = {
            'Military Exercise': ['exercise', 'drill', 'training', 'maneuver'],
            'Diplomatic Crisis': ['diplomatic', 'embassy', 'ambassador', 'summit', 'negotiation'],
            'Military Conflict': ['war', 'battle', 'combat', 'attack', 'strike', 'missile'],
            'Economic Sanctions': ['sanctions', 'embargo', 'tariff', 'trade war'],
            'Cyber Attack': ['cyber', 'hack', 'malware', 'ransomware'],
            'Nuclear Issue': ['nuclear', 'uranium', 'plutonium', 'reactor'],
            'Territorial Dispute': ['territory', 'border', 'sovereignty', 'island', 'strait']
        }
        
        for event_type, keywords in event_patterns.items():
            if any(keyword in all_text for keyword in keywords):
                return event_type
        
        return 'General News'
    
    def _calculate_significance_score(self, articles: List[Dict[str, Any]], metadata: Dict[str, Any]) -> float:
        """Calculate significance score for the cluster."""
        score = 0.0
        
        # Article count factor
        score += min(0.3, len(articles) / 10)
        
        # Escalation level factor
        score += metadata.get('escalation_level', 0) * 0.4
        
        # Actor importance factor
        important_actors = ['Pentagon', 'State Department', 'White House', 'Kremlin', 'Beijing']
        if any(actor in metadata.get('primary_actors', []) for actor in important_actors):
            score += 0.2
        
        # Geographic importance factor
        critical_regions = ['Taiwan', 'South China Sea', 'Ukraine', 'Middle East']
        if any(region in str(metadata.get('primary_locations', [])) for region in critical_regions):
            score += 0.1
        
        return min(1.0, score)
    
    def _calculate_cluster_quality(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate quality score for cluster coherence."""
        if len(articles) < 2:
            return 0.0
        
        # Check entity overlap
        entity_overlap = self._calculate_entity_overlap(articles)
        
        # Check temporal coherence
        temporal_coherence = self._calculate_temporal_coherence(articles)
        
        # Check semantic coherence (simplified)
        semantic_coherence = 0.7  # Placeholder - would need embeddings
        
        return (entity_overlap + temporal_coherence + semantic_coherence) / 3.0
    
    def _calculate_entity_overlap(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate entity overlap between articles in cluster."""
        all_entities = []
        
        for article in articles:
            entities = article.get('entities', {})
            article_entities = set()
            for entity_list in entities.values():
                article_entities.update(entity_list)
            all_entities.append(article_entities)
        
        if not all_entities:
            return 0.0
        
        # Calculate pairwise Jaccard similarity
        similarities = []
        for i in range(len(all_entities)):
            for j in range(i + 1, len(all_entities)):
                intersection = len(all_entities[i] & all_entities[j])
                union = len(all_entities[i] | all_entities[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_temporal_coherence(self, articles: List[Dict[str, Any]]) -> float:
        """Calculate temporal coherence of articles in cluster."""
        dates = []
        for article in articles:
            timestamps = article.get('timestamps', {})
            if timestamps.get('published_at'):
                try:
                    date = pd.to_datetime(timestamps['published_at'])
                    dates.append(date)
                except:
                    continue
        
        if len(dates) < 2:
            return 1.0
        
        # Calculate date span
        date_span = (max(dates) - min(dates)).days
        
        # Coherence decreases with larger spans
        return max(0.0, 1.0 - (date_span / 7.0))  # 7-day window
    
    def _generate_clustering_results(self, validated_clusters: Dict[str, Any], 
                                   enriched_articles: List[Dict[str, Any]], 
                                   original_articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate final comprehensive clustering results."""
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'total_articles': len(original_articles),
            'clustered_articles': validated_clusters['total_clustered'],
            'noise_points': validated_clusters['noise_points'],
            'n_clusters': validated_clusters['n_clusters'],
            'clusters': [],
            'summary': {}
        }
        
        # Process each cluster
        for cluster_id, cluster_data in validated_clusters['clusters'].items():
            cluster_result = {
                'cluster_id': cluster_id,
                'article_count': cluster_data['article_count'],
                'quality_score': cluster_data['quality_score'],
                'metadata': cluster_data['metadata'],
                'articles': []
            }
            
            # Add article details
            for article in cluster_data['articles']:
                article_summary = {
                    'id': article.get('id', article.get('article_id')),
                    'title': article.get('title', ''),
                    'url': article.get('url', ''),
                    'source': article.get('source', ''),
                    'published_at': article.get('timestamps', {}).get('published_at'),
                    'escalation_score': article.get('escalation_score', 0.0),
                    'entities': article.get('entities', {}),
                    'geopolitical_features': article.get('geopolitical_features', {})
                }
                cluster_result['articles'].append(article_summary)
            
            results['clusters'].append(cluster_result)
        
        # Generate summary statistics
        results['summary'] = self._generate_summary_statistics(results)
        
        return results
    
    def _generate_summary_statistics(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary statistics for clustering results."""
        clusters = results['clusters']
        
        if not clusters:
            return {}
        
        # Calculate statistics
        cluster_sizes = [c['article_count'] for c in clusters]
        quality_scores = [c['quality_score'] for c in clusters]
        escalation_levels = []
        
        for cluster in clusters:
            cluster_escalation = [a.get('escalation_score', 0) for a in cluster['articles']]
            if cluster_escalation:
                escalation_levels.append(np.mean(cluster_escalation))
        
        summary = {
            'clustering_efficiency': results['clustered_articles'] / results['total_articles'],
            'average_cluster_size': np.mean(cluster_sizes),
            'largest_cluster_size': max(cluster_sizes),
            'average_quality_score': np.mean(quality_scores),
            'high_quality_clusters': len([q for q in quality_scores if q > 0.7]),
            'average_escalation_level': np.mean(escalation_levels) if escalation_levels else 0.0,
            'high_escalation_clusters': len([e for e in escalation_levels if e > 0.7])
        }
        
        return summary
    
    def _empty_clustering_result(self) -> Dict[str, Any]:
        """Return empty clustering result."""
        return {
            'timestamp': datetime.now().isoformat(),
            'total_articles': 0,
            'clustered_articles': 0,
            'noise_points': 0,
            'n_clusters': 0,
            'clusters': [],
            'summary': {}
        }


class EntityNormalizer:
    """Normalize entity names for better clustering."""
    
    def __init__(self):
        self.country_mappings = {
            'US': 'United States',
            'USA': 'United States',
            'America': 'United States',
            'PRC': 'China',
            'ROC': 'Taiwan',
            'DPRK': 'North Korea',
            'ROK': 'South Korea'
        }
    
    def normalize_entities(self, entities: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Normalize entity names."""
        normalized = {}
        
        for entity_type, entity_list in entities.items():
            normalized[entity_type] = []
            for entity in entity_list:
                normalized_entity = self._normalize_single_entity(entity, entity_type)
                if normalized_entity:
                    normalized[entity_type].append(normalized_entity)
        
        return normalized
    
    def _normalize_single_entity(self, entity: str, entity_type: str) -> Optional[str]:
        """Normalize a single entity."""
        if entity_type == 'countries':
            return self.country_mappings.get(entity, entity)
        
        return entity


# Main function for external use
def cluster_articles(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to cluster articles using the unified pipeline.
    
    Args:
        articles: List of article dictionaries
        
    Returns:
        Comprehensive clustering results
    """
    clusterer = UnifiedOSINTClustering()
    return clusterer.cluster_articles(articles)