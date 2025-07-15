#!/usr/bin/env python3
"""
High-Speed Article Clustering Pipeline for OSINT System
Uses FAISS for similarity search and Supabase for metadata storage
"""

import logging
import numpy as np
import faiss
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime
import uuid
from collections import Counter, defaultdict
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class OSINTClusterer:
    def __init__(self, dimension: int = 384):
        """
        Initialize OSINT clustering system
        
        Args:
            dimension: Vector dimension (default 384 for MiniLM)
        """
        self.dimension = dimension
        self.index = None
        self.article_ids = []
        self.article_metadata = {}  # {article_id: {cluster_id, tags, entities, etc}}
        self.cluster_counter = 0
        self.similarity_threshold = 0.5  # More lenient similarity threshold for better clustering
        
    def build_faiss_index(self, articles: List[Dict[str, Any]]) -> bool:
        """
        Build FAISS index from articles with embeddings
        
        Args:
            articles: List of articles with embeddings from Supabase
            
        Returns:
            bool: Success status
        """
        try:
            logger.info(f"🔧 Building FAISS index for {len(articles)} articles")
            
            if not articles:
                logger.warning("⚠️ No articles provided for indexing")
                return False
            
            # Extract embeddings and metadata
            embeddings = []
            self.article_ids = []
            self.article_metadata = {}
            
            for article in articles:
                article_id = article.get('id') or article.get('article_id')
                embedding = article.get('embedding')
                
                if not article_id or not embedding:
                    logger.warning(f"⚠️ Skipping article with missing ID or embedding")
                    continue
                
                # Validate embedding
                if len(embedding) != self.dimension:
                    logger.warning(f"⚠️ Skipping article {article_id}: invalid embedding dimension {len(embedding)}")
                    continue
                
                # Convert to numpy array and normalize
                vector = np.array(embedding, dtype=np.float32)
                faiss.normalize_L2(vector.reshape(1, -1))
                
                embeddings.append(vector)
                self.article_ids.append(article_id)
                
                # Store metadata
                self.article_metadata[article_id] = {
                    'cluster_id': article.get('cluster_id'),
                    'tags': article.get('tags', []),
                    'entities': article.get('entities', []),
                    'title': article.get('title', ''),
                    'timestamp': article.get('published_at') or article.get('inserted_at'),
                    'embedding': embedding
                }
            
            if not embeddings:
                logger.error("❌ No valid embeddings found")
                return False
            
            # Create FAISS index
            vectors = np.vstack(embeddings)
            self.index = faiss.IndexFlatIP(self.dimension)
            self.index.add(vectors)
            
            logger.info(f"✅ FAISS index built with {len(self.article_ids)} vectors")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error building FAISS index: {e}")
            return False
    
    def find_nearest_neighbors(self, article_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors for an article
        
        Args:
            article_id: Article ID to find neighbors for
            k: Number of neighbors to return
            
        Returns:
            List of (neighbor_article_id, similarity_score) tuples
        """
        try:
            if article_id not in self.article_metadata:
                logger.warning(f"⚠️ Article {article_id} not found in index")
                return []
            
            # Get article index
            article_idx = self.article_ids.index(article_id)
            
            # Query FAISS
            query_vector = np.array(self.article_metadata[article_id]['embedding'], dtype=np.float32)
            faiss.normalize_L2(query_vector.reshape(1, -1))
            
            similarities, indices = self.index.search(query_vector.reshape(1, -1), min(k + 1, len(self.article_ids)))
            
            # Return results (excluding self)
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.article_ids) and idx != article_idx:
                    neighbor_id = self.article_ids[idx]
                    results.append((neighbor_id, float(similarity)))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error finding neighbors for {article_id}: {e}")
            return []
    
    def assign_clusters(self) -> Dict[str, int]:
        """
        Assign cluster IDs to unclustered articles based on topic similarity
        
        Returns:
            Dict mapping article_id to cluster_id
        """
        try:
            logger.info("🔄 Starting topic-based cluster assignment")
            
            # Track cluster assignments
            cluster_assignments = {}
            cluster_members = defaultdict(list)
            
            # Process unclustered articles
            unclustered_articles = [
                article_id for article_id in self.article_ids
                if not self.article_metadata[article_id].get('cluster_id')
            ]
            
            logger.info(f"📊 Found {len(unclustered_articles)} unclustered articles")
            
            # First pass: find existing clusters and their topics
            existing_clusters = defaultdict(list)
            cluster_topics = {}  # Track the main topic of each cluster
            
            for article_id in self.article_ids:
                cluster_id = self.article_metadata[article_id].get('cluster_id')
                if cluster_id:
                    existing_clusters[cluster_id].append(article_id)
                    # Track the main topic of this cluster
                    if cluster_id not in cluster_topics:
                        cluster_topics[cluster_id] = self._get_cluster_topic([article_id])
            
            # Find clusters that need more members (less than 3)
            small_clusters = {cid: members for cid, members in existing_clusters.items() 
                            if len(members) < 3}
            
            logger.info(f"📊 Found {len(small_clusters)} small clusters needing more members")
            
            # Process unclustered articles
            for article_id in unclustered_articles:
                # Find nearest neighbors
                neighbors = self.find_nearest_neighbors(article_id, k=15)  # More neighbors for better matching
                
                if not neighbors:
                    continue
                
                # Get the topic of this article
                article_topic = self._get_article_topic(article_id)
                
                # Find the best matching cluster based on topic similarity
                best_cluster = None
                best_similarity = 0
                best_topic_similarity = 0
                
                # First, try to join small clusters with similar topics
                for neighbor_id, similarity in neighbors:
                    if similarity >= self.similarity_threshold:
                        neighbor_cluster = self.article_metadata[neighbor_id].get('cluster_id')
                        if neighbor_cluster and neighbor_cluster in small_clusters:
                            # Check topic similarity
                            cluster_topic = cluster_topics.get(neighbor_cluster, "")
                            topic_similarity = self._calculate_topic_similarity(article_topic, cluster_topic)
                            
                            # Combined score: 70% topic similarity + 30% embedding similarity
                            combined_score = (topic_similarity * 0.7) + (similarity * 0.3)
                            
                            if combined_score > best_similarity:
                                best_similarity = combined_score
                                best_cluster = neighbor_cluster
                                best_topic_similarity = topic_similarity
                
                if best_cluster and best_topic_similarity > 0.1:  # More lenient topic similarity
                    # Join the small cluster
                    cluster_assignments[article_id] = best_cluster
                    cluster_members[best_cluster].append(article_id)
                    small_clusters[best_cluster].append(article_id)
                    
                    # Update cluster topic
                    cluster_topics[best_cluster] = self._get_cluster_topic(small_clusters[best_cluster])
                    
                    # Remove cluster from small_clusters if it now has 3+ members
                    if len(small_clusters[best_cluster]) >= 3:
                        del small_clusters[best_cluster]
                else:
                    # Look for any existing cluster with similar topic
                    for neighbor_id, similarity in neighbors:
                        if similarity >= self.similarity_threshold:
                            neighbor_cluster = self.article_metadata[neighbor_id].get('cluster_id')
                            if neighbor_cluster and neighbor_cluster not in small_clusters:
                                cluster_topic = cluster_topics.get(neighbor_cluster, "")
                                topic_similarity = self._calculate_topic_similarity(article_topic, cluster_topic)
                                
                                if topic_similarity > 0.2:  # More lenient threshold for existing clusters
                                    cluster_assignments[article_id] = neighbor_cluster
                                    cluster_members[neighbor_cluster].append(article_id)
                                    break
                    else:
                        # No suitable existing cluster found - will create new one in second pass
                        pass
            
            # Second pass: create new topic-based clusters for remaining articles
            remaining_unclustered = [aid for aid in unclustered_articles if aid not in cluster_assignments]
            
            if remaining_unclustered:
                logger.info(f"📊 Creating new topic-based clusters for {len(remaining_unclustered)} remaining articles")
                
                # Group articles by topic similarity
                topic_groups = self._group_articles_by_topic(remaining_unclustered)
                
                for topic, article_group in topic_groups.items():
                    if len(article_group) >= 3:
                        # Create new cluster for this topic
                        self.cluster_counter += 1
                        for aid in article_group:
                            cluster_assignments[aid] = self.cluster_counter
                            cluster_members[self.cluster_counter].append(aid)
                        cluster_topics[self.cluster_counter] = topic
                    else:
                        # Try to merge with existing small clusters
                        for aid in article_group:
                            best_cluster = self._find_best_small_cluster(aid, small_clusters, cluster_topics)
                            if best_cluster:
                                cluster_assignments[aid] = best_cluster
                                cluster_members[best_cluster].append(aid)
                                small_clusters[best_cluster].append(aid)
                            else:
                                # Create new small cluster
                                self.cluster_counter += 1
                                cluster_assignments[aid] = self.cluster_counter
                                cluster_members[self.cluster_counter].append(aid)
                                cluster_topics[self.cluster_counter] = self._get_article_topic(aid)
            
            logger.info(f"✅ Assigned {len(cluster_assignments)} articles to {len(cluster_members)} topic-based clusters")
            return cluster_assignments
            
        except Exception as e:
            logger.error(f"❌ Error assigning clusters: {e}")
            return {}
    
    def summarize_cluster(self, cluster_id: int, member_articles: List[str]) -> Dict[str, Any]:
        """
        Generate topic-based cluster summary from member articles
        
        Args:
            cluster_id: Cluster ID
            member_articles: List of article IDs in cluster
            
        Returns:
            Dict with cluster summary
        """
        try:
            # Collect all tags and entities
            all_tags = []
            all_entities = []
            titles = []
            
            for article_id in member_articles:
                metadata = self.article_metadata[article_id]
                all_tags.extend(metadata.get('tags', []))
                all_entities.extend(metadata.get('entities', []))
                titles.append(metadata.get('title', ''))
            
            # Get top tags and entities
            top_tags = [tag for tag, count in Counter(all_tags).most_common(10)]
            top_entities = [entity for entity, count in Counter(all_entities).most_common(10)]
            
            # Get cluster topic
            cluster_topic = self._get_cluster_topic(member_articles)
            
            # Generate topic-based summary
            if cluster_topic:
                summary = f"Topic Cluster: {cluster_topic}. "
                summary += f"Contains {len(member_articles)} articles. "
            else:
                summary = f"Cluster of {len(member_articles)} articles. "
            
            if top_entities:
                summary += f"Key entities: {', '.join(top_entities[:3])}. "
            if top_tags:
                summary += f"Common tags: {', '.join(top_tags[:3])}."
            
            # Find representative article (closest to cluster centroid)
            representative_article_id = self._find_representative_article(member_articles)
            
            return {
                'cluster_id': cluster_id,
                'top_tags': top_tags,
                'top_entities': top_entities,
                'summary': summary,
                'representative_article_id': representative_article_id,
                'member_count': len(member_articles),
                'topic': cluster_topic
            }
            
        except Exception as e:
            logger.error(f"❌ Error summarizing cluster {cluster_id}: {e}")
            return {
                'cluster_id': cluster_id,
                'top_tags': [],
                'top_entities': [],
                'summary': f"Cluster {cluster_id} with {len(member_articles)} articles",
                'representative_article_id': None,
                'member_count': len(member_articles)
            }
    
    def update_supabase_clusters(self, cluster_assignments: Dict[str, int]) -> bool:
        """
        Update cluster assignments in Supabase
        
        Args:
            cluster_assignments: Dict mapping article_id to cluster_id
            
        Returns:
            bool: Success status
        """
        try:
            from db.supabase_client import get_supabase_client
            
            logger.info(f"💾 Updating {len(cluster_assignments)} cluster assignments in Supabase")
            
            supabase = get_supabase_client()
            
            # Update articles table
            for article_id, cluster_id in cluster_assignments.items():
                try:
                    # Update cluster_id for article
                    result = supabase.table('articles').update({
                        'cluster_id': cluster_id
                    }).eq('id', article_id).execute()
                    
                    if not result.data:
                        logger.warning(f"⚠️ Failed to update cluster for article {article_id}")
                    
                except Exception as e:
                    logger.error(f"❌ Error updating article {article_id}: {e}")
            
            # Generate cluster summaries and update clusters table
            cluster_members = defaultdict(list)
            for article_id, cluster_id in cluster_assignments.items():
                cluster_members[cluster_id].append(article_id)
            
            for cluster_id, members in cluster_members.items():
                try:
                    summary = self.summarize_cluster(cluster_id, members)
                    
                    # Upsert cluster record
                    cluster_data = {
                        'cluster_id': cluster_id,
                        'top_tags': summary['top_tags'],
                        'top_entities': summary['top_entities'],
                        'summary': summary['summary'],
                        'representative_article_id': summary['representative_article_id'],
                        'member_count': summary['member_count'],
                        'updated_at': datetime.utcnow().isoformat()
                    }
                    
                    result = supabase.table('clusters').upsert(cluster_data).execute()
                    
                    if result.data:
                        logger.info(f"✅ Updated cluster {cluster_id} with {len(members)} members")
                    else:
                        logger.warning(f"⚠️ Failed to update cluster {cluster_id}")
                        
                except Exception as e:
                    logger.error(f"❌ Error updating cluster {cluster_id}: {e}")
            
            logger.info("✅ Cluster updates completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error updating Supabase clusters: {e}")
            return False
    
    def _get_article_topic(self, article_id: str) -> str:
        """Extract the main topic from an article's tags and entities"""
        try:
            metadata = self.article_metadata[article_id]
            tags = metadata.get('tags', [])
            entities = metadata.get('entities', [])
            
            # Combine tags and entities for topic analysis
            all_terms = tags + entities
            
            # Extract topic keywords (focus on GEO, CMD, ACT prefixes)
            topic_keywords = []
            for term in all_terms:
                if isinstance(term, str):
                    if term.startswith(('GEO:', 'CMD:', 'ACT:')):
                        topic_keywords.append(term.split(':', 1)[1] if ':' in term else term)
                    elif len(term) > 2:  # More lenient term length
                        topic_keywords.append(term)
            
            # Also include some title words for better topic matching
            title_words = metadata.get('title', '').split()[:5]  # First 5 words
            topic_keywords.extend([word for word in title_words if len(word) > 2])
            
            # Return the most common topic or a combination
            if topic_keywords:
                return ' '.join(topic_keywords[:3])  # Top 3 keywords
            else:
                return metadata.get('title', '')[:50]  # Fallback to title
                
        except Exception as e:
            logger.error(f"❌ Error getting article topic for {article_id}: {e}")
            return ""
    
    def _get_cluster_topic(self, article_ids: List[str]) -> str:
        """Extract the main topic from a cluster of articles"""
        try:
            all_topics = []
            for article_id in article_ids:
                topic = self._get_article_topic(article_id)
                if topic:
                    all_topics.append(topic)
            
            if not all_topics:
                return ""
            
            # Find the most common topic keywords
            all_words = []
            for topic in all_topics:
                all_words.extend(topic.split())
            
            # Count word frequencies
            word_counts = Counter(all_words)
            
            # Return the most common words as the cluster topic
            common_words = [word for word, count in word_counts.most_common(5) if len(word) > 2]
            return ' '.join(common_words)
            
        except Exception as e:
            logger.error(f"❌ Error getting cluster topic: {e}")
            return ""
    
    def _calculate_topic_similarity(self, topic1: str, topic2: str) -> float:
        """Calculate similarity between two topics"""
        try:
            if not topic1 or not topic2:
                return 0.0
            
            # Convert to sets of words
            words1 = set(topic1.lower().split())
            words2 = set(topic2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.error(f"❌ Error calculating topic similarity: {e}")
            return 0.0
    
    def _group_articles_by_topic(self, article_ids: List[str]) -> Dict[str, List[str]]:
        """Group articles by topic similarity"""
        try:
            topic_groups = {}
            
            for article_id in article_ids:
                article_topic = self._get_article_topic(article_id)
                
                # Find the best matching existing topic group
                best_group = None
                best_similarity = 0
                
                for group_topic, group_articles in topic_groups.items():
                    similarity = self._calculate_topic_similarity(article_topic, group_topic)
                    if similarity > best_similarity and similarity > 0.1:  # More lenient similarity threshold
                        best_similarity = similarity
                        best_group = group_topic
                
                if best_group:
                    # Add to existing group
                    topic_groups[best_group].append(article_id)
                else:
                    # Create new group
                    topic_groups[article_topic] = [article_id]
            
            return topic_groups
            
        except Exception as e:
            logger.error(f"❌ Error grouping articles by topic: {e}")
            return {}
    
    def _find_best_small_cluster(self, article_id: str, small_clusters: Dict, cluster_topics: Dict) -> Optional[int]:
        """Find the best small cluster for an article based on topic similarity"""
        try:
            article_topic = self._get_article_topic(article_id)
            best_cluster = None
            best_similarity = 0
            
            for cluster_id, members in small_clusters.items():
                cluster_topic = cluster_topics.get(cluster_id, "")
                similarity = self._calculate_topic_similarity(article_topic, cluster_topic)
                
                if similarity > best_similarity and similarity > 0.1:
                    best_similarity = similarity
                    best_cluster = cluster_id
            
            return best_cluster
            
        except Exception as e:
            logger.error(f"❌ Error finding best small cluster: {e}")
            return None
    
    def _find_representative_article(self, member_articles: List[str]) -> Optional[str]:
        """Find the article closest to the cluster centroid"""
        try:
            if not member_articles:
                return None
            
            if len(member_articles) == 1:
                return member_articles[0]
            
            # Calculate cluster centroid
            centroid_embedding = np.zeros(self.dimension)
            for article_id in member_articles:
                embedding = self.article_metadata[article_id].get('embedding', [])
                if embedding and len(embedding) == self.dimension:
                    centroid_embedding += np.array(embedding)
            
            centroid_embedding /= len(member_articles)
            
            # Find article closest to centroid
            best_article = None
            best_distance = float('inf')
            
            for article_id in member_articles:
                embedding = self.article_metadata[article_id].get('embedding', [])
                if embedding and len(embedding) == self.dimension:
                    distance = np.linalg.norm(np.array(embedding) - centroid_embedding)
                    if distance < best_distance:
                        best_distance = distance
                        best_article = article_id
            
            return best_article or member_articles[0]
            
        except Exception as e:
            logger.error(f"❌ Error finding representative article: {e}")
            return member_articles[0] if member_articles else None
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get statistics about current clustering"""
        if not self.article_metadata:
            return {}
        
        total_articles = len(self.article_metadata)
        clustered_articles = sum(1 for meta in self.article_metadata.values() if meta.get('cluster_id'))
        unclustered_articles = total_articles - clustered_articles
        
        cluster_counts = Counter(
            meta.get('cluster_id') for meta in self.article_metadata.values() 
            if meta.get('cluster_id')
        )
        
        return {
            'total_articles': total_articles,
            'clustered_articles': clustered_articles,
            'unclustered_articles': unclustered_articles,
            'total_clusters': len(cluster_counts),
            'avg_cluster_size': sum(cluster_counts.values()) / len(cluster_counts) if cluster_counts else 0,
            'largest_cluster': max(cluster_counts.values()) if cluster_counts else 0
        }

def run_clustering_pipeline() -> Dict[str, Any]:
    """
    Run the complete clustering pipeline
    
    Returns:
        Dict with pipeline results
    """
    try:
        logger.info("🚀 Starting OSINT Clustering Pipeline")
        start_time = datetime.now()
        
        # Load articles from Supabase
        from db.supabase_client import get_supabase_client
        
        logger.info("📥 Loading articles from Supabase...")
        supabase = get_supabase_client()
        result = supabase.table('articles').select('*').execute()
        articles = result.data
        
        if not articles:
            logger.warning("⚠️ No articles found in Supabase")
            return {'status': 'no_articles'}
        
        logger.info(f"📊 Loaded {len(articles)} articles from Supabase")
        
        # Initialize clusterer
        clusterer = OSINTClusterer()
        
        # Build FAISS index
        if not clusterer.build_faiss_index(articles):
            logger.error("❌ Failed to build FAISS index")
            return {'status': 'index_build_failed'}
        
        # Get initial stats
        initial_stats = clusterer.get_cluster_stats()
        logger.info(f"📈 Initial stats: {initial_stats}")
        
        # Assign clusters
        cluster_assignments = clusterer.assign_clusters()
        
        if not cluster_assignments:
            logger.warning("⚠️ No new cluster assignments made")
            return {'status': 'no_assignments', 'stats': initial_stats}
        
        # Update Supabase
        if not clusterer.update_supabase_clusters(cluster_assignments):
            logger.error("❌ Failed to update Supabase")
            return {'status': 'supabase_update_failed'}
        
        # Get final stats
        final_stats = clusterer.get_cluster_stats()
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Clustering pipeline completed in {duration:.2f} seconds")
        logger.info(f"📊 Final stats: {final_stats}")
        
        return {
            'status': 'success',
            'duration_seconds': duration,
            'articles_processed': len(articles),
            'new_assignments': len(cluster_assignments),
            'initial_stats': initial_stats,
            'final_stats': final_stats
        }
        
    except Exception as e:
        logger.error(f"❌ Clustering pipeline failed: {e}")
        return {'status': 'error', 'error': str(e)}

if __name__ == "__main__":
    """Run clustering pipeline"""
    results = run_clustering_pipeline()
    print(f"Pipeline Results: {results}") 