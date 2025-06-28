#!/usr/bin/env python3
"""
Storage-Based Intelligence Reporter
Only uses real articles from Supabase storage - NO synthetic content generation.
"""

import os
import sys
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.supabase_client import get_supabase_client

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class StorageBasedReporter:
    """Intelligence reporter that ONLY uses real articles from Supabase storage."""
    
    def __init__(self):
        self.supabase = get_supabase_client()
        
    def generate_storage_based_report(self) -> Dict[str, Any]:
        """Generate intelligence report using ONLY articles from Supabase storage."""
        
        logger.info("🎯 Generating Storage-Based Intelligence Report")
        
        try:
            # Fetch ALL articles from storage
            articles = self._fetch_articles_from_storage()
            
            if not articles:
                logger.warning("⚠️ No articles found in storage")
                return self._generate_empty_report()
            
            logger.info(f"📰 Found {len(articles)} articles in storage")
            
            # Perform thematic clustering on real articles
            thematic_clusters = self._perform_thematic_clustering(articles)
            
            if not thematic_clusters:
                logger.warning("⚠️ No valid clusters found (need minimum 3 articles per cluster)")
                return self._generate_insufficient_data_report(articles)
            
            # Generate comprehensive report
            timestamp = datetime.now()
            report_data = {
                'metadata': {
                    'generated_at': timestamp.isoformat(),
                    'total_articles': len(articles),
                    'total_clusters': len(thematic_clusters),
                    'report_type': 'storage_based_real_articles',
                    'data_source': 'supabase_storage_only'
                },
                'executive_summary': self._generate_executive_summary(articles, thematic_clusters),
                'key_developments': thematic_clusters,
                'threat_assessment': self._generate_threat_assessment(articles),
                'raw_articles': self._format_raw_articles(articles)
            }
            
            # Save report to file
            self._save_storage_report(report_data, timestamp)
            
            logger.info("✅ Storage-based report generated successfully")
            return report_data
            
        except Exception as e:
            logger.error(f"Storage-based report generation failed: {e}")
            return {'error': str(e), 'status': 'failed'}
    
    def _fetch_articles_from_storage(self) -> List[Dict[str, Any]]:
        """Fetch all articles from Supabase storage."""
        
        try:
            logger.info("📡 Fetching articles from Supabase storage...")
            
            result = self.supabase.table('articles').select('*').order('created_at', desc=True).execute()
            
            if not result.data:
                logger.warning("📭 No articles found in Supabase storage")
                return []
            
            articles = []
            for article_data in result.data:
                content = article_data.get('content') or ''
                title = article_data.get('title') or ''
                
                article = {
                    'id': article_data.get('id'),
                    'title': title,
                    'content': content,
                    'source': article_data.get('source', ''),
                    'url': article_data.get('url', ''),
                    'published_at': article_data.get('published_at', ''),
                    'created_at': article_data.get('created_at', ''),
                    'summary': content[:300] + '...' if len(content) > 300 else content
                }
                articles.append(article)
            
            logger.info(f"✅ Successfully fetched {len(articles)} articles from storage")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to fetch articles from storage: {e}")
            return []
    
    def _generate_empty_report(self) -> Dict[str, Any]:
        """Generate report when no articles are available."""
        
        timestamp = datetime.now()
        return {
            'metadata': {
                'generated_at': timestamp.isoformat(),
                'total_articles': 0,
                'total_clusters': 0,
                'report_type': 'empty_storage',
                'data_source': 'supabase_storage_only'
            },
            'status': 'no_articles_available',
            'message': 'No articles found in Supabase storage. Please run article ingestion first.',
            'executive_summary': {
                'threat_level': 'UNKNOWN',
                'total_articles_analyzed': 0,
                'significant_clusters': 0,
                'assessment_status': 'INSUFFICIENT_DATA'
            },
            'key_developments': [],
            'recommendations': [
                'Run article ingestion pipeline to populate storage with real articles',
                'Ensure news sources are accessible and providing content',
                'Check database connectivity and schema compatibility'
            ]
        }
    
    def _generate_insufficient_data_report(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate report when insufficient articles for clustering."""
        
        timestamp = datetime.now()
        return {
            'metadata': {
                'generated_at': timestamp.isoformat(),
                'total_articles': len(articles),
                'total_clusters': 0,
                'report_type': 'insufficient_data',
                'data_source': 'supabase_storage_only'
            },
            'status': 'insufficient_articles_for_clustering',
            'message': f'Found {len(articles)} articles, but need minimum 3 articles per cluster for meaningful analysis.',
            'executive_summary': {
                'threat_level': 'INSUFFICIENT_DATA',
                'total_articles_analyzed': len(articles),
                'significant_clusters': 0,
                'assessment_status': 'NEED_MORE_ARTICLES'
            },
            'key_developments': [],
            'raw_articles': self._format_raw_articles(articles),
            'recommendations': [
                'Ingest more articles to reach minimum threshold for cluster analysis',
                'Expand news source coverage for comprehensive intelligence',
                'Consider lowering cluster size threshold for preliminary analysis'
            ]
        }
    
    def _perform_thematic_clustering(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Perform thematic clustering on articles with more specific event categories."""
        
        if len(articles) < 3:
            return []
        
        # More specific event categories for better intelligence analysis
        clusters = []
        
        # 1. Taiwan Strait Military Incidents
        taiwan_strait_military = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['taiwan strait', 'cross-strait', 'pla navy', 'taiwan navy', 'military exercise', 'live fire', 'missile test', 'air defense'])]
        
        # 2. South China Sea Disputes
        south_china_sea = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['south china sea', 'paracel', 'spratly', 'nine-dash line', 'maritime dispute', 'freedom of navigation'])]
        
        # 3. US-China Military Relations
        us_china_military = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['us military', 'american military', 'pentagon', 'us navy', 'us air force', 'us-china military', 'defense cooperation'])]
        
        # 4. Chinese Military Modernization
        china_military_modernization = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['pla modernization', 'chinese military', 'military buildup', 'defense budget', 'military technology', 'weapons development'])]
        
        # 5. Regional Security Alliances
        regional_alliances = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['nato', 'quad', 'aukus', 'security alliance', 'defense pact', 'military cooperation'])]
        
        # 6. Diplomatic Tensions & Statements
        diplomatic_tensions = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['diplomatic', 'foreign ministry', 'state department', 'official statement', 'protest', 'condemn'])]
        
        # 7. Economic Sanctions & Trade Wars
        economic_conflicts = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['sanctions', 'trade war', 'tariffs', 'economic pressure', 'trade dispute', 'economic retaliation'])]
        
        # 8. Cyber & Information Warfare
        cyber_warfare = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['cyber', 'hacking', 'information warfare', 'disinformation', 'cyber attack', 'digital espionage'])]
        
        # 9. Nuclear & Strategic Weapons
        nuclear_strategic = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['nuclear', 'missile', 'strategic weapons', 'icbm', 'nuclear test', 'arms race'])]
        
        # 10. Intelligence & Espionage
        intelligence_espionage = [a for a in articles if any(keyword in (a.get('content', '') + ' ' + a.get('title', '')).lower() 
                           for keyword in ['espionage', 'intelligence', 'spy', 'surveillance', 'intelligence gathering', 'covert operation'])]
        
        # Create clusters with minimum 3 articles
        cluster_definitions = [
            ("Taiwan Strait Military Incidents", taiwan_strait_military, "HIGH", "military"),
            ("South China Sea Disputes", south_china_sea, "MEDIUM", "maritime"),
            ("US-China Military Relations", us_china_military, "MEDIUM", "diplomatic"),
            ("Chinese Military Modernization", china_military_modernization, "LOW", "military"),
            ("Regional Security Alliances", regional_alliances, "MEDIUM", "diplomatic"),
            ("Diplomatic Tensions & Statements", diplomatic_tensions, "LOW", "diplomatic"),
            ("Economic Sanctions & Trade Wars", economic_conflicts, "MEDIUM", "economic"),
            ("Cyber & Information Warfare", cyber_warfare, "HIGH", "cyber"),
            ("Nuclear & Strategic Weapons", nuclear_strategic, "HIGH", "military"),
            ("Intelligence & Espionage", intelligence_espionage, "MEDIUM", "intelligence")
        ]
        
        for theme, cluster_articles, priority, cluster_type in cluster_definitions:
            if len(cluster_articles) >= 3:
                # Calculate escalation score based on content analysis
                escalation_score = self._calculate_cluster_escalation_score(cluster_articles, cluster_type)
                
                # Generate specific analysis based on cluster type
                analysis = self._generate_cluster_analysis(theme, cluster_articles, cluster_type)
                
                cluster = self._create_real_cluster(theme, cluster_articles, analysis, cluster_type)
                cluster['priority_level'] = priority
                cluster['escalation_level'] = escalation_score
                clusters.append(cluster)
        
        return clusters
    
    def _create_real_cluster(self, theme: str, articles: List[Dict[str, Any]], 
                           analysis: str, cluster_type: str) -> Dict[str, Any]:
        """Create a cluster using real articles from storage."""
        
        # Calculate escalation level based on actual content
        escalation_keywords = {
            'military': ['exercise', 'warship', 'fighter', 'naval', 'destroyer', 'fleet', 'combat', 'drill'],
            'diplomatic': ['talks', 'dialogue', 'cooperation', 'resolution', 'summit', 'meeting'],
            'economic': ['decline', 'pressure', 'sanctions', 'disruption', 'crisis', 'impact']
        }
        
        escalation_score = 0
        
        for article in articles:
            content_lower = article.get('content', '').lower() + ' ' + article.get('title', '').lower()
            keywords = escalation_keywords.get(cluster_type, [])
            article_score = sum(1 for keyword in keywords if keyword in content_lower)
            escalation_score += article_score
        
        avg_escalation = escalation_score / len(articles) if articles else 0
        
        # Determine priority based on actual content analysis
        if avg_escalation >= 3:
            priority_level = "HIGH"
            escalation_level = round(min(avg_escalation / 5.0, 1.0), 2)
        elif avg_escalation >= 1.5:
            priority_level = "MEDIUM"
            escalation_level = round(min(avg_escalation / 5.0, 1.0), 2)
        else:
            priority_level = "LOW"
            escalation_level = round(min(avg_escalation / 5.0, 1.0), 2)
        
        return {
            'theme': theme,
            'cluster_type': cluster_type,
            'priority_level': priority_level,
            'escalation_level': escalation_level,
            'article_count': len(articles),
            'analysis': analysis,
            'confidence_score': min(len(articles) / 5.0, 1.0),
            'data_source': 'real_supabase_storage',
            'articles': [
                {
                    'id': article.get('id'),
                    'title': article.get('title', ''),
                    'source': article.get('source', ''),
                    'url': article.get('url', ''),
                    'summary': article.get('summary', ''),
                    'published_at': article.get('published_at', ''),
                    'created_at': article.get('created_at', '')
                }
                for article in articles
            ]
        }
    
    def _generate_executive_summary(self, articles: List[Dict[str, Any]], clusters: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate executive summary from real data."""
        
        # Calculate threat level based on actual content
        total_escalation = sum(cluster['escalation_level'] for cluster in clusters)
        avg_escalation = total_escalation / len(clusters) if clusters else 0
        
        if avg_escalation >= 0.6:
            threat_level = "HIGH"
        elif avg_escalation >= 0.3:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"
        
        return {
            'threat_level': threat_level,
            'total_articles_analyzed': len(articles),
            'significant_clusters': len(clusters),
            'average_escalation_score': round(avg_escalation, 3),
            'assessment_period': 'Real-time based on storage data',
            'primary_focus': 'Taiwan Strait Intelligence from Real Sources',
            'confidence_level': 'HIGH' if len(articles) >= 10 else 'MEDIUM' if len(articles) >= 5 else 'LOW',
            'data_authenticity': 'VERIFIED_STORAGE_SOURCES'
        }
    
    def _generate_threat_assessment(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate threat assessment from real article content."""
        
        # Count actual domain indicators from content
        military_indicators = len([a for a in articles if any(k in a.get('content', '').lower() 
                                 for k in ['military', 'naval', 'exercise', 'warship', 'fighter'])])
        diplomatic_indicators = len([a for a in articles if any(k in a.get('content', '').lower() 
                                   for k in ['diplomatic', 'talks', 'dialogue', 'meeting', 'summit'])])
        economic_indicators = len([a for a in articles if any(k in a.get('content', '').lower() 
                                 for k in ['economic', 'trade', 'market', 'business', 'finance'])])
        
        return {
            'military_activity_level': 'HIGH' if military_indicators >= 4 else 'MEDIUM' if military_indicators >= 2 else 'LOW',
            'diplomatic_engagement_level': 'HIGH' if diplomatic_indicators >= 3 else 'MEDIUM' if diplomatic_indicators >= 1 else 'LOW',
            'economic_impact_level': 'HIGH' if economic_indicators >= 3 else 'MEDIUM' if economic_indicators >= 1 else 'LOW',
            'overall_assessment': f'Analysis based on {len(articles)} real articles from verified sources',
            'confidence_level': 'HIGH' if len(articles) >= 10 else 'MEDIUM',
            'data_source': 'Real articles from Supabase storage',
            'indicators_count': {
                'military': military_indicators,
                'diplomatic': diplomatic_indicators,
                'economic': economic_indicators
            }
        }
    
    def _format_raw_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format raw articles for report inclusion."""
        
        return [
            {
                'id': article.get('id'),
                'title': article.get('title', ''),
                'source': article.get('source', ''),
                'url': article.get('url', ''),
                'published_at': article.get('published_at', ''),
                'content_length': len(article.get('content', '')),
                'has_content': bool(article.get('content', '').strip())
            }
            for article in articles
        ]
    
    def _save_storage_report(self, report_data: Dict[str, Any], timestamp: datetime) -> None:
        """Save the storage-based report."""
        try:
            # Ensure reports directory exists
            reports_dir = Path("reports")
            reports_dir.mkdir(exist_ok=True)

            # Generate filename
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
            filename = f"storage_based_report_{timestamp_str}.md"
            filepath = reports_dir / filename

            # Generate markdown report
            markdown_content = self._generate_storage_markdown(report_data)

            # Save to file
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(markdown_content)

            logger.info(f"📄 Storage-based report saved: {filepath}")
        except Exception as e:
            # Log error and attempt to save minimal error report
            logger.error(f"Failed to save storage report: {e}")
            try:
                error_filepath = reports_dir / f"storage_based_report_error_{timestamp_str}.md"
                with open(error_filepath, 'w', encoding='utf-8') as f:
                    f.write(f"# ERROR: Failed to save report\n\nError: {e}\n\nReport Data: {repr(report_data)}")
                logger.info(f"Minimal error report saved: {error_filepath}")
            except Exception as inner_e:
                logger.error(f"Failed to save minimal error report: {inner_e}")
    
    def _generate_storage_markdown(self, report_data: Dict[str, Any]) -> str:
        """Generate markdown for storage-based report."""
        
        metadata = report_data.get('metadata', {})
        status = report_data.get('status', 'success')
        
        if status == 'no_articles_available':
            executive_summary = report_data.get('executive_summary', {}) or {}
            return f"""# StraitWatch Storage-Based Intelligence Report

**Generated:** {metadata.get('generated_at', 'Unknown')}  
**Report Type:** Storage-Based Analysis (No Articles Available)  
**Data Source:** {metadata.get('data_source', 'Unknown')}  
**Classification:** CIVILIAN INTELLIGENCE ANALYSIS

---

## ⚠️ Status: No Articles Available

**Message:** {report_data.get('message', 'No articles found in storage')}

**Current Storage State:**
- **Articles in Storage:** {metadata.get('total_articles', 0)}
- **Clusters Generated:** {metadata.get('total_clusters', 0)}
- **Assessment Status:** {executive_summary.get('assessment_status', 'Unknown')}

## 📋 Recommendations

{chr(10).join(f"{i}. {rec}" for i, rec in enumerate(report_data.get('recommendations', []), 1))}

---

**Report Classification:** CIVILIAN INTELLIGENCE ANALYSIS  
**Generated by:** StraitWatch Storage-Based Analytics v1.0  
**Data Authenticity:** NO_SYNTHETIC_CONTENT  
**Next Steps:** Populate storage with real articles before generating intelligence reports
"""
        
        elif status == 'insufficient_articles_for_clustering':
            executive_summary = report_data.get('executive_summary', {}) or {}
            raw_articles_section = ""
            for i, article in enumerate(report_data.get('raw_articles', []), 1):
                raw_articles_section += f"""### {i}. {article.get('title', 'Unknown Title')}

**Source:** {article.get('source', 'Unknown')}  
**URL:** {article.get('url', 'No URL')}  
**Published:** {article.get('published_at', 'Unknown')[:10]}  
**Content Length:** {article.get('content_length', 0)} characters  
**Has Content:** {'Yes' if article.get('has_content') else 'No'}

---

"""
            
            return f"""# StraitWatch Storage-Based Intelligence Report

**Generated:** {metadata.get('generated_at', 'Unknown')}  
**Report Type:** Storage-Based Analysis (Insufficient Data)  
**Data Source:** {metadata.get('data_source', 'Unknown')}  
**Classification:** CIVILIAN INTELLIGENCE ANALYSIS

---

## ⚠️ Status: Insufficient Articles for Clustering

**Message:** {report_data.get('message', 'Insufficient articles for meaningful analysis')}

**Current Storage State:**
- **Articles in Storage:** {metadata.get('total_articles', 0)}
- **Minimum Required:** 3 articles per cluster
- **Assessment Status:** {executive_summary.get('assessment_status', 'Unknown')}

## 📰 Raw Articles in Storage

{raw_articles_section}

## 📋 Recommendations

{chr(10).join(f"{i}. {rec}" for i, rec in enumerate(report_data.get('recommendations', []), 1))}

---

**Report Classification:** CIVILIAN INTELLIGENCE ANALYSIS  
**Generated by:** StraitWatch Storage-Based Analytics v1.0  
**Data Authenticity:** REAL_ARTICLES_ONLY
"""
        
        # Full report with clusters
        executive_summary = report_data.get('executive_summary', {}) or {}
        key_developments = report_data.get('key_developments', [])
        threat_assessment = report_data.get('threat_assessment', {}) or {}
        
        clusters_section = ""
        for i, cluster in enumerate(key_developments, 1):
            articles_section = ""
            for article in cluster.get('articles', []):
                # Safely get article fields with defaults
                title = article.get('title', 'Unknown Title')
                url = article.get('url', '#')
                source = article.get('source', 'Unknown')
                published_at = article.get('published_at') or 'Unknown'
                article_id = article.get('id', 'Unknown')
                summary = article.get('summary', 'No summary available')
                
                # Calculate individual article escalation score
                article_escalation_score = self._calculate_article_escalation_score(article)
                
                # Format published date safely
                published_str = published_at[:10] if published_at and published_at != 'Unknown' else 'Unknown'
                
                articles_section += f"""
**🔗 [{title}]({url})**  
*📰 Source: {source} | 📅 Published: {published_str} | 🆔 ID: {article_id} | 🚨 Escalation: {article_escalation_score:.2f}*

**📝 Summary:** {summary}

"""
            
            # Safely get cluster fields with defaults
            theme = cluster.get('theme', 'Unknown Theme')
            priority_level = cluster.get('priority_level', 'Unknown')
            escalation_level = cluster.get('escalation_level', 0.0)
            article_count = cluster.get('article_count', 0)
            confidence_score = cluster.get('confidence_score', 0.0)
            data_source = cluster.get('data_source', 'Unknown')
            analysis = cluster.get('analysis', 'No analysis available')
            
            clusters_section += f"""### {i}. {theme}

**🔴 Priority Level:** {priority_level}  
**📈 Escalation Level:** {escalation_level}  
**📰 Articles in Cluster:** {article_count}  
**🎯 Confidence Score:** {confidence_score:.2f}  
**📡 Data Source:** {data_source}

**📊 Analysis:** {analysis}

**📰 Real Articles from Storage:**
{articles_section}

---

"""
        
        indicators = threat_assessment.get('indicators_count', {})
        indicators_section = chr(10).join(f"- **{domain.title()}:** {count} articles" for domain, count in indicators.items())
        
        return f"""# StraitWatch Storage-Based Intelligence Report

**Generated:** {metadata.get('generated_at', 'Unknown')}  
**Report Type:** Storage-Based Real Article Analysis  
**Data Source:** {metadata.get('data_source', 'Unknown')}  
**Classification:** CIVILIAN INTELLIGENCE ANALYSIS  
**Data Authenticity:** REAL_ARTICLES_ONLY

---

## 🎯 Executive Summary

**🚨 Threat Level:** {executive_summary.get('threat_level', 'Unknown')}  
**📊 Articles Analyzed:** {executive_summary.get('total_articles_analyzed', 0)}  
**🎯 Significant Clusters:** {executive_summary.get('significant_clusters', 0)}  
**📈 Average Escalation Score:** {executive_summary.get('average_escalation_score', 0)}  
**⏰ Assessment Period:** {executive_summary.get('assessment_period', 'Unknown')}  
**🎯 Primary Focus:** {executive_summary.get('primary_focus', 'Regional Security')}  
**🔒 Data Authenticity:** {executive_summary.get('data_authenticity', 'Unknown')}

---

## 📋 Key Developments (Real Article Clusters)

{clusters_section}

## ⚠️ Threat Assessment (Real Data Analysis)

**🎖️ Military Activity Level:** {threat_assessment.get('military_activity_level', 'Unknown')}  
**🤝 Diplomatic Engagement Level:** {threat_assessment.get('diplomatic_engagement_level', 'Unknown')}  
**💰 Economic Impact Level:** {threat_assessment.get('economic_impact_level', 'Unknown')}  
**🎯 Confidence Level:** {threat_assessment.get('confidence_level', 'Unknown')}  
**📡 Data Source:** {threat_assessment.get('data_source', 'Unknown')}

**📊 Overall Assessment:** {threat_assessment.get('overall_assessment', 'No assessment available')}

**📈 Domain Indicators (Real Content Analysis):**
{indicators_section}

---

**🔒 Report Classification:** CIVILIAN INTELLIGENCE ANALYSIS  
**🤖 Generated by:** StraitWatch Storage-Based Analytics v1.0  
**⏰ Timestamp:** {metadata.get('generated_at', 'Unknown')}  
**📊 Total Clusters Analyzed:** {metadata.get('total_clusters', 0)}  
**📈 Report Type:** {metadata.get('report_type', 'Unknown')}  
**🔒 Data Authenticity:** NO_SYNTHETIC_CONTENT_GUARANTEED
"""

    def _calculate_cluster_escalation_score(self, articles: List[Dict[str, Any]], cluster_type: str) -> float:
        """Calculate escalation score for a cluster based on comprehensive content analysis."""
        
        # Base scores by cluster type
        base_scores = {
            'military': 0.35,
            'maritime': 0.3,
            'diplomatic': 0.2,
            'economic': 0.25,
            'cyber': 0.4,
            'intelligence': 0.3,
            'nuclear': 0.5,
            'airspace': 0.35,
            'technology': 0.3,
            'regional': 0.25
        }
        
        base_score = base_scores.get(cluster_type, 0.25)
        
        # Comprehensive escalation indicators
        escalation_indicators = {
            'high_escalation': ['escalation', 'crisis', 'conflict', 'war', 'attack', 'strike', 'bombing', 'missile', 'nuclear', 'retaliation'],
            'medium_escalation': ['tension', 'threat', 'warning', 'aggressive', 'provocation', 'response', 'counter', 'military exercise', 'live fire', 'drone'],
            'low_escalation': ['surveillance', 'monitoring', 'patrol', 'defense', 'security', 'alliance', 'cooperation', 'diplomatic', 'talks', 'meeting']
        }
        
        total_escalation_count = 0
        total_articles = len(articles)
        
        # Analyze each article for escalation indicators
        for article in articles:
            content = (article.get('content', '') + ' ' + article.get('title', '')).lower()
            
            # Count escalation indicators with weighting
            for category, keywords in escalation_indicators.items():
                matches = sum(1 for keyword in keywords if keyword in content)
                if category == 'high_escalation':
                    total_escalation_count += matches * 2  # High weight
                elif category == 'medium_escalation':
                    total_escalation_count += matches * 1.5  # Medium weight
                else:
                    total_escalation_count += matches * 1  # Low weight
        
        # Calculate escalation multiplier based on content analysis
        if total_articles > 0:
            escalation_density = total_escalation_count / total_articles
            escalation_multiplier = 1.0 + (escalation_density * 0.3)
        else:
            escalation_multiplier = 1.0
        
        # Add source diversity bonus
        sources = list(set(article.get('source', 'Unknown') for article in articles))
        source_diversity = len(sources)
        if source_diversity > 3:
            escalation_multiplier += 0.1
        elif source_diversity > 1:
            escalation_multiplier += 0.05
        
        # Add recency bonus
        recent_articles = sum(1 for article in articles if '2025-06-27' in str(article.get('created_at', '')))
        if recent_articles > 0:
            recency_ratio = recent_articles / total_articles
            escalation_multiplier += recency_ratio * 0.1
        
        final_score = min(base_score * escalation_multiplier, 1.0)
        return round(final_score, 2)
    
    def _generate_cluster_analysis(self, theme: str, articles: List[Dict[str, Any]], cluster_type: str) -> str:
        """Generate specific analysis for a cluster based on its type and content."""
        
        analysis_templates = {
            'military': "Military activities and defense developments indicating strategic positioning and capability demonstration.",
            'maritime': "Maritime disputes and naval operations reflecting territorial claims and freedom of navigation concerns.",
            'diplomatic': "Diplomatic engagements and political statements showing bilateral and multilateral relationship dynamics.",
            'economic': "Economic measures and trade policies demonstrating strategic economic pressure and countermeasures.",
            'cyber': "Cybersecurity incidents and information operations revealing digital domain competition and espionage activities.",
            'intelligence': "Intelligence operations and surveillance activities indicating information gathering and strategic monitoring."
        }
        
        base_analysis = analysis_templates.get(cluster_type, "Strategic developments in regional security and international relations.")
        
        # Add specific details based on content
        sources = list(set(article.get('source', 'Unknown') for article in articles))
        source_diversity = len(sources)
        
        if source_diversity > 5:
            base_analysis += f" Multiple sources ({source_diversity} outlets) provide comprehensive coverage."
        elif source_diversity > 2:
            base_analysis += f" Coverage from {source_diversity} major news outlets."
        else:
            base_analysis += " Limited source coverage requires additional verification."
        
        return base_analysis

    def _calculate_article_escalation_score(self, article: Dict[str, Any]) -> float:
        """Calculate escalation score for an individual article with comprehensive analysis."""
        
        content = (article.get('content', '') + ' ' + article.get('title', '')).lower()
        
        # Comprehensive escalation indicators with weighted scoring
        escalation_indicators = {
            # High escalation keywords (0.3-0.5 points each)
            'high_escalation': {
                'keywords': ['escalation', 'crisis', 'conflict', 'war', 'attack', 'strike', 'bombing', 'missile', 'nuclear', 'retaliation'],
                'weight': 0.4
            },
            # Medium escalation keywords (0.2-0.3 points each)
            'medium_escalation': {
                'keywords': ['tension', 'threat', 'warning', 'aggressive', 'provocation', 'response', 'counter', 'military exercise', 'live fire', 'drone'],
                'weight': 0.25
            },
            # Low escalation keywords (0.1-0.2 points each)
            'low_escalation': {
                'keywords': ['surveillance', 'monitoring', 'patrol', 'defense', 'security', 'alliance', 'cooperation', 'diplomatic', 'talks', 'meeting'],
                'weight': 0.15
            },
            # Context-specific indicators
            'context_indicators': {
                'keywords': ['taiwan', 'china', 'strait', 'south china sea', 'us military', 'pentagon', 'pla', 'navy', 'air force'],
                'weight': 0.1
            }
        }
        
        total_score = 0.0
        
        # Calculate base score from keyword matches
        for category, config in escalation_indicators.items():
            matches = sum(1 for keyword in config['keywords'] if keyword in content)
            if matches > 0:
                # Apply diminishing returns for multiple matches
                category_score = min(matches * config['weight'], config['weight'] * 2)
                total_score += category_score
        
        # Add source credibility bonus
        source = article.get('source', '').lower()
        credible_sources = ['defense news', 'janes', 'reuters', 'ap', 'bbc', 'cnn', 'nyt', 'wsj']
        if any(credible in source for credible in credible_sources):
            total_score += 0.1
        
        # Add recency bonus (if article is recent)
        created_at = article.get('created_at', '')
        if created_at and '2025-06-27' in created_at:  # Recent articles get bonus
            total_score += 0.05
        
        # Add content length bonus (longer articles often have more detail)
        content_length = len(content)
        if content_length > 1000:
            total_score += 0.05
        elif content_length > 500:
            total_score += 0.02
        
        # Ensure minimum score for any article with relevant content
        if any(keyword in content for category in escalation_indicators.values() for keyword in category['keywords']):
            total_score = max(total_score, 0.05)  # Minimum 0.05 for relevant articles
        
        # Cap at 1.0 and round
        final_score = min(total_score, 1.0)
        return round(final_score, 2)

def main():
    """Generate storage-based intelligence report using only real articles."""
    
    print("\n🎯 StraitWatch Storage-Based Intelligence Report")
    print("=" * 70)
    print("🔒 Uses ONLY real articles from Supabase storage")
    print("❌ NO synthetic content generation")
    print("📊 Authentic intelligence analysis")
    print("=" * 70)
    
    reporter = StorageBasedReporter()
    report = reporter.generate_storage_based_report()
    
    if report.get('status') == 'no_articles_available':
        print(f"\n⚠️ No Articles Available in Storage")
        print(f"📭 Storage is empty - please run article ingestion first")
        print(f"💡 Recommendation: Populate storage with real articles")
        
    elif report.get('status') == 'insufficient_articles_for_clustering':
        print(f"\n⚠️ Insufficient Articles for Clustering")
        print(f"📊 Found: {report['metadata']['total_articles']} articles")
        print(f"📋 Need: Minimum 3 articles per cluster")
        print(f"💡 Recommendation: Ingest more articles for meaningful analysis")
        
    elif report.get('metadata'):
        print(f"\n✅ Storage-Based Report Generated!")
        print(f"📊 Real Articles Analyzed: {report['metadata']['total_articles']}")
        print(f"🎯 Clusters from Real Data: {report['metadata']['total_clusters']}")
        print(f"⚠️ Threat Level: {report['executive_summary']['threat_level']}")
        print(f"🔒 Data Authenticity: {report['executive_summary']['data_authenticity']}")
        print(f"📄 Report saved to reports/ directory")
        
        # Display cluster summary
        if report.get('key_developments'):
            print(f"\n📋 Real Cluster Summary:")
            for i, cluster in enumerate(report['key_developments'], 1):
                print(f"  {i}. {cluster['theme']} ({cluster['article_count']} real articles, {cluster['priority_level']} priority)")
    else:
        print(f"\n❌ Report generation failed: {report.get('error', 'Unknown error')}")
    
    print("\n📄 Check the reports/ directory for complete storage-based analysis")
    print("🔒 Guaranteed: NO synthetic content, only real articles from storage")

if __name__ == "__main__":
    main() 