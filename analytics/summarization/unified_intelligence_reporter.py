#!/usr/bin/env python3
"""
Unified Intelligence Reporter
Combines all summarization capabilities to generate comprehensive intelligence reports.
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class UnifiedIntelligenceReporter:
    """
    Unified intelligence reporter that combines:
    - Event summarization
    - Significance detection
    - Escalation analysis
    - Cluster analysis
    - Trend identification
    - Comprehensive reporting
    """
    
    def __init__(self):
        """Initialize the unified intelligence reporter."""
        self.report_templates = self._load_report_templates()
        
    def _load_report_templates(self) -> Dict[str, str]:
        """Load report templates for different types of intelligence reports."""
        return {
            'executive_summary': """
# EXECUTIVE INTELLIGENCE SUMMARY
**Generated:** {timestamp}
**Analysis Period:** {time_period}
**Total Articles Analyzed:** {total_articles}

## KEY FINDINGS
{key_findings}

## THREAT ASSESSMENT
{threat_assessment}

## RECOMMENDATIONS
{recommendations}
""",
            'cluster_analysis': """
## CLUSTER ANALYSIS: {cluster_title}
**Event Type:** {event_type}
**Escalation Level:** {escalation_level}/10
**Significance Score:** {significance_score}/10
**Article Count:** {article_count}
**Time Span:** {time_span}

### Primary Actors
{primary_actors}

### Key Locations
{key_locations}

### Summary
{cluster_summary}

### Individual Articles
{article_details}
""",
            'threat_matrix': """
## THREAT ASSESSMENT MATRIX

| Threat Level | Event Count | Primary Regions | Key Indicators |
|-------------|-------------|-----------------|----------------|
{threat_matrix_rows}
""",
            'trend_analysis': """
## TREND ANALYSIS

### Escalation Trends
{escalation_trends}

### Geographic Patterns
{geographic_patterns}

### Temporal Patterns
{temporal_patterns}

### Emerging Threats
{emerging_threats}
"""
        }
    
    def generate_comprehensive_report(self, 
                                    articles: List[Dict[str, Any]], 
                                    clustering_results: Dict[str, Any],
                                    escalation_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a comprehensive intelligence report combining all analytics.
        
        Args:
            articles: List of analyzed articles
            clustering_results: Results from clustering analysis
            escalation_predictions: Results from escalation prediction
            
        Returns:
            Comprehensive intelligence report
        """
        logger.info(f"📊 Generating comprehensive intelligence report for {len(articles)} articles...")
        
        # Generate report components
        executive_summary = self._generate_executive_summary(articles, clustering_results, escalation_predictions)
        cluster_analyses = self._generate_cluster_analyses(clustering_results, escalation_predictions)
        threat_assessment = self._generate_threat_assessment(articles, escalation_predictions)
        trend_analysis = self._generate_trend_analysis(articles, escalation_predictions)
        individual_articles = self._generate_individual_article_analysis(articles, escalation_predictions)
        
        # Combine into comprehensive report
        report = {
            'metadata': {
                'generated_at': datetime.now().isoformat(),
                'total_articles': len(articles),
                'analysis_period': self._get_analysis_period(articles),
                'report_version': '2.0',
                'confidence_level': self._calculate_overall_confidence(escalation_predictions)
            },
            'executive_summary': executive_summary,
            'threat_assessment': threat_assessment,
            'cluster_analysis': cluster_analyses,
            'trend_analysis': trend_analysis,
            'individual_articles': individual_articles,
            'recommendations': self._generate_recommendations(articles, clustering_results, escalation_predictions),
            'appendices': {
                'methodology': self._generate_methodology_appendix(),
                'data_sources': self._generate_data_sources_appendix(articles),
                'confidence_metrics': self._generate_confidence_metrics(escalation_predictions)
            }
        }
        
        # Generate formatted text report
        text_report = self._format_text_report(report)
        report['formatted_report'] = text_report
        
        logger.info("✅ Comprehensive intelligence report generated")
        return report
    
    def _generate_executive_summary(self, 
                                  articles: List[Dict[str, Any]], 
                                  clustering_results: Dict[str, Any],
                                  escalation_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate executive summary section."""
        
        # Calculate key metrics
        high_escalation_count = len([p for p in escalation_predictions if p.get('escalation_score', 0) > 0.7])
        medium_escalation_count = len([p for p in escalation_predictions if 0.4 <= p.get('escalation_score', 0) <= 0.7])
        
        # Identify top threats
        top_threats = sorted(
            [(i, p) for i, p in enumerate(escalation_predictions)], 
            key=lambda x: x[1].get('escalation_score', 0), 
            reverse=True
        )[:5]
        
        # Geographic distribution
        geographic_distribution = self._analyze_geographic_distribution(articles)
        
        # Temporal analysis
        temporal_analysis = self._analyze_temporal_patterns(articles)
        
        return {
            'key_metrics': {
                'total_articles': len(articles),
                'high_escalation_events': high_escalation_count,
                'medium_escalation_events': medium_escalation_count,
                'clusters_identified': clustering_results.get('n_clusters', 0),
                'primary_regions': list(geographic_distribution.keys())[:3],
                'analysis_timespan': temporal_analysis.get('timespan_days', 0)
            },
            'top_threats': [
                {
                    'article_index': idx,
                    'title': articles[idx].get('title', 'Unknown'),
                    'escalation_score': pred.get('escalation_score', 0),
                    'confidence': pred.get('confidence', 0),
                    'source': articles[idx].get('source', 'Unknown Source'),
                    'url': articles[idx].get('url', ''),
                    'summary': articles[idx].get('content', articles[idx].get('summary', ''))[:300] + '...' if articles[idx].get('content') or articles[idx].get('summary') else 'No summary available',
                    'primary_risk_factors': pred.get('explanation', {}).get('risk_indicators', [])
                }
                for idx, pred in top_threats
            ],
            'geographic_hotspots': geographic_distribution,
            'threat_level_distribution': {
                'high': high_escalation_count,
                'medium': medium_escalation_count,
                'low': len(articles) - high_escalation_count - medium_escalation_count
            },
            'key_findings': self._generate_key_findings(articles, clustering_results, escalation_predictions)
        }
    
    def _generate_cluster_analyses(self, 
                                 clustering_results: Dict[str, Any],
                                 escalation_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed analysis for each cluster."""
        cluster_analyses = []
        
        clusters = clustering_results.get('clusters', [])
        
        for cluster in clusters:
            cluster_id = cluster.get('cluster_id')
            metadata = cluster.get('metadata', {})
            articles = cluster.get('articles', [])
            
            # Calculate cluster escalation
            cluster_escalation_scores = []
            for article in articles:
                article_id = article.get('id')
                if article_id and article_id < len(escalation_predictions):
                    cluster_escalation_scores.append(escalation_predictions[article_id].get('escalation_score', 0))
            
            avg_escalation = np.mean(cluster_escalation_scores) if cluster_escalation_scores else 0.0
            
            # Generate cluster summary
            cluster_summary = self._generate_cluster_summary(cluster, avg_escalation)
            
            # Analyze cluster significance
            significance_analysis = self._analyze_cluster_significance(cluster, avg_escalation)
            
            cluster_analysis = {
                'cluster_id': cluster_id,
                'title': self._generate_cluster_title(metadata),
                'event_type': metadata.get('event_type', 'Unknown'),
                'escalation_level': round(avg_escalation * 10, 1),
                'significance_score': round(metadata.get('significance_score', 0) * 10, 1),
                'article_count': len(articles),
                'time_span': self._format_time_span(metadata.get('time_range', {})),
                'primary_actors': metadata.get('primary_actors', [])[:3],
                'key_locations': metadata.get('primary_countries', [])[:3],
                'summary': cluster_summary,
                'significance_analysis': significance_analysis,
                'articles': [
                    {
                        'title': article.get('title', 'Unknown'),
                        'source': article.get('source', 'Unknown'),
                        'url': article.get('url', ''),
                        'published_at': article.get('published_at'),
                        'content': article.get('content', article.get('summary', '')),
                        'escalation_score': escalation_predictions[article.get('id', 0)].get('escalation_score', 0) if article.get('id', 0) < len(escalation_predictions) else 0
                    }
                    for article in articles[:10]  # Limit to top 10 articles
                ],
                'top_articles': sorted([
                    {
                        'title': article.get('title', 'Unknown'),
                        'source': article.get('source', 'Unknown'),
                        'url': article.get('url', ''),
                        'published_at': article.get('published_at'),
                        'content': article.get('content', article.get('summary', '')),
                        'escalation_score': escalation_predictions[article.get('id', 0)].get('escalation_score', 0) if article.get('id', 0) < len(escalation_predictions) else 0
                    }
                    for article in articles
                ], key=lambda x: x['escalation_score'], reverse=True)[:5]
            }
            
            cluster_analyses.append(cluster_analysis)
        
        # Sort by significance
        cluster_analyses.sort(key=lambda x: x['significance_score'], reverse=True)
        
        return cluster_analyses
    
    def _generate_threat_assessment(self, 
                                  articles: List[Dict[str, Any]], 
                                  escalation_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive threat assessment."""
        
        # Categorize threats by level
        threat_levels = {
            'critical': [],  # > 0.8
            'high': [],      # 0.6-0.8
            'medium': [],    # 0.4-0.6
            'low': []        # < 0.4
        }
        
        for i, prediction in enumerate(escalation_predictions):
            score = prediction.get('escalation_score', 0)
            article_data = {
                'index': i,
                'title': articles[i].get('title', 'Unknown') if i < len(articles) else 'Unknown',
                'score': score,
                'confidence': prediction.get('confidence', 0),
                'risk_factors': prediction.get('explanation', {}).get('risk_indicators', [])
            }
            
            if score > 0.8:
                threat_levels['critical'].append(article_data)
            elif score > 0.6:
                threat_levels['high'].append(article_data)
            elif score > 0.4:
                threat_levels['medium'].append(article_data)
            else:
                threat_levels['low'].append(article_data)
        
        # Generate threat matrix
        threat_matrix = self._generate_threat_matrix(threat_levels, articles)
        
        # Identify emerging threats
        emerging_threats = self._identify_emerging_threats(articles, escalation_predictions)
        
        # Risk assessment by region
        regional_risk = self._assess_regional_risk(articles, escalation_predictions)
        
        return {
            'overall_threat_level': self._calculate_overall_threat_level(escalation_predictions),
            'threat_distribution': {
                'critical': len(threat_levels['critical']),
                'high': len(threat_levels['high']),
                'medium': len(threat_levels['medium']),
                'low': len(threat_levels['low'])
            },
            'threat_matrix': threat_matrix,
            'emerging_threats': emerging_threats,
            'regional_risk_assessment': regional_risk,
            'critical_alerts': threat_levels['critical'][:5],  # Top 5 critical threats
            'confidence_assessment': self._assess_threat_confidence(escalation_predictions)
        }
    
    def _generate_trend_analysis(self, 
                               articles: List[Dict[str, Any]], 
                               escalation_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate trend analysis section."""
        
        # Temporal trends
        temporal_trends = self._analyze_escalation_trends_over_time(articles, escalation_predictions)
        
        # Geographic trends
        geographic_trends = self._analyze_geographic_escalation_trends(articles, escalation_predictions)
        
        # Actor trends
        actor_trends = self._analyze_actor_involvement_trends(articles)
        
        # Event type trends
        event_type_trends = self._analyze_event_type_trends(articles, escalation_predictions)
        
        # Prediction trends
        prediction_trends = self._analyze_prediction_model_trends(escalation_predictions)
        
        return {
            'temporal_trends': temporal_trends,
            'geographic_trends': geographic_trends,
            'actor_trends': actor_trends,
            'event_type_trends': event_type_trends,
            'prediction_model_performance': prediction_trends,
            'trend_summary': self._generate_trend_summary(temporal_trends, geographic_trends, actor_trends)
        }
    
    def _generate_individual_article_analysis(self, 
                                            articles: List[Dict[str, Any]], 
                                            escalation_predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate detailed analysis for individual high-priority articles."""
        
        # Select top articles by escalation score
        top_articles = []
        for i, prediction in enumerate(escalation_predictions):
            if prediction.get('escalation_score', 0) > 0.5 and i < len(articles):
                article_analysis = {
                    'index': i,
                    'title': articles[i].get('title', 'Unknown'),
                    'source': articles[i].get('source', 'Unknown'),
                    'url': articles[i].get('url', ''),
                    'published_at': articles[i].get('published_at'),
                    'escalation_score': prediction.get('escalation_score', 0),
                    'confidence': prediction.get('confidence', 0),
                    'prediction_explanation': prediction.get('explanation', {}),
                    'key_entities': articles[i].get('entities', {}),
                    'geographic_relevance': articles[i].get('geopolitical_features', {}),
                    'significance_assessment': self._assess_article_significance(articles[i], prediction),
                    'related_articles': self._find_related_articles(i, articles, escalation_predictions)
                }
                top_articles.append(article_analysis)
        
        # Sort by escalation score
        top_articles.sort(key=lambda x: x['escalation_score'], reverse=True)
        
        return top_articles[:20]  # Limit to top 20 articles
    
    def _generate_recommendations(self, 
                                articles: List[Dict[str, Any]], 
                                clustering_results: Dict[str, Any],
                                escalation_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate actionable recommendations based on analysis."""
        
        recommendations = {
            'immediate_actions': [],
            'monitoring_priorities': [],
            'strategic_considerations': [],
            'intelligence_gaps': []
        }
        
        # Immediate actions for critical threats
        critical_threats = [p for p in escalation_predictions if p.get('escalation_score', 0) > 0.8]
        if critical_threats:
            recommendations['immediate_actions'].append(
                f"URGENT: {len(critical_threats)} critical escalation events require immediate attention"
            )
            recommendations['immediate_actions'].append(
                "Initiate enhanced monitoring protocols for high-risk regions"
            )
        
        # Monitoring priorities
        high_risk_regions = self._identify_high_risk_regions(articles, escalation_predictions)
        for region in high_risk_regions[:3]:
            recommendations['monitoring_priorities'].append(
                f"Increase surveillance and intelligence collection in {region}"
            )
        
        # Strategic considerations
        if clustering_results.get('n_clusters', 0) > 5:
            recommendations['strategic_considerations'].append(
                "Multiple simultaneous events detected - consider coordinated response planning"
            )
        
        # Intelligence gaps
        low_confidence_predictions = [p for p in escalation_predictions if p.get('confidence', 0) < 0.5]
        if len(low_confidence_predictions) > len(escalation_predictions) * 0.3:
            recommendations['intelligence_gaps'].append(
                "Significant uncertainty in threat assessment - recommend additional intelligence collection"
            )
        
        return recommendations
    
    def _format_text_report(self, report: Dict[str, Any]) -> str:
        """Format the report as a readable text document."""
        
        text_report = []
        
        # Header
        text_report.append("=" * 80)
        text_report.append("PAPERBOY INTELLIGENCE REPORT")
        text_report.append("=" * 80)
        text_report.append(f"Generated: {report['metadata']['generated_at']}")
        text_report.append(f"Analysis Period: {report['metadata']['analysis_period']}")
        text_report.append(f"Total Articles: {report['metadata']['total_articles']}")
        text_report.append(f"Confidence Level: {report['metadata']['confidence_level']:.1%}")
        text_report.append("")
        
        # Executive Summary
        text_report.append("EXECUTIVE SUMMARY")
        text_report.append("-" * 40)
        exec_summary = report['executive_summary']
        
        text_report.append(f"• Total Events Analyzed: {exec_summary['key_metrics']['total_articles']}")
        text_report.append(f"• High Escalation Events: {exec_summary['key_metrics']['high_escalation_events']}")
        text_report.append(f"• Medium Escalation Events: {exec_summary['key_metrics']['medium_escalation_events']}")
        text_report.append(f"• Event Clusters Identified: {exec_summary['key_metrics']['clusters_identified']}")
        text_report.append(f"• Primary Regions: {', '.join(exec_summary['key_metrics']['primary_regions'])}")
        text_report.append("")
        
        # Top Threats
        text_report.append("TOP THREATS")
        text_report.append("-" * 40)
        for i, threat in enumerate(exec_summary['top_threats'][:5], 1):
            text_report.append(f"{i}. {threat['title']}")
            text_report.append(f"   Escalation Score: {threat['escalation_score']:.3f}")
            text_report.append(f"   Confidence: {threat['confidence']:.3f}")
            
            # Add source and link if available
            if threat.get('source'):
                text_report.append(f"   Source: {threat['source']}")
            if threat.get('url'):
                text_report.append(f"   🔗 Link: {threat['url']}")
            
            text_report.append(f"   Risk Factors: {', '.join(threat['primary_risk_factors'])}")
            
            # Add summary if available
            if threat.get('summary'):
                summary = threat['summary']
                if len(summary) > 200:
                    summary = summary[:200] + "..."
                text_report.append(f"   📄 Summary: {summary}")
            
            text_report.append("")
        
        # Threat Assessment
        text_report.append("THREAT ASSESSMENT")
        text_report.append("-" * 40)
        threat_assessment = report['threat_assessment']
        text_report.append(f"Overall Threat Level: {threat_assessment['overall_threat_level']}")
        text_report.append("")
        text_report.append("Threat Distribution:")
        for level, count in threat_assessment['threat_distribution'].items():
            text_report.append(f"  {level.upper()}: {count} events")
        text_report.append("")
        
        # Critical Alerts
        if threat_assessment['critical_alerts']:
            text_report.append("CRITICAL ALERTS")
            text_report.append("-" * 40)
            for alert in threat_assessment['critical_alerts']:
                text_report.append(f"• {alert['title']}")
                text_report.append(f"  Score: {alert['score']:.3f} | Confidence: {alert['confidence']:.3f}")
                text_report.append("")
        
        # Cluster Analysis
        if report['cluster_analysis']:
            text_report.append("CLUSTER ANALYSIS")
            text_report.append("-" * 40)
            for cluster in report['cluster_analysis'][:5]:  # Top 5 clusters
                text_report.append(f"Cluster: {cluster['title']}")
                text_report.append(f"  Event Type: {cluster['event_type']}")
                text_report.append(f"  Escalation Level: {cluster['escalation_level']}/10")
                text_report.append(f"  Articles: {cluster['article_count']}")
                text_report.append(f"  Key Actors: {', '.join(cluster['primary_actors'])}")
                text_report.append(f"  Locations: {', '.join(cluster['key_locations'])}")
                text_report.append(f"  Summary: {cluster['summary']}")
                
                # Add top articles with links and summaries
                if 'top_articles' in cluster and cluster['top_articles']:
                    text_report.append("  Top Articles:")
                    for i, article in enumerate(cluster['top_articles'][:5], 1):
                        title = article.get('title', 'No title')[:80] + ('...' if len(article.get('title', '')) > 80 else '')
                        url = article.get('url', 'No URL')
                        source = article.get('source', 'Unknown source')
                        escalation = article.get('escalation_score', 0)
                        
                        # Get article summary/content
                        content = article.get('content', article.get('summary', ''))
                        if content:
                            # Clean and truncate content for summary
                            summary = content.replace('\n', ' ').replace('\r', ' ')
                            summary = ' '.join(summary.split())  # Remove extra whitespace
                            if len(summary) > 250:
                                summary = summary[:250] + "..."
                        else:
                            summary = "No summary available"
                        
                        text_report.append(f"    {i}. {title}")
                        text_report.append(f"       Source: {source} | Escalation: {escalation:.3f}")
                        text_report.append(f"       🔗 Link: {url}")
                        text_report.append(f"       📄 Summary: {summary}")
                        if i < len(cluster['top_articles'][:5]):
                            text_report.append("")
                text_report.append("")
        
        # Recommendations
        text_report.append("RECOMMENDATIONS")
        text_report.append("-" * 40)
        recommendations = report['recommendations']
        
        if recommendations['immediate_actions']:
            text_report.append("IMMEDIATE ACTIONS:")
            for action in recommendations['immediate_actions']:
                text_report.append(f"• {action}")
            text_report.append("")
        
        if recommendations['monitoring_priorities']:
            text_report.append("MONITORING PRIORITIES:")
            for priority in recommendations['monitoring_priorities']:
                text_report.append(f"• {priority}")
            text_report.append("")
        
        if recommendations['strategic_considerations']:
            text_report.append("STRATEGIC CONSIDERATIONS:")
            for consideration in recommendations['strategic_considerations']:
                text_report.append(f"• {consideration}")
            text_report.append("")
        
        # Footer
        text_report.append("=" * 80)
        text_report.append("END OF REPORT")
        text_report.append("=" * 80)
        
        return "\n".join(text_report)
    
    # Helper methods for analysis
    
    def _get_analysis_period(self, articles: List[Dict[str, Any]]) -> str:
        """Get the analysis time period from articles."""
        dates = []
        for article in articles:
            pub_date = article.get('published_at') or article.get('timestamps', {}).get('published_at')
            if pub_date:
                try:
                    dates.append(pd.to_datetime(pub_date))
                except:
                    continue
        
        if dates:
            start_date = min(dates).strftime('%Y-%m-%d')
            end_date = max(dates).strftime('%Y-%m-%d')
            return f"{start_date} to {end_date}"
        
        return "Unknown"
    
    def _calculate_overall_confidence(self, predictions: List[Dict[str, Any]]) -> float:
        """Calculate overall confidence across all predictions."""
        confidences = [p.get('confidence', 0) for p in predictions]
        return np.mean(confidences) if confidences else 0.0
    
    def _analyze_geographic_distribution(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze geographic distribution of articles."""
        regions = []
        for article in articles:
            # Check geopolitical features
            geo_features = article.get('geopolitical_features', {})
            if geo_features.get('primary_region'):
                regions.append(geo_features['primary_region'])
            
            # Check entities
            entities = article.get('entities', {})
            countries = entities.get('countries', [])
            regions.extend(countries)
        
        return dict(Counter(regions).most_common(10))
    
    def _analyze_temporal_patterns(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze temporal patterns in articles."""
        dates = []
        for article in articles:
            pub_date = article.get('published_at') or article.get('timestamps', {}).get('published_at')
            if pub_date:
                try:
                    dates.append(pd.to_datetime(pub_date))
                except:
                    continue
        
        if dates:
            timespan = (max(dates) - min(dates)).days
            return {
                'earliest_date': min(dates).isoformat(),
                'latest_date': max(dates).isoformat(),
                'timespan_days': timespan,
                'articles_per_day': len(articles) / max(1, timespan)
            }
        
        return {'timespan_days': 0}
    
    def _generate_key_findings(self, 
                             articles: List[Dict[str, Any]], 
                             clustering_results: Dict[str, Any],
                             escalation_predictions: List[Dict[str, Any]]) -> List[str]:
        """Generate key findings from the analysis."""
        findings = []
        
        # High escalation events
        high_escalation = len([p for p in escalation_predictions if p.get('escalation_score', 0) > 0.7])
        if high_escalation > 0:
            findings.append(f"{high_escalation} high-escalation events detected requiring immediate attention")
        
        # Clustering insights
        n_clusters = clustering_results.get('n_clusters', 0)
        if n_clusters > 3:
            findings.append(f"{n_clusters} distinct event clusters identified, suggesting multiple simultaneous developments")
        
        # Geographic concentration
        geo_dist = self._analyze_geographic_distribution(articles)
        if geo_dist:
            top_region = list(geo_dist.keys())[0]
            findings.append(f"Primary activity concentrated in {top_region} region")
        
        # Temporal concentration
        temporal = self._analyze_temporal_patterns(articles)
        if temporal.get('timespan_days', 0) < 3:
            findings.append("Events concentrated within 72-hour window, indicating rapid development")
        
        return findings
    
    def _generate_cluster_title(self, metadata: Dict[str, Any]) -> str:
        """Generate a descriptive title for a cluster."""
        event_type = metadata.get('event_type', 'Event')
        primary_countries = metadata.get('primary_countries', [])
        primary_actors = metadata.get('primary_actors', [])
        
        if primary_countries:
            location = primary_countries[0]
        elif primary_actors:
            location = primary_actors[0]
        else:
            location = "Unknown Region"
        
        return f"{event_type} - {location}"
    
    def _format_time_span(self, time_range: Dict[str, Any]) -> str:
        """Format time span for display."""
        if not time_range:
            return "Unknown"
        
        span_days = time_range.get('span_days', 0)
        if span_days == 0:
            return "Single day"
        elif span_days == 1:
            return "2 days"
        else:
            return f"{span_days + 1} days"
    
    def _generate_cluster_summary(self, cluster: Dict[str, Any], avg_escalation: float) -> str:
        """Generate a summary for a cluster."""
        metadata = cluster.get('metadata', {})
        article_count = cluster.get('article_count', 0)
        event_type = metadata.get('event_type', 'events')
        
        escalation_desc = "high" if avg_escalation > 0.7 else "moderate" if avg_escalation > 0.4 else "low"
        
        primary_actors = metadata.get('primary_actors', [])
        primary_countries = metadata.get('primary_countries', [])
        
        actors_str = ", ".join(primary_actors[:2]) if primary_actors else "various actors"
        countries_str = ", ".join(primary_countries[:2]) if primary_countries else "multiple regions"
        
        return f"Cluster of {article_count} articles covering {event_type.lower()} involving {actors_str} in {countries_str}. Escalation level assessed as {escalation_desc}."
    
    def _analyze_cluster_significance(self, cluster: Dict[str, Any], avg_escalation: float) -> str:
        """Analyze the significance of a cluster."""
        metadata = cluster.get('metadata', {})
        significance_score = metadata.get('significance_score', 0)
        
        if significance_score > 0.8:
            return "CRITICAL - Requires immediate attention and monitoring"
        elif significance_score > 0.6:
            return "HIGH - Significant development requiring close monitoring"
        elif significance_score > 0.4:
            return "MODERATE - Notable event worth continued observation"
        else:
            return "LOW - Routine monitoring sufficient"
    
    def _calculate_overall_threat_level(self, predictions: List[Dict[str, Any]]) -> str:
        """Calculate overall threat level."""
        scores = [p.get('escalation_score', 0) for p in predictions]
        avg_score = np.mean(scores) if scores else 0
        
        if avg_score > 0.7:
            return "CRITICAL"
        elif avg_score > 0.5:
            return "HIGH"
        elif avg_score > 0.3:
            return "MODERATE"
        else:
            return "LOW"
    
    def _generate_threat_matrix(self, threat_levels: Dict[str, List], articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate threat assessment matrix."""
        matrix = []
        
        for level, threats in threat_levels.items():
            if threats:
                # Analyze regions for this threat level
                regions = []
                indicators = []
                
                for threat in threats:
                    if threat['index'] < len(articles):
                        article = articles[threat['index']]
                        geo_features = article.get('geopolitical_features', {})
                        if geo_features.get('primary_region'):
                            regions.append(geo_features['primary_region'])
                        indicators.extend(threat['risk_factors'])
                
                matrix.append({
                    'threat_level': level.upper(),
                    'event_count': len(threats),
                    'primary_regions': list(Counter(regions).most_common(3)),
                    'key_indicators': list(Counter(indicators).most_common(5))
                })
        
        return matrix
    
    def _identify_emerging_threats(self, articles: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify emerging threats from the analysis."""
        emerging = []
        
        # Look for recent articles with high escalation
        recent_threshold = datetime.now() - timedelta(days=2)
        
        for i, prediction in enumerate(predictions):
            if i < len(articles) and prediction.get('escalation_score', 0) > 0.6:
                article = articles[i]
                pub_date = article.get('published_at')
                
                if pub_date:
                    try:
                        if pd.to_datetime(pub_date) > recent_threshold:
                            emerging.append({
                                'title': article.get('title', 'Unknown'),
                                'escalation_score': prediction.get('escalation_score', 0),
                                'published_at': pub_date,
                                'risk_factors': prediction.get('explanation', {}).get('risk_indicators', [])
                            })
                    except:
                        continue
        
        return sorted(emerging, key=lambda x: x['escalation_score'], reverse=True)[:10]
    
    def _assess_regional_risk(self, articles: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Assess risk by geographic region."""
        regional_data = defaultdict(list)
        
        for i, prediction in enumerate(predictions):
            if i < len(articles):
                article = articles[i]
                geo_features = article.get('geopolitical_features', {})
                region = geo_features.get('primary_region', 'Unknown')
                
                regional_data[region].append(prediction.get('escalation_score', 0))
        
        regional_risk = {}
        for region, scores in regional_data.items():
            if scores:
                regional_risk[region] = {
                    'average_escalation': np.mean(scores),
                    'max_escalation': max(scores),
                    'event_count': len(scores),
                    'risk_level': 'HIGH' if np.mean(scores) > 0.6 else 'MODERATE' if np.mean(scores) > 0.3 else 'LOW'
                }
        
        return regional_risk
    
    def _assess_threat_confidence(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Assess confidence in threat predictions."""
        confidences = [p.get('confidence', 0) for p in predictions]
        
        return {
            'average_confidence': np.mean(confidences) if confidences else 0,
            'high_confidence_predictions': len([c for c in confidences if c > 0.7]),
            'low_confidence_predictions': len([c for c in confidences if c < 0.3]),
            'confidence_distribution': {
                'high': len([c for c in confidences if c > 0.7]),
                'medium': len([c for c in confidences if 0.3 <= c <= 0.7]),
                'low': len([c for c in confidences if c < 0.3])
            }
        }
    
    def _analyze_escalation_trends_over_time(self, articles: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how escalation scores trend over time."""
        time_series = []
        
        for i, prediction in enumerate(predictions):
            if i < len(articles):
                article = articles[i]
                pub_date = article.get('published_at')
                if pub_date:
                    try:
                        time_series.append({
                            'date': pd.to_datetime(pub_date),
                            'escalation_score': prediction.get('escalation_score', 0)
                        })
                    except:
                        continue
        
        if not time_series:
            return {}
        
        # Sort by date
        time_series.sort(key=lambda x: x['date'])
        
        # Calculate trends
        recent_scores = [item['escalation_score'] for item in time_series[-10:]]  # Last 10 articles
        early_scores = [item['escalation_score'] for item in time_series[:10]]   # First 10 articles
        
        trend_direction = "INCREASING" if np.mean(recent_scores) > np.mean(early_scores) else "DECREASING"
        
        return {
            'trend_direction': trend_direction,
            'recent_average': np.mean(recent_scores),
            'early_average': np.mean(early_scores),
            'overall_average': np.mean([item['escalation_score'] for item in time_series]),
            'volatility': np.std([item['escalation_score'] for item in time_series])
        }
    
    def _analyze_geographic_escalation_trends(self, articles: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze escalation trends by geographic region."""
        regional_scores = defaultdict(list)
        
        for i, prediction in enumerate(predictions):
            if i < len(articles):
                article = articles[i]
                geo_features = article.get('geopolitical_features', {})
                region = geo_features.get('primary_region', 'Unknown')
                
                regional_scores[region].append(prediction.get('escalation_score', 0))
        
        regional_trends = {}
        for region, scores in regional_scores.items():
            if scores and len(scores) >= 2:
                regional_trends[region] = {
                    'average_escalation': np.mean(scores),
                    'trend': 'INCREASING' if scores[-1] > scores[0] else 'DECREASING',
                    'volatility': np.std(scores),
                    'event_count': len(scores)
                }
        
        return regional_trends
    
    def _analyze_actor_involvement_trends(self, articles: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze trends in actor involvement."""
        actors = []
        
        for article in articles:
            entities = article.get('entities', {})
            organizations = entities.get('organizations', [])
            persons = entities.get('persons', [])
            
            actors.extend(organizations)
            actors.extend(persons)
        
        return dict(Counter(actors).most_common(10))
    
    def _analyze_event_type_trends(self, articles: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze escalation trends by event type."""
        event_type_scores = defaultdict(list)
        
        for i, prediction in enumerate(predictions):
            if i < len(articles):
                article = articles[i]
                geo_features = article.get('geopolitical_features', {})
                
                # Infer event type from conflict indicators
                conflict_indicators = geo_features.get('conflict_indicators', [])
                if 'military' in conflict_indicators:
                    event_type = 'Military'
                elif 'diplomatic' in conflict_indicators:
                    event_type = 'Diplomatic'
                elif 'nuclear' in conflict_indicators:
                    event_type = 'Nuclear'
                else:
                    event_type = 'General'
                
                event_type_scores[event_type].append(prediction.get('escalation_score', 0))
        
        event_trends = {}
        for event_type, scores in event_type_scores.items():
            if scores:
                event_trends[event_type] = {
                    'average_escalation': np.mean(scores),
                    'max_escalation': max(scores),
                    'event_count': len(scores)
                }
        
        return event_trends
    
    def _analyze_prediction_model_trends(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction model performance trends."""
        model_usage = defaultdict(int)
        model_scores = defaultdict(list)
        
        for prediction in predictions:
            models_used = prediction.get('models_used', [])
            escalation_score = prediction.get('escalation_score', 0)
            
            for model in models_used:
                model_usage[model] += 1
                model_scores[model].append(escalation_score)
        
        model_performance = {}
        for model, scores in model_scores.items():
            if scores:
                model_performance[model] = {
                    'usage_count': model_usage[model],
                    'average_score': np.mean(scores),
                    'score_range': max(scores) - min(scores)
                }
        
        return {
            'model_usage_distribution': dict(model_usage),
            'model_performance': model_performance,
            'most_used_model': max(model_usage.items(), key=lambda x: x[1])[0] if model_usage else 'None'
        }
    
    def _generate_trend_summary(self, temporal_trends: Dict[str, Any], geographic_trends: Dict[str, Any], actor_trends: Dict[str, int]) -> List[str]:
        """Generate summary of key trends."""
        summary = []
        
        # Temporal trends
        if temporal_trends.get('trend_direction') == 'INCREASING':
            summary.append("Escalation levels are trending upward over the analysis period")
        elif temporal_trends.get('trend_direction') == 'DECREASING':
            summary.append("Escalation levels are trending downward over the analysis period")
        
        # Geographic trends
        if geographic_trends:
            high_risk_regions = [region for region, data in geographic_trends.items() 
                               if data.get('average_escalation', 0) > 0.6]
            if high_risk_regions:
                summary.append(f"High-risk regions identified: {', '.join(high_risk_regions)}")
        
        # Actor trends
        if actor_trends:
            top_actor = list(actor_trends.keys())[0]
            summary.append(f"Most frequently mentioned actor: {top_actor}")
        
        return summary
    
    def _assess_article_significance(self, article: Dict[str, Any], prediction: Dict[str, Any]) -> str:
        """Assess the significance of an individual article."""
        escalation_score = prediction.get('escalation_score', 0)
        confidence = prediction.get('confidence', 0)
        
        if escalation_score > 0.8 and confidence > 0.7:
            return "CRITICAL - High confidence critical threat"
        elif escalation_score > 0.6:
            return "HIGH - Significant escalation detected"
        elif escalation_score > 0.4:
            return "MODERATE - Notable development"
        else:
            return "LOW - Routine monitoring"
    
    def _find_related_articles(self, article_index: int, articles: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> List[int]:
        """Find articles related to the given article."""
        if article_index >= len(articles):
            return []
        
        target_article = articles[article_index]
        target_entities = target_article.get('entities', {})
        target_countries = set(target_entities.get('countries', []))
        target_orgs = set(target_entities.get('organizations', []))
        
        related = []
        for i, article in enumerate(articles):
            if i == article_index:
                continue
            
            entities = article.get('entities', {})
            countries = set(entities.get('countries', []))
            orgs = set(entities.get('organizations', []))
            
            # Check for entity overlap
            country_overlap = len(target_countries & countries)
            org_overlap = len(target_orgs & orgs)
            
            if country_overlap > 0 or org_overlap > 0:
                related.append(i)
        
        return related[:5]  # Limit to 5 related articles
    
    def _identify_high_risk_regions(self, articles: List[Dict[str, Any]], predictions: List[Dict[str, Any]]) -> List[str]:
        """Identify high-risk geographic regions."""
        regional_risk = self._assess_regional_risk(articles, predictions)
        
        high_risk_regions = [
            region for region, data in regional_risk.items()
            if data.get('average_escalation', 0) > 0.5
        ]
        
        return sorted(high_risk_regions, key=lambda r: regional_risk[r]['average_escalation'], reverse=True)
    
    def _generate_methodology_appendix(self) -> Dict[str, Any]:
        """Generate methodology appendix."""
        return {
            'clustering_method': 'Multi-layer clustering using SBERT embeddings, NER, and temporal analysis',
            'escalation_prediction': 'Ensemble model combining OSINT-trained models, XGBoost, and text analysis',
            'confidence_calculation': 'Based on model agreement and prediction consistency',
            'threat_assessment': 'Multi-factor analysis including escalation scores, geographic risk, and temporal patterns'
        }
    
    def _generate_data_sources_appendix(self, articles: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate data sources appendix."""
        sources = Counter(article.get('source', 'Unknown') for article in articles)
        
        return {
            'total_sources': len(sources),
            'primary_sources': dict(sources.most_common(10)),
            'source_diversity': len(sources) / len(articles) if articles else 0
        }
    
    def _generate_confidence_metrics(self, predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate confidence metrics appendix."""
        confidences = [p.get('confidence', 0) for p in predictions]
        
        return {
            'average_confidence': np.mean(confidences) if confidences else 0,
            'confidence_std': np.std(confidences) if confidences else 0,
            'high_confidence_percentage': len([c for c in confidences if c > 0.7]) / len(confidences) if confidences else 0,
            'confidence_range': {
                'min': min(confidences) if confidences else 0,
                'max': max(confidences) if confidences else 0
            }
        }


# Main function for external use
def generate_intelligence_report(articles: List[Dict[str, Any]], 
                               clustering_results: Dict[str, Any],
                               escalation_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to generate comprehensive intelligence report.
    
    Args:
        articles: List of analyzed articles
        clustering_results: Results from clustering analysis
        escalation_predictions: Results from escalation prediction
        
    Returns:
        Comprehensive intelligence report
    """
    reporter = UnifiedIntelligenceReporter()
    return reporter.generate_comprehensive_report(articles, clustering_results, escalation_predictions)