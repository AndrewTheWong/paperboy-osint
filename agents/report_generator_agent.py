"""
Report Generator Agent for StraitWatch
Generates professional intelligence analyst reports for Taiwan Strait monitoring
"""

import asyncio
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .base_agent import BaseAgent

class ReportGeneratorAgent(BaseAgent):
    """Agent responsible for generating professional intelligence analyst reports"""
    
    def __init__(self):
        super().__init__("report_generator_agent")
        
        # Ensure reports directory exists
        self.reports_dir = Path("reports")
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Intelligence report template configuration
        self.classification_level = "UNCLASSIFIED//FOR OFFICIAL USE ONLY"
        self.report_type = "DAILY INTELLIGENCE ASSESSMENT"
        
    async def run(self) -> Dict[str, Any]:
        """Main intelligence report generation workflow"""
        self.logger.info("Generating StraitWatch Intelligence Assessment")
        
        try:
            # Gather structured intelligence data
            self.logger.debug("Gathering intelligence data...")
            intelligence_data = await self.gather_intelligence_data()
            
            # Generate professional intelligence report
            self.logger.debug("Generating intelligence report...")
            report_content = await self.generate_intelligence_report(intelligence_data)
            
            # Save report in multiple formats
            self.logger.debug("Saving intelligence report...")
            report_paths = await self.save_intelligence_report(report_content)
            
            # Store in database for tracking
            self.logger.debug("Storing intelligence report in database...")
            await self.store_intelligence_report(report_content, report_paths)
            
            result = {
                "success": True,
                "report_paths": report_paths,
                "report_date": date.today().isoformat(),
                "classification": self.classification_level,
                "total_events": len(intelligence_data.get("recent_events", [])),
                "threat_assessment": report_content.get("threat_assessment", "UNKNOWN"),
                "escalation_forecast": report_content.get("escalation_forecast", "No forecast available")
            }
            
            self.logger.info(f"Intelligence assessment completed: {report_paths.get('text')}")
            return result
            
        except Exception as e:
            self.logger.error(f"Intelligence report generation failed: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return {"success": False, "error": str(e)}
    
    async def gather_intelligence_data(self) -> Dict[str, Any]:
        """Gather structured intelligence data from database"""
        now = datetime.now()
        
        # Intelligence collection timeframes
        last_24h = now - timedelta(hours=24)
        last_48h = now - timedelta(hours=48)
        last_72h = now - timedelta(hours=72)
        last_7d = now - timedelta(days=7)
        
        # Gather all intelligence inputs
        data = {
            "collection_timestamp": now,
            "recent_events": await self.get_structured_events(last_72h),
            "recent_articles": await self.get_tagged_articles(last_48h),
            "escalation_forecast": await self.get_escalation_forecast(),
            "sentiment_analysis": await self.get_sentiment_trends(last_48h),
            "entity_patterns": await self.get_entity_clustering(last_72h),
            "geographic_analysis": await self.get_geographic_patterns(last_72h),
            "confidence_metrics": await self.get_model_confidence(),
            "historical_baseline": await self.get_historical_baseline(last_7d)
        }
        
        return data
    
    async def generate_intelligence_report(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate professional intelligence analyst report"""
        
        # Assess overall situation
        threat_assessment = self.assess_threat_level(data)
        escalation_risk = self.assess_escalation_risk(data)
        
        # Generate report sections
        report = {
            "header": self.generate_report_header(data),
            "executive_summary": self.generate_executive_summary(data, threat_assessment, escalation_risk),
            "key_developments": self.generate_key_developments(data),
            "pattern_analysis": self.generate_pattern_analysis(data),
            "escalation_forecast": self.generate_escalation_forecast(data),
            "strategic_warning_indicators": self.generate_strategic_warning_indicators(data),
            "source_reference_table": self.generate_source_reference_table(data),
            "threat_assessment": threat_assessment,
            "escalation_risk": escalation_risk,
            "confidence_assessment": self.generate_confidence_assessment(data)
        }
        
        return report
    
    def generate_report_header(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Generate intelligence report header"""
        return {
            "classification": self.classification_level,
            "title": "STRAITWATCH DAILY INTELLIGENCE ASSESSMENT",
            "subtitle": "Taiwan Strait Political-Military Developments",
            "date": date.today().strftime("%d %B %Y"),
            "time": data["collection_timestamp"].strftime("%H%M UTC"),
            "originator": "StraitWatch OSINT Analysis Platform",
            "distribution": "Authorized Recipients Only"
        }
    
    def generate_executive_summary(self, data: Dict[str, Any], threat_level: str, escalation_risk: str) -> str:
        """Generate executive summary (150 words max)"""
        events = data.get("recent_events", [])
        forecast = data.get("escalation_forecast", {})
        
        # Analyze trend direction
        recent_severity = sum(e.get("confidence", 0) for e in events[-5:]) / max(len(events[-5:]), 1)
        trend_direction = "rising" if recent_severity > 0.6 else "stable" if recent_severity > 0.3 else "declining"
        
        # Key developments
        high_impact_events = [e for e in events if e.get("confidence", 0) > 0.7]
        if high_impact_events:
            key_development = high_impact_events[0].get("event_description", high_impact_events[0].get("description", "No major developments"))
        else:
            key_development = "Routine activities observed"
        
        # Forecast assessment
        forecast_trend = forecast.get("trend", "stable")
        forecast_confidence = forecast.get("confidence", "medium")
        
        summary = f"""
ASSESSMENT: Taiwan Strait tensions are currently {trend_direction.upper()} with threat level assessed as {threat_level}.

SITUATION: {key_development}. Analysis of {len(events)} events over past 72 hours indicates {trend_direction} military signaling and diplomatic posturing.

FORECAST: Models predict {forecast_trend} escalation trajectory over next 7 days ({forecast_confidence} confidence). Elevated military activity expected through mid-week based on historical patterns.

INDICATORS: Monitoring {len([e for e in events if 'military' in e.get('event_type', '')])} military-related developments and {len([e for e in events if 'diplomatic' in e.get('event_type', '')])} diplomatic indicators for early warning signals.
""".strip()
        
        return summary
    
    def generate_key_developments(self, data: Dict[str, Any]) -> str:
        """Generate key developments section with enhanced event descriptions"""
        events = data.get("recent_events", [])
        articles = data.get("recent_articles", [])
        
        if not events:
            return "• No significant developments detected in the assessment period"
        
        # Sort by confidence and escalation score
        sorted_events = sorted(events, key=lambda x: (
            x.get("escalation_score", 0) * 0.6 + x.get("confidence", 0) * 0.4
        ), reverse=True)
        
        developments = []
        for i, event in enumerate(sorted_events[:10]):
            # Parse event data - FIXED: Use 'description' field with fallback
            event_type = event.get("event_type", "Unknown").replace("_", " ").title()
            description = event.get("description", event.get("event_description", "No description"))
            location = event.get("location", "Unspecified location")
            confidence_score = event.get("confidence", 0)
            escalation_score = event.get("escalation_score", 0)
            timestamp = event.get("created_at", "")
            
            # Find related article for full context
            article_id = event.get("article_id")
            related_article = None
            if article_id:
                related_article = next((a for a in articles if str(a.get("id", -1)) == str(article_id)), None)
            
            # Determine confidence level
            conf_level = "High" if confidence_score > 0.8 else "Medium" if confidence_score > 0.5 else "Low"
            
            # Determine risk level using escalation score if available
            if escalation_score > 0:
                risk_level = "Critical" if escalation_score > 0.8 else "High" if escalation_score > 0.6 else "Medium" if escalation_score > 0.4 else "Low"
            else:
                risk_level = "High" if confidence_score > 0.6 else "Medium" if confidence_score > 0.4 else "Low"
            
            # Enhanced source information
            if related_article:
                source = related_article.get("source", "Unknown Source")
                article_title = related_article.get("title", "")
                article_url = related_article.get("url", "")
                
                # Generate article summary (first 200 chars of content)
                content = related_article.get("content", "")
                summary = content[:200] + "..." if len(content) > 200 else content
                
                # Create source link if URL available
                source_link = f"[{source}]({article_url})" if article_url else source
                
                # FIXED: Safe access to escalation_analysis with proper None handling
                escalation_analysis = event.get('escalation_analysis') or 'Not available'
                analysis_preview = escalation_analysis[:100] + "..." if len(escalation_analysis) > 100 else escalation_analysis
                
                development = f"""
• **{description}**
  ├─ Event Type: {event_type} | Risk Level: {risk_level} | Confidence: {conf_level}
  ├─ Location: {location} | Timestamp: {timestamp[:16] if timestamp else 'Recent'}Z
  ├─ Escalation Score: {escalation_score:.2f}/1.0 | Analysis: {analysis_preview}
  ├─ Source Article: "{article_title[:80]}..." 
  ├─ Source: {source_link}
  └─ Context: {summary}
"""
            else:
                # Fallback for events without linked articles
                source = "OSINT Collection"
                
                # FIXED: Safe access to escalation_analysis with proper None handling
                escalation_analysis = event.get('escalation_analysis') or 'Not available'
                analysis_preview = escalation_analysis[:100] + "..." if len(escalation_analysis) > 100 else escalation_analysis
                
                development = f"""
• **{description}**
  ├─ Event Type: {event_type} | Risk Level: {risk_level} | Confidence: {conf_level}
  ├─ Location: {location} | Timestamp: {timestamp[:16] if timestamp else 'Recent'}Z
  ├─ Escalation Score: {escalation_score:.2f}/1.0 | Analysis: {analysis_preview}
  └─ Source: {source} (Multi-source intelligence fusion)
"""
            
            developments.append(development)
        
        if not developments:
            return "• No significant developments detected in the assessment period"
        
        return "\n".join(developments)
    
    def generate_pattern_analysis(self, data: Dict[str, Any]) -> str:
        """Generate pattern analysis section"""
        events = data.get("recent_events", [])
        entity_patterns = data.get("entity_patterns", {})
        geographic_patterns = data.get("geographic_analysis", {})
        
        if not events:
            return "Insufficient data for pattern analysis."
        
        # Event type clustering
        event_types = {}
        for event in events:
            event_type = event.get("event_type", "unknown")
            event_types[event_type] = event_types.get(event_type, 0) + 1
        
        # Geographic concentration
        locations = {}
        for event in events:
            location = event.get("location", "unknown")
            locations[location] = locations.get(location, 0) + 1
        
        # Temporal patterns
        time_distribution = self.analyze_temporal_patterns(events)
        
        analysis = f"""
EVENT TYPE CLUSTERING:
{self.format_frequency_analysis(event_types, "event type")}

GEOGRAPHIC CONCENTRATION:
{self.format_frequency_analysis(locations, "location")}

TEMPORAL ANALYSIS:
{time_distribution}

ENTITY PATTERNS:
{self.format_entity_patterns(entity_patterns)}

ASSESSMENT: {'Significant clustering detected in military activities' if any('military' in k for k in event_types.keys()) else 'Dispersed activity pattern with no clear concentration'}
"""
        
        return analysis.strip()
    
    def generate_escalation_forecast(self, data: Dict[str, Any]) -> str:
        """Generate escalation forecast section"""
        forecast = data.get("escalation_forecast", {})
        confidence = data.get("confidence_metrics", {})
        
        if not forecast:
            return "Escalation forecasting models unavailable."
        
        # Extract forecast data
        trend = forecast.get("trend", "stable")
        score = forecast.get("current_score", 0.5)
        projected_scores = forecast.get("7_day_projection", [])
        
        # Model confidence
        model_confidence = confidence.get("forecast_confidence", "medium")
        uncertainty = confidence.get("uncertainty_level", "moderate")
        
        # Risk assessment
        risk_level = "Elevated" if score > 0.6 else "Moderate" if score > 0.4 else "Low"
        
        # Generate projection lines
        projection_lines = []
        for i, proj in enumerate(projected_scores[:7]):
            direction = "↗" if i > 0 and proj > projected_scores[i-1] else "↘" if i > 0 and proj < projected_scores[i-1] else "→"
            projection_lines.append(f"Day +{i+1}: {proj:.2f} {direction}")
        
        projection_text = "\n".join(projection_lines) if projection_lines else "No projection data available"
        
        forecast_text = f"""
7-DAY ESCALATION OUTLOOK:

Current Escalation Score: {score:.2f} ({risk_level} Risk)
Trend Direction: {trend.upper()}
Model Confidence: {model_confidence.upper()}

PROJECTION:
{projection_text}

ANALYST INTERPRETATION:
{self.generate_forecast_interpretation(forecast, trend, score)}

UNCERTAINTY ASSESSMENT:
{uncertainty.title()} confidence beyond 3 days due to limited input data and dynamic geopolitical environment.

WARNING INDICATORS:
Monitor for deployment of amphibious assets, unusual diplomatic rhetoric, or cross-domain operations suggesting escalatory intent.
"""
        
        return forecast_text.strip()
    
    def generate_strategic_warning_indicators(self, data: Dict[str, Any]) -> str:
        """Generate strategic warning indicators section"""
        events = data.get("recent_events", [])
        
        # Define strategic indicators
        military_indicators = []
        political_indicators = []
        economic_indicators = []
        cyber_indicators = []
        
        for event in events:
            description = event.get("event_description", event.get("description", "")).lower()
            event_type = event.get("event_type", "").lower()
            severity = event.get("confidence", 0)
            
            # Categorize indicators
            if any(term in description for term in ["deploy", "exercise", "naval", "air force", "military"]):
                if severity > 0.6:
                    military_indicators.append(f"• {event.get('event_description', event.get('description', ''))} - Indicates possible pre-positioning")
                    
            elif any(term in description for term in ["diplomatic", "summit", "statement", "rhetoric"]):
                if severity > 0.5:
                    political_indicators.append(f"• {event.get('event_description', event.get('description', ''))} - May reflect policy shift preparation")
                    
            elif any(term in description for term in ["trade", "economic", "sanctions", "embargo"]):
                if severity > 0.5:
                    economic_indicators.append(f"• {event.get('event_description', event.get('description', ''))} - Suggests economic leverage preparation")
                    
            elif any(term in description for term in ["cyber", "information", "propaganda"]):
                cyber_indicators.append(f"• {event.get('event_description', event.get('description', ''))} - Cross-domain operations detected")
        
        # Format indicator lists
        military_text = "\n".join(military_indicators) if military_indicators else "• No significant military posture changes detected"
        political_text = "\n".join(political_indicators) if political_indicators else "• Routine diplomatic activity observed"
        economic_text = "\n".join(economic_indicators) if economic_indicators else "• No economic coercion indicators detected"
        cyber_text = "\n".join(cyber_indicators) if cyber_indicators else "• Standard information environment activity"
        
        indicators_text = f"""
MILITARY POSTURE INDICATORS:
{military_text}

POLITICAL/DIPLOMATIC INDICATORS:
{political_text}

ECONOMIC WARFARE INDICATORS:
{economic_text}

INFORMATION/CYBER INDICATORS:
{cyber_text}

RISK ASSESSMENT: {self.assess_strategic_risk(military_indicators, political_indicators, economic_indicators, cyber_indicators)}
"""
        
        return indicators_text.strip()
    
    def generate_source_reference_table(self, data: Dict[str, Any]) -> str:
        """Generate source reference table"""
        articles = data.get("recent_articles", [])
        
        if not articles:
            return "No source articles available for reference."
        
        # Sort by relevance and recency
        sorted_articles = sorted(articles, key=lambda x: (bool(x.get("relevant", False)), x.get("created_at", "")), reverse=True)
        
        table_header = f"{'Article Title':<50} | {'Source':<20} | {'Timestamp':<16} | {'Confidence':<10}"
        table_separator = "-" * len(table_header)
        
        rows = [table_header, table_separator]
        
        for article in sorted_articles[:15]:  # Top 15 sources
            title = article.get("title", "No title")[:47] + "..." if len(article.get("title", "")) > 50 else article.get("title", "No title")
            source = article.get("source", "Unknown")[:17] + "..." if len(article.get("source", "")) > 20 else article.get("source", "Unknown")
            timestamp = article.get("created_at", "")[:16] if article.get("created_at") else "Unknown"
            confidence = "High" if article.get("relevant", False) else "Medium"
            
            row = f"{title:<50} | {source:<20} | {timestamp:<16} | {confidence:<10}"
            rows.append(row)
        
        return "\n".join(rows)
    
    def assess_threat_level(self, data: Dict[str, Any]) -> str:
        """Assess overall threat level"""
        events = data.get("recent_events", [])
        
        if not events:
            return "LOW"
        
        avg_severity = sum(e.get("confidence", 0) for e in events) / len(events)
        max_severity = max(e.get("confidence", 0) for e in events)
        
        # Count high-severity events
        critical_events = len([e for e in events if e.get("confidence", 0) > 0.8])
        high_events = len([e for e in events if e.get("confidence", 0) > 0.6])
        
        if critical_events > 2 or max_severity > 0.9:
            return "CRITICAL"
        elif critical_events > 0 or high_events > 3 or avg_severity > 0.7:
            return "HIGH"
        elif high_events > 1 or avg_severity > 0.5:
            return "MEDIUM"
        else:
            return "LOW"
    
    def assess_escalation_risk(self, data: Dict[str, Any]) -> str:
        """Assess escalation risk"""
        forecast = data.get("escalation_forecast", {})
        events = data.get("recent_events", [])
        
        forecast_score = forecast.get("current_score", 0.5)
        trend = forecast.get("trend", "stable")
        
        military_events = len([e for e in events if "military" in e.get("event_type", "").lower()])
        
        if forecast_score > 0.8 or (trend == "rising" and military_events > 5):
            return "HIGH"
        elif forecast_score > 0.6 or (trend == "rising" and military_events > 2):
            return "MEDIUM"
        else:
            return "LOW"
    
    # Helper methods for data gathering
    async def get_structured_events(self, since_date: datetime) -> List[Dict[str, Any]]:
        """Get structured events from database"""
        try:
            result = self.supabase.table("events")\
                .select("*")\
                .gte("created_at", since_date.isoformat())\
                .order("confidence", desc=True)\
                .limit(50)\
                .execute()
            return result.data or []
        except Exception as e:
            self.logger.error(f"Error getting structured events: {e}")
            return []
    
    async def get_tagged_articles(self, since_date: datetime) -> List[Dict[str, Any]]:
        """Get tagged articles from database"""
        try:
            result = self.supabase.table("articles")\
                .select("*")\
                .gte("created_at", since_date.isoformat())\
                .order("created_at", desc=True)\
                .limit(30)\
                .execute()
            return result.data or []
        except Exception as e:
            self.logger.error(f"Error getting tagged articles: {e}")
            return []
    
    async def get_escalation_forecast(self) -> Dict[str, Any]:
        """Get latest escalation forecast"""
        try:
            result = self.supabase.table("forecasts")\
                .select("*")\
                .order("created_at", desc=True)\
                .limit(1)\
                .execute()
            
            if result.data and len(result.data) > 0:
                forecast_data = result.data[0].get("forecast_data", {})
                if forecast_data:
                    return {
                        "current_score": forecast_data.get("escalation_score", 0.5),
                        "trend": forecast_data.get("trend", "stable"),
                        "7_day_projection": forecast_data.get("projection", [0.5] * 7),
                        "confidence": forecast_data.get("confidence", 0.7)
                    }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting escalation forecast: {e}")
            return {}
    
    async def get_sentiment_trends(self, since_date: datetime) -> Dict[str, Any]:
        """Get sentiment analysis trends"""
        try:
            result = self.supabase.table("article_sentiment")\
                .select("*")\
                .gte("created_at", since_date.isoformat())\
                .execute()
            
            sentiments = result.data or []
            if not sentiments:
                return {}
            
            sentiment_scores = [s.get("sentiment_score", 0) for s in sentiments if s.get("sentiment_score") is not None]
            if not sentiment_scores:
                return {}
            
            avg_sentiment = sum(sentiment_scores) / len(sentiments)
            return {
                "average_sentiment": avg_sentiment,
                "total_analyzed": len(sentiments),
                "trend": "negative" if avg_sentiment < 0.3 else "neutral" if avg_sentiment < 0.7 else "positive"
            }
        except Exception as e:
            self.logger.error(f"Error getting sentiment trends: {e}")
            return {}
    
    async def get_entity_clustering(self, since_date: datetime) -> Dict[str, Any]:
        """Get entity clustering patterns"""
        try:
            result = self.supabase.table("entities")\
                .select("entity_text, entity_type")\
                .gte("created_at", since_date.isoformat())\
                .execute()
            
            entities = result.data or []
            entity_freq = {}
            type_freq = {}
            
            for entity in entities:
                name = entity.get("entity_text", "")
                entity_type = entity.get("entity_type", "")
                entity_freq[name] = entity_freq.get(name, 0) + 1
                type_freq[entity_type] = type_freq.get(entity_type, 0) + 1
            
            return {
                "top_entities": sorted(entity_freq.items(), key=lambda x: x[1], reverse=True)[:10],
                "entity_types": type_freq
            }
        except Exception as e:
            self.logger.error(f"Error getting entity clustering: {e}")
            return {}
    
    async def get_geographic_patterns(self, since_date: datetime) -> Dict[str, Any]:
        """Get geographic distribution patterns"""
        try:
            result = self.supabase.table("events")\
                .select("location, event_type")\
                .gte("created_at", since_date.isoformat())\
                .execute()
            
            events = result.data or []
            location_freq = {}
            
            for event in events:
                location = event.get("location", "Unknown")
                location_freq[location] = location_freq.get(location, 0) + 1
            
            return {
                "hotspots": sorted(location_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        except Exception as e:
            self.logger.error(f"Error getting geographic patterns: {e}")
            return {}
    
    async def get_model_confidence(self) -> Dict[str, Any]:
        """Get model confidence metrics"""
        # This would interface with actual ML models
        return {
            "forecast_confidence": "medium",
            "uncertainty_level": "moderate",
            "data_quality": "good"
        }
    
    async def get_historical_baseline(self, since_date: datetime) -> Dict[str, Any]:
        """Get historical baseline for comparison"""
        # This would provide historical context
        return {
            "average_daily_events": 5,
            "baseline_severity": 0.3,
            "historical_trend": "stable"
        }
    
    # Helper formatting methods
    def format_frequency_analysis(self, freq_dict: Dict[str, int], category: str) -> str:
        """Format frequency analysis for reporting"""
        if not freq_dict:
            return f"No {category} patterns detected"
        
        sorted_items = sorted(freq_dict.items(), key=lambda x: x[1], reverse=True)
        lines = []
        for item, count in sorted_items[:5]:
            lines.append(f"  - {item}: {count} occurrences")
        
        return "\n".join(lines)
    
    def format_entity_patterns(self, entity_patterns: Dict[str, Any]) -> str:
        """Format entity patterns for reporting"""
        top_entities = entity_patterns.get("top_entities", [])
        if not top_entities:
            return "No significant entity patterns detected"
        
        lines = []
        for entity, count in top_entities[:5]:
            lines.append(f"  - {entity}: {count} mentions")
        
        return "\n".join(lines)
    
    def analyze_temporal_patterns(self, events: List[Dict[str, Any]]) -> str:
        """Analyze temporal distribution of events"""
        if not events:
            return "Insufficient data for temporal analysis"
        
        # Simple temporal analysis
        recent_count = len([e for e in events if "created_at" in e])
        return f"Event distribution: {recent_count} events detected across assessment period"
    
    def generate_forecast_interpretation(self, forecast: Dict[str, Any], trend: str, score: float) -> str:
        """Generate analyst interpretation of forecast"""
        if trend == "rising" and score > 0.6:
            return "Models predict elevated military signaling through Friday, with likelihood of continued naval activities in southeastern waters."
        elif trend == "stable":
            return "Forecast indicates stable tension levels with routine military posturing expected."
        else:
            return "Declining trend suggests temporary de-escalation, but monitoring continues for reversal indicators."
    
    def assess_strategic_risk(self, military: List, political: List, economic: List, cyber: List) -> str:
        """Assess overall strategic risk level"""
        total_indicators = len(military) + len(political) + len(economic) + len(cyber)
        
        if total_indicators > 6:
            return "HIGH - Multiple domain indicators suggest coordinated escalatory preparation"
        elif total_indicators > 3:
            return "MEDIUM - Moderate indicator activity across domains"
        else:
            return "LOW - Limited strategic warning indicators detected"
    
    def generate_confidence_assessment(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate overall confidence assessment"""
        events_count = len(data.get("recent_events", []))
        articles_count = len(data.get("recent_articles", []))
        
        data_quality = "high" if events_count > 10 and articles_count > 15 else "medium" if events_count > 5 else "low"
        
        return {
            "overall_confidence": "medium",
            "data_quality": data_quality,
            "collection_gaps": "Limited HUMINT correlation",
            "reliability": "Good technical indicators, moderate contextual depth"
        }
    
    async def save_intelligence_report(self, report_content: Dict[str, Any]) -> Dict[str, str]:
        """Save intelligence report in multiple formats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate formatted text report
        formatted_report = self.format_intelligence_report(report_content)
        
        # Save paths
        text_path = self.reports_dir / f"straitwatch_intelligence_{timestamp}.txt"
        json_path = self.reports_dir / f"straitwatch_intelligence_{timestamp}.json"
        
        # Save text format
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write(formatted_report)
        
        # Save JSON format
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_content, f, indent=2, default=str)
        
        return {
            "text": str(text_path),
            "json": str(json_path)
        }
    
    def format_intelligence_report(self, report: Dict[str, Any]) -> str:
        """Format intelligence report for text output"""
        header = report["header"]
        
        formatted = f"""
{header["classification"]}

{header["title"]}
{header["subtitle"]}

Date: {header["date"]} | Time: {header["time"]}
Originator: {header["originator"]}
Distribution: {header["distribution"]}

{'='*80}

1. EXECUTIVE SUMMARY
{'='*80}

{report["executive_summary"]}

{'='*80}

2. KEY DEVELOPMENTS
{'='*80}

{report["key_developments"]}

{'='*80}

3. PATTERN ANALYSIS
{'='*80}

{report["pattern_analysis"]}

{'='*80}

4. ESCALATION FORECAST (7-DAY OUTLOOK)
{'='*80}

{report["escalation_forecast"]}

{'='*80}

5. STRATEGIC WARNING INDICATORS
{'='*80}

{report["strategic_warning_indicators"]}

{'='*80}

6. SOURCE REFERENCE TABLE
{'='*80}

{report["source_reference_table"]}

{'='*80}

CONFIDENCE ASSESSMENT:
Overall Assessment Confidence: {report["confidence_assessment"]["overall_confidence"].upper()}
Data Quality: {report["confidence_assessment"]["data_quality"].upper()}
Collection Gaps: {report["confidence_assessment"]["collection_gaps"]}
Reliability: {report["confidence_assessment"]["reliability"]}

{'='*80}

{header["classification"]}
End of Report
"""
        
        return formatted
    
    async def store_intelligence_report(self, report_content: Dict[str, Any], report_paths: Dict[str, str]):
        """Store intelligence report metadata in database"""
        try:
            # Create intelligence_reports table entry
            report_record = {
                "report_date": date.today().isoformat(),
                "report_type": "daily_intelligence_assessment",
                "threat_level": report_content["threat_assessment"],
                "escalation_risk": report_content["escalation_risk"],
                "report_data": report_content,
                "file_path": report_paths.get("text"),
                "created_at": datetime.now().isoformat()
            }
            
            result = self.supabase.table("intelligence_reports")\
                .insert(report_record)\
                .execute()
            
            self.logger.info(f"Intelligence report stored in database: {result.data[0]['id'] if result.data else 'unknown'}")
            
        except Exception as e:
            self.logger.error(f"Error storing intelligence report: {e}")

async def main():
    """Test the report generator agent"""
    agent = ReportGeneratorAgent()
    result = await agent.safe_run()
    print(f"Report generation result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 