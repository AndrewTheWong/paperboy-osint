"""
Time Series Builder Agent for StraitWatch
Constructs daily escalation time series from tagged articles and events
"""

import asyncio
import pandas as pd
import logging
from datetime import datetime, timedelta, date
from typing import Dict, Any, List
from pathlib import Path
import os

from .base_agent import BaseAgent

class TimeSeriesBuilderAgent(BaseAgent):
    """Agent responsible for building daily escalation time series"""
    
    def __init__(self):
        super().__init__("timeseries_builder_agent")
        
        # Ensure output directory exists
        self.output_dir = Path("data/time_series")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    async def run(self) -> Dict[str, Any]:
        """Main time series building workflow"""
        self.logger.info("Starting time series construction")
        
        try:
            # Build escalation series for last 180 days
            series_data = await self.build_escalation_series(days=180)
            
            if not series_data:
                self.logger.warning("No time series data generated")
                return {"success": False, "error": "No data"}
            
            # Convert to DataFrame
            df = pd.DataFrame(series_data)
            
            # Save to CSV
            output_path = self.output_dir / "escalation_series.csv"
            df.to_csv(output_path, index=False)
            
            # Generate summary stats
            stats = self.calculate_summary_stats(df)
            
            result = {
                "success": True,
                "output_path": str(output_path),
                "days_processed": len(df),
                "date_range": {
                    "start": df['date'].min() if not df.empty else None,
                    "end": df['date'].max() if not df.empty else None
                },
                "summary_stats": stats
            }
            
            self.logger.info(f"Time series built: {len(df)} days, saved to {output_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Time series construction failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def build_escalation_series(self, days: int = 180) -> List[Dict[str, Any]]:
        """Build daily escalation time series"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        self.logger.info(f"Building escalation series from {start_date} to {end_date}")
        
        series_data = []
        current_date = start_date
        
        while current_date <= end_date:
            daily_data = await self.get_daily_metrics(current_date)
            series_data.append(daily_data)
            current_date += timedelta(days=1)
        
        return series_data
    
    async def get_daily_metrics(self, target_date: date) -> Dict[str, Any]:
        """Get metrics for a specific date"""
        date_str = target_date.isoformat()
        next_date = target_date + timedelta(days=1)
        next_date_str = next_date.isoformat()
        
        try:
            # Count relevant tagged events
            event_count = await self.count_daily_events(date_str, next_date_str)
            
            # Get average escalation score
            avg_escalation = await self.get_daily_escalation_score(date_str, next_date_str)
            
            # Get sentiment metrics
            sentiment_metrics = await self.get_daily_sentiment(date_str, next_date_str)
            
            # Get entity mentions
            entity_counts = await self.get_daily_entity_counts(date_str, next_date_str)
            
            return {
                "date": date_str,
                "event_count": event_count,
                "avg_escalation_score": avg_escalation,
                "sentiment_score": sentiment_metrics.get("avg_sentiment", 0.5),
                "entity_mentions": entity_counts,
                "tension_level": self.calculate_tension_level(event_count, avg_escalation, sentiment_metrics)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting daily metrics for {date_str}: {e}")
            return {
                "date": date_str,
                "event_count": 0,
                "avg_escalation_score": 0.0,
                "sentiment_score": 0.5,
                "entity_mentions": 0,
                "tension_level": "low"
            }
    
    async def count_daily_events(self, start_date: str, end_date: str) -> int:
        """Count events for a specific day"""
        try:
            # Count from article_tags
            tags_result = self.supabase.table("article_tags")\
                .select("id", count="exact")\
                .eq("tag_type", "event")\
                .gte("created_at", start_date)\
                .lt("created_at", end_date)\
                .execute()
            
            tags_count = tags_result.count or 0
            
            # Count from events table
            events_result = self.supabase.table("events")\
                .select("id", count="exact")\
                .gte("created_at", start_date)\
                .lt("created_at", end_date)\
                .execute()
            
            events_count = events_result.count or 0
            
            return max(tags_count, events_count)  # Use higher count
            
        except Exception as e:
            self.logger.error(f"Error counting daily events: {e}")
            return 0
    
    async def get_daily_escalation_score(self, start_date: str, end_date: str) -> float:
        """Get average escalation score for a day"""
        try:
            # Get escalation tags
            result = self.supabase.table("article_tags")\
                .select("tag_data, confidence")\
                .eq("tag_type", "escalation")\
                .gte("created_at", start_date)\
                .lt("created_at", end_date)\
                .execute()
            
            if not result.data:
                return 0.0
            
            scores = []
            for tag in result.data:
                tag_data = tag.get("tag_data", {})
                if isinstance(tag_data, dict):
                    escalation_score = tag_data.get("escalation_score", 0.0)
                    confidence = tag.get("confidence", 0.0)
                    # Weight by confidence
                    weighted_score = escalation_score * confidence
                    scores.append(weighted_score)
            
            if scores:
                return sum(scores) / len(scores)
            else:
                return 0.0
                
        except Exception as e:
            self.logger.error(f"Error getting escalation score: {e}")
            return 0.0
    
    async def get_daily_sentiment(self, start_date: str, end_date: str) -> Dict[str, float]:
        """Get sentiment metrics for a day"""
        try:
            result = self.supabase.table("article_sentiment")\
                .select("sentiment_score, confidence")\
                .gte("created_at", start_date)\
                .lt("created_at", end_date)\
                .execute()
            
            if not result.data:
                return {"avg_sentiment": 0.5, "sentiment_count": 0}
            
            sentiment_scores = []
            for sentiment in result.data:
                score = sentiment.get("sentiment_score", 0.5)
                confidence = sentiment.get("confidence", 0.0)
                weighted_score = score * confidence
                sentiment_scores.append(weighted_score)
            
            avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.5
            
            return {
                "avg_sentiment": avg_sentiment,
                "sentiment_count": len(sentiment_scores)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment: {e}")
            return {"avg_sentiment": 0.5, "sentiment_count": 0}
    
    async def get_daily_entity_counts(self, start_date: str, end_date: str) -> int:
        """Count entity mentions for a day"""
        try:
            result = self.supabase.table("entities")\
                .select("id", count="exact")\
                .gte("created_at", start_date)\
                .lt("created_at", end_date)\
                .execute()
            
            return result.count or 0
            
        except Exception as e:
            self.logger.error(f"Error counting entities: {e}")
            return 0
    
    def calculate_tension_level(self, event_count: int, escalation_score: float, sentiment_metrics: Dict[str, float]) -> str:
        """Calculate overall tension level for the day"""
        # Normalize event count (assume 0-20 events per day is normal range)
        normalized_events = min(event_count / 20.0, 1.0)
        
        # Sentiment factor (lower sentiment = higher tension)
        sentiment_factor = 1.0 - sentiment_metrics.get("avg_sentiment", 0.5)
        
        # Combined tension score
        tension_score = (normalized_events * 0.4 + escalation_score * 0.4 + sentiment_factor * 0.2)
        
        if tension_score >= 0.7:
            return "high"
        elif tension_score >= 0.4:
            return "medium"
        else:
            return "low"
    
    def calculate_summary_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate summary statistics for the time series"""
        if df.empty:
            return {}
        
        try:
            stats = {
                "total_days": len(df),
                "avg_daily_events": df['event_count'].mean(),
                "max_daily_events": df['event_count'].max(),
                "avg_escalation_score": df['avg_escalation_score'].mean(),
                "max_escalation_score": df['avg_escalation_score'].max(),
                "tension_distribution": df['tension_level'].value_counts().to_dict(),
                "data_quality": {
                    "non_zero_days": (df['event_count'] > 0).sum(),
                    "complete_data_days": df.dropna().shape[0]
                }
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating stats: {e}")
            return {}

async def main():
    """Test the time series builder agent"""
    agent = TimeSeriesBuilderAgent()
    result = await agent.safe_run()
    print(f"Time series result: {result}")

if __name__ == "__main__":
    asyncio.run(main()) 