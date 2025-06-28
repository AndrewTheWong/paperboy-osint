#!/usr/bin/env python3
"""
Time Series Data Processor

This module handles data preparation, feature engineering, and preprocessing
for time series analysis of escalation patterns.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeSeriesProcessor:
    """
    Comprehensive time series data processor for escalation analysis.
    
    Features:
    - Multi-source data aggregation
    - Feature engineering for temporal patterns
    - Data quality assessment
    - Missing data handling
    - Trend and seasonality analysis
    """
    
    def __init__(self, output_dir: str = "analytics/time_series/processed"):
        """Initialize time series processor."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.raw_data = None
        self.processed_data = None
        self.features = None
        self.metadata = {}
        
        logger.info("TimeSeriesProcessor initialized")
    
    def load_articles_data(self, articles: List[Dict], escalation_scores: List[float]) -> pd.DataFrame:
        """
        Load and structure articles data for time series analysis.
        
        Args:
            articles: List of article dictionaries
            escalation_scores: List of escalation scores
            
        Returns:
            Structured DataFrame with temporal information
        """
        try:
            data_points = []
            
            for i, (article, score) in enumerate(zip(articles, escalation_scores)):
                try:
                    # Extract temporal information
                    date_str = (article.get('scraped_at') or 
                               article.get('published_at') or 
                               article.get('created_at') or
                               article.get('date'))
                    
                    if date_str:
                        date = pd.to_datetime(date_str, errors='coerce')
                    else:
                        # Generate synthetic dates if none available
                        base_date = datetime.now() - timedelta(days=len(articles)-i)
                        date = base_date
                    
                    # Extract article features
                    data_point = {
                        'date': date,
                        'escalation_score': score,
                        'title': article.get('title', ''),
                        'content': article.get('content', ''),
                        'url': article.get('url', ''),
                        'source': article.get('source', 'unknown'),
                        'region': self._extract_region(article),
                        'topic': self._extract_topic(article),
                        'article_length': len(article.get('content', '')),
                        'title_length': len(article.get('title', '')),
                        'hour': date.hour if pd.notna(date) else 0,
                        'day_of_week': date.dayofweek if pd.notna(date) else 0,
                        'month': date.month if pd.notna(date) else 1
                    }
                    
                    data_points.append(data_point)
                    
                except Exception as e:
                    logger.warning(f"Failed to process article {i}: {e}")
                    continue
            
            # Create DataFrame
            df = pd.DataFrame(data_points)
            df = df.dropna(subset=['date'])
            df = df.sort_values('date').reset_index(drop=True)
            
            logger.info(f"Loaded {len(df)} articles with temporal data")
            logger.info(f"Date range: {df['date'].min()} to {df['date'].max()}")
            
            self.raw_data = df
            return df
            
        except Exception as e:
            logger.error(f"Failed to load articles data: {e}")
            return pd.DataFrame()
    
    def _extract_region(self, article: Dict) -> str:
        """Extract geographical region from article."""
        content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
        
        regions = {
            'taiwan': ['taiwan', 'taipei', 'kaohsiung'],
            'china': ['china', 'beijing', 'shanghai', 'chinese'],
            'usa': ['usa', 'united states', 'america', 'washington'],
            'japan': ['japan', 'tokyo', 'japanese'],
            'south_korea': ['south korea', 'seoul', 'korean'],
            'north_korea': ['north korea', 'pyongyang', 'dprk']
        }
        
        for region, keywords in regions.items():
            if any(keyword in content for keyword in keywords):
                return region
        
        return 'other'
    
    def _extract_topic(self, article: Dict) -> str:
        """Extract topic category from article."""
        content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
        
        topics = {
            'military': ['military', 'army', 'navy', 'exercise', 'drill', 'weapon', 'defense'],
            'diplomatic': ['diplomatic', 'embassy', 'ambassador', 'negotiation', 'talks'],
            'economic': ['trade', 'economic', 'tariff', 'business', 'market', 'investment'],
            'technology': ['technology', 'cyber', 'tech', 'semiconductor', 'chip'],
            'politics': ['political', 'election', 'government', 'policy', 'parliament']
        }
        
        for topic, keywords in topics.items():
            if any(keyword in content for keyword in keywords):
                return topic
        
        return 'general'
    
    def aggregate_by_time(self, df: pd.DataFrame, freq: str = 'D') -> pd.DataFrame:
        """
        Aggregate data by time periods.
        
        Args:
            df: Raw articles DataFrame
            freq: Frequency for aggregation ('D', 'H', 'W', 'M')
            
        Returns:
            Time-aggregated DataFrame
        """
        try:
            # Set date as index
            df_indexed = df.set_index('date')
            
            # Define aggregation functions
            agg_functions = {
                'escalation_score': ['mean', 'std', 'min', 'max', 'count'],
                'article_length': ['mean', 'sum'],
                'title_length': ['mean'],
                'region': lambda x: x.value_counts().index[0] if len(x) > 0 else 'other',
                'topic': lambda x: x.value_counts().index[0] if len(x) > 0 else 'general'
            }
            
            # Aggregate by frequency
            aggregated = df_indexed.resample(freq).agg(agg_functions)
            
            # Flatten column names
            aggregated.columns = ['_'.join(col).strip() for col in aggregated.columns.values]
            
            # Add time-based features
            aggregated['hour_avg'] = df_indexed.resample(freq)['hour'].mean()
            aggregated['day_of_week_mode'] = df_indexed.resample(freq)['day_of_week'].apply(
                lambda x: x.value_counts().index[0] if len(x) > 0 else 0
            )
            
            # Forward fill missing values
            aggregated = aggregated.fillna(method='ffill')
            
            # Add lag features
            aggregated['escalation_score_mean_lag1'] = aggregated['escalation_score_mean'].shift(1)
            aggregated['escalation_score_mean_lag7'] = aggregated['escalation_score_mean'].shift(7)
            aggregated['escalation_score_mean_lag30'] = aggregated['escalation_score_mean'].shift(30)
            
            # Add rolling statistics
            aggregated['escalation_score_ma7'] = aggregated['escalation_score_mean'].rolling(window=7).mean()
            aggregated['escalation_score_ma30'] = aggregated['escalation_score_mean'].rolling(window=30).mean()
            aggregated['escalation_score_std7'] = aggregated['escalation_score_mean'].rolling(window=7).std()
            
            logger.info(f"Aggregated data by {freq}: {len(aggregated)} periods")
            
            self.processed_data = aggregated
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate data: {e}")
            return pd.DataFrame()
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engineered features for time series modeling.
        
        Args:
            df: Aggregated time series DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        try:
            features_df = df.copy()
            
            # Temporal features
            features_df['year'] = features_df.index.year
            features_df['month'] = features_df.index.month
            features_df['day'] = features_df.index.day
            features_df['quarter'] = features_df.index.quarter
            features_df['day_of_year'] = features_df.index.dayofyear
            features_df['week_of_year'] = features_df.index.isocalendar().week
            features_df['is_weekend'] = (features_df.index.dayofweek >= 5).astype(int)
            features_df['is_month_start'] = features_df.index.is_month_start.astype(int)
            features_df['is_month_end'] = features_df.index.is_month_end.astype(int)
            
            # Cyclical features
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day'] / 31)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day'] / 31)
            features_df['dow_sin'] = np.sin(2 * np.pi * features_df.index.dayofweek / 7)
            features_df['dow_cos'] = np.cos(2 * np.pi * features_df.index.dayofweek / 7)
            
            # Trend features
            features_df['time_trend'] = range(len(features_df))
            features_df['time_trend_sq'] = features_df['time_trend'] ** 2
            
            # Volatility features
            if 'escalation_score_mean' in features_df.columns:
                features_df['volatility_7d'] = features_df['escalation_score_mean'].rolling(window=7).std()
                features_df['volatility_30d'] = features_df['escalation_score_mean'].rolling(window=30).std()
                
                # Change features
                features_df['escalation_change'] = features_df['escalation_score_mean'].diff()
                features_df['escalation_pct_change'] = features_df['escalation_score_mean'].pct_change()
                
                # Momentum features
                features_df['momentum_3d'] = features_df['escalation_score_mean'].diff(3)
                features_df['momentum_7d'] = features_df['escalation_score_mean'].diff(7)
                
                # Relative position features
                features_df['position_in_30d_range'] = (
                    (features_df['escalation_score_mean'] - 
                     features_df['escalation_score_mean'].rolling(30).min()) /
                    (features_df['escalation_score_mean'].rolling(30).max() - 
                     features_df['escalation_score_mean'].rolling(30).min())
                )
            
            logger.info(f"Created {len(features_df.columns)} features")
            
            self.features = features_df
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to create features: {e}")
            return df
    
    def assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess data quality and completeness.
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Data quality metrics
        """
        try:
            quality_metrics = {
                'total_records': len(df),
                'date_range': {
                    'start': str(df.index.min()) if hasattr(df.index, 'min') else 'N/A',
                    'end': str(df.index.max()) if hasattr(df.index, 'max') else 'N/A',
                    'days': (df.index.max() - df.index.min()).days if hasattr(df.index, 'min') else 0
                },
                'missing_values': {},
                'data_consistency': {},
                'outliers': {}
            }
            
            # Missing values analysis
            for col in df.columns:
                missing_count = df[col].isnull().sum()
                quality_metrics['missing_values'][col] = {
                    'count': int(missing_count),
                    'percentage': float(missing_count / len(df) * 100)
                }
            
            # Data consistency checks
            if 'escalation_score_mean' in df.columns:
                escalation_col = df['escalation_score_mean']
                quality_metrics['data_consistency']['escalation_score'] = {
                    'min': float(escalation_col.min()),
                    'max': float(escalation_col.max()),
                    'mean': float(escalation_col.mean()),
                    'std': float(escalation_col.std()),
                    'negative_values': int((escalation_col < 0).sum()),
                    'zero_values': int((escalation_col == 0).sum())
                }
                
                # Outlier detection using IQR method
                Q1 = escalation_col.quantile(0.25)
                Q3 = escalation_col.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = escalation_col[(escalation_col < lower_bound) | (escalation_col > upper_bound)]
                quality_metrics['outliers']['escalation_score'] = {
                    'count': len(outliers),
                    'percentage': float(len(outliers) / len(escalation_col) * 100),
                    'values': outliers.tolist()
                }
            
            # Frequency analysis
            if hasattr(df.index, 'freq'):
                quality_metrics['frequency'] = str(df.index.freq)
            else:
                # Infer frequency
                if len(df) > 1:
                    time_diffs = df.index.to_series().diff().dropna()
                    most_common_diff = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else timedelta(days=1)
                    quality_metrics['frequency'] = str(most_common_diff)
            
            logger.info(f"Data quality assessment completed")
            logger.info(f"Records: {quality_metrics['total_records']}, Date range: {quality_metrics['date_range']['days']} days")
            
            self.metadata['quality_metrics'] = quality_metrics
            return quality_metrics
            
        except Exception as e:
            logger.error(f"Failed to assess data quality: {e}")
            return {}
    
    def create_visualization(self, df: pd.DataFrame) -> bool:
        """
        Create visualization of time series data and features.
        
        Args:
            df: DataFrame to visualize
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Create comprehensive visualization
            fig, axes = plt.subplots(3, 2, figsize=(16, 18))
            fig.suptitle('Time Series Analysis Overview', fontsize=16, fontweight='bold')
            
            # Plot 1: Main escalation time series
            if 'escalation_score_mean' in df.columns:
                axes[0, 0].plot(df.index, df['escalation_score_mean'], 'b-', alpha=0.7, linewidth=1.5)
                if 'escalation_score_ma7' in df.columns:
                    axes[0, 0].plot(df.index, df['escalation_score_ma7'], 'r-', alpha=0.8, linewidth=2, label='7-day MA')
                if 'escalation_score_ma30' in df.columns:
                    axes[0, 0].plot(df.index, df['escalation_score_ma30'], 'g-', alpha=0.8, linewidth=2, label='30-day MA')
                axes[0, 0].set_title('Escalation Score Over Time')
                axes[0, 0].set_ylabel('Escalation Score')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
            
            # Plot 2: Article count
            if 'escalation_score_count' in df.columns:
                axes[0, 1].bar(df.index, df['escalation_score_count'], alpha=0.7, color='orange')
                axes[0, 1].set_title('Daily Article Count')
                axes[0, 1].set_ylabel('Number of Articles')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Plot 3: Volatility
            if 'volatility_7d' in df.columns:
                axes[1, 0].plot(df.index, df['volatility_7d'], 'purple', alpha=0.7)
                axes[1, 0].set_title('7-Day Escalation Volatility')
                axes[1, 0].set_ylabel('Standard Deviation')
                axes[1, 0].grid(True, alpha=0.3)
            
            # Plot 4: Weekly pattern
            if 'escalation_score_mean' in df.columns and len(df) > 7:
                weekly_pattern = df.groupby(df.index.dayofweek)['escalation_score_mean'].mean()
                days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                axes[1, 1].bar(days, weekly_pattern.values, alpha=0.7, color='green')
                axes[1, 1].set_title('Average Escalation by Day of Week')
                axes[1, 1].set_ylabel('Average Escalation Score')
                axes[1, 1].grid(True, alpha=0.3)
            
            # Plot 5: Monthly pattern
            if 'escalation_score_mean' in df.columns and len(df) > 30:
                monthly_pattern = df.groupby(df.index.month)['escalation_score_mean'].mean()
                months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                         'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                axes[2, 0].bar(months[:len(monthly_pattern)], monthly_pattern.values, alpha=0.7, color='red')
                axes[2, 0].set_title('Average Escalation by Month')
                axes[2, 0].set_ylabel('Average Escalation Score')
                axes[2, 0].tick_params(axis='x', rotation=45)
                axes[2, 0].grid(True, alpha=0.3)
            
            # Plot 6: Distribution
            if 'escalation_score_mean' in df.columns:
                axes[2, 1].hist(df['escalation_score_mean'].dropna(), bins=30, alpha=0.7, color='blue')
                axes[2, 1].set_title('Escalation Score Distribution')
                axes[2, 1].set_xlabel('Escalation Score')
                axes[2, 1].set_ylabel('Frequency')
                axes[2, 1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save visualization
            viz_path = self.output_dir / 'time_series_overview.png'
            plt.savefig(viz_path, dpi=300, bbox_inches='tight')
            logger.info(f"Time series visualization saved to {viz_path}")
            
            plt.close()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
            return False
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = None) -> bool:
        """
        Save processed time series data.
        
        Args:
            df: Processed DataFrame
            filename: Optional filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"processed_timeseries_{timestamp}.csv"
            
            # Save CSV
            csv_path = self.output_dir / filename
            df.to_csv(csv_path)
            
            # Save metadata
            metadata_path = self.output_dir / filename.replace('.csv', '_metadata.json')
            import json
            with open(metadata_path, 'w') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            
            logger.info(f"Processed data saved to {csv_path}")
            logger.info(f"Metadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {e}")
            return False
    
    def process_pipeline(self, articles: List[Dict], escalation_scores: List[float], 
                        freq: str = 'D') -> pd.DataFrame:
        """
        Complete processing pipeline.
        
        Args:
            articles: List of article dictionaries
            escalation_scores: List of escalation scores
            freq: Aggregation frequency
            
        Returns:
            Fully processed DataFrame
        """
        try:
            logger.info("Starting time series processing pipeline...")
            
            # Load data
            raw_df = self.load_articles_data(articles, escalation_scores)
            if raw_df.empty:
                logger.error("Failed to load data")
                return pd.DataFrame()
            
            # Aggregate by time
            agg_df = self.aggregate_by_time(raw_df, freq)
            if agg_df.empty:
                logger.error("Failed to aggregate data")
                return pd.DataFrame()
            
            # Create features
            features_df = self.create_features(agg_df)
            
            # Assess quality
            quality_metrics = self.assess_data_quality(features_df)
            
            # Create visualization
            self.create_visualization(features_df)
            
            # Save processed data
            self.save_processed_data(features_df)
            
            logger.info("Time series processing pipeline completed successfully")
            logger.info(f"Final dataset: {len(features_df)} records, {len(features_df.columns)} features")
            
            return features_df
            
        except Exception as e:
            logger.error(f"Processing pipeline failed: {e}")
            return pd.DataFrame() 