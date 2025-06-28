#!/usr/bin/env python3
"""
Live GDELT Pipeline with Geographic Extraction
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import json
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from google.cloud import bigquery
from supabase import create_client
from pipelines.tagging.geographic_extractor import extract_geographic_info

logger = logging.getLogger(__name__)

class LiveGDELTPipeline:
    """Live GDELT data pipeline with geographic extraction."""
    
    def __init__(self):
        """Initialize the pipeline."""
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_KEY')
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase credentials not provided")
        
        self.supabase_client = create_client(self.supabase_url, self.supabase_key)
        self.bigquery_client = self._initialize_bigquery()
        self.chunk_size = 100
        
    def _initialize_bigquery(self) -> bigquery.Client:
        """Initialize BigQuery client."""
        credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        if not credentials_path:
            raise FileNotFoundError("GOOGLE_APPLICATION_CREDENTIALS not found")
        
        return bigquery.Client()
    
    def pull_daily_gdelt_data(self, days_back: int = 7) -> Optional[pd.DataFrame]:
        """Pull GDELT data from the last N days."""
        logger.info(f"Pulling GDELT data for last {days_back} day(s)...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        start_date_str = start_date.strftime('%Y%m%d')
        end_date_str = end_date.strftime('%Y%m%d')
        
        query = f"""
        SELECT
            PARSE_DATE('%Y%m%d', SQLDATE) as event_date,
            Actor1CountryCode as actor1_country,
            Actor2CountryCode as actor2_country,
            EventCode as event_code,
            GoldsteinScale as goldstein,
            AvgTone as avg_tone,
            NumMentions as num_mentions,
            ActionGeo_FullName as location_name,
            ActionGeo_CountryCode as country_code,
            ActionGeo_Lat as latitude,
            ActionGeo_Long as longitude
        FROM `gdelt-bq.gdeltv2.events`
        WHERE SQLDATE BETWEEN '{start_date_str}' AND '{end_date_str}'
        AND Actor1CountryCode IN ('USA', 'CHN', 'TWN', 'JPN', 'KOR')
        LIMIT 5000
        """
        
        try:
            query_job = self.bigquery_client.query(query)
            df = query_job.to_dataframe()
            logger.info(f"✅ Pulled {len(df)} GDELT events")
            return df
        except Exception as e:
            logger.error(f"❌ Failed to execute GDELT query: {e}")
            return None
    
    def enhance_with_geographic_tags(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhance GDELT data with geographic tags."""
        logger.info("Enhancing GDELT data with geographic tags...")
        
        enhanced_data = []
        for idx, row in df.iterrows():
            # Extract text for geographic analysis
            text_components = []
            if pd.notna(row.get('location_name')):
                text_components.append(str(row['location_name']))
            
            text_for_extraction = " ".join(text_components)
            
            # Extract geographic information
            if text_for_extraction.strip():
                geo_info = extract_geographic_info(text_for_extraction)
            else:
                geo_info = {"primary_country": None, "all_locations": [], "geographic_confidence": 0.0}
            
            # Create enhanced record
            enhanced_record = row.to_dict()
            enhanced_record.update({
                'primary_country': geo_info.get('primary_country'),
                'all_locations': json.dumps(geo_info.get('all_locations', [])),
                'geographic_confidence': geo_info.get('geographic_confidence', 0.0)
            })
            
            enhanced_data.append(enhanced_record)
        
        return pd.DataFrame(enhanced_data)
    
    def upload_to_supabase(self, df: pd.DataFrame) -> bool:
        """Upload enhanced data to Supabase."""
        logger.info(f"Uploading {len(df)} records to Supabase...")
        
        # Create table if it doesn't exist
        create_sql = """
        CREATE TABLE IF NOT EXISTS gdelt_events_enhanced (
            id BIGSERIAL PRIMARY KEY,
            event_date DATE,
            actor1_country TEXT,
            actor2_country TEXT,
            event_code TEXT,
            goldstein FLOAT,
            avg_tone FLOAT,
            num_mentions INTEGER,
            location_name TEXT,
            country_code TEXT,
            latitude FLOAT,
            longitude FLOAT,
            primary_country TEXT,
            all_locations JSONB,
            geographic_confidence FLOAT,
            created_at TIMESTAMP DEFAULT NOW()
        );
        """
        
        try:
            self.supabase_client.postgrest.rpc("execute_sql", {"sql": create_sql})
        except Exception as e:
            logger.warning(f"Table creation issue: {e}")
        
        # Upload data in chunks
        total_uploaded = 0
        for i in range(0, len(df), self.chunk_size):
            chunk = df.iloc[i:i + self.chunk_size]
            
            try:
                records = []
                for _, row in chunk.iterrows():
                    record = {
                        'event_date': row['event_date'].strftime('%Y-%m-%d') if pd.notna(row['event_date']) else None,
                        'actor1_country': row.get('actor1_country'),
                        'actor2_country': row.get('actor2_country'),
                        'event_code': row.get('event_code'),
                        'goldstein': float(row['goldstein']) if pd.notna(row['goldstein']) else None,
                        'avg_tone': float(row['avg_tone']) if pd.notna(row['avg_tone']) else None,
                        'num_mentions': int(row['num_mentions']) if pd.notna(row['num_mentions']) else None,
                        'location_name': row.get('location_name'),
                        'country_code': row.get('country_code'),
                        'latitude': float(row['latitude']) if pd.notna(row['latitude']) else None,
                        'longitude': float(row['longitude']) if pd.notna(row['longitude']) else None,
                        'primary_country': row.get('primary_country'),
                        'all_locations': row.get('all_locations'),
                        'geographic_confidence': float(row['geographic_confidence']) if pd.notna(row['geographic_confidence']) else 0.0
                    }
                    records.append(record)
                
                response = self.supabase_client.table('gdelt_events_enhanced').insert(records).execute()
                
                if response.data:
                    total_uploaded += len(records)
                    logger.info(f"✅ Uploaded chunk {i//self.chunk_size + 1}")
                    
            except Exception as e:
                logger.error(f"❌ Error uploading chunk: {e}")
        
        logger.info(f"Upload complete: {total_uploaded} records uploaded")
        return total_uploaded > 0
    
    def run_pipeline(self, days_back: int = 1) -> Dict[str, Any]:
        """Run the complete live GDELT pipeline."""
        logger.info("🚀 Starting Live GDELT Pipeline")
        
        results = {'status': 'failed', 'records_pulled': 0, 'records_uploaded': 0}
        
        try:
            # Pull GDELT data
            df = self.pull_daily_gdelt_data(days_back)
            if df is None or len(df) == 0:
                return results
            
            results['records_pulled'] = len(df)
            
            # Enhance with geographic tags
            enhanced_df = self.enhance_with_geographic_tags(df)
            
            # Upload to Supabase
            upload_success = self.upload_to_supabase(enhanced_df)
            
            if upload_success:
                results['status'] = 'success'
                results['records_uploaded'] = len(enhanced_df)
            
            logger.info("🎉 Live GDELT Pipeline completed!")
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
        
        return results

def main():
    """Main function."""
    pipeline = LiveGDELTPipeline()
    results = pipeline.run_pipeline()
    print(f"Results: {results}")

if __name__ == '__main__':
    main() 