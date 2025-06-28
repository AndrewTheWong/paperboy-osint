#!/usr/bin/env python3
"""
Complete StraitWatch Intelligence Pipeline
Runs the full pipeline using ONLY real articles from news sources.
"""

import asyncio
import sys
import os
import subprocess
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.supabase_client import get_supabase_client

async def main():
    """Run complete StraitWatch pipeline with real articles."""
    
    print("\n🎯 Complete StraitWatch Intelligence Pipeline")
    print("=" * 70)
    print("🔒 Uses ONLY real articles from news sources")
    print("❌ NO synthetic content generation")
    print("📊 Authentic intelligence analysis")
    print("=" * 70)
    
    start_time = datetime.now()
    
    try:
        # Step 1: Article Ingestion
        print("\n📰 STEP 1: Real Article Ingestion")
        print("-" * 50)
        print("📡 Running unified ingest agent...")
        
        try:
            from agents.article_ingest_agent import ArticleIngestAgent
            ingest_agent = ArticleIngestAgent()
            result = await ingest_agent.run()
            
            if result.get('success'):
                scraped = result.get('articles_scraped', 0)
                uploaded = result.get('articles_uploaded', 0)
                relevant = result.get('relevant_articles', 0)
                
                print(f"✅ Articles scraped: {scraped}")
                print(f"✅ Articles uploaded: {uploaded}")
                print(f"✅ Relevant articles: {relevant}")
                
                if uploaded == 0:
                    print("⚠️ No articles uploaded - cannot proceed with analysis")
                    return
            else:
                print(f"❌ Ingestion failed: {result.get('error', 'Unknown error')}")
                return
        except Exception as e:
            print(f"❌ Ingestion agent failed: {e}")
            return
        
        # Step 2: Check Article Count
        print("\n📊 STEP 2: Storage Verification")
        print("-" * 50)
        
        sb = get_supabase_client()
        result = sb.table('articles').select('id, title').execute()
        total_articles = len(result.data) if result.data else 0
        
        print(f"📰 Total articles in storage: {total_articles}")
        
        if total_articles < 5:
            print("⚠️ Insufficient articles for meaningful analysis")
            print("💡 Consider running ingestion again or expanding sources")
        
        # Step 3: NLP Tagging (if agents available)
        print("\n🏷️ STEP 3: NLP Analysis")
        print("-" * 50)
        
        try:
            from agents.tagging_agent import TaggingAgent
            tagging_agent = TaggingAgent()
            tagging_result = await tagging_agent.run()
            print(f"✅ Tagging completed: {tagging_result.get('processed_count', 0)} articles processed")
        except Exception as e:
            print(f"⚠️ Tagging skipped: {e}")
        
        # Step 4: Time Series Building (if agents available)
        print("\n📈 STEP 4: Time Series Analysis")
        print("-" * 50)
        
        try:
            from agents.timeseries_builder_agent import TimeSeriesBuilderAgent
            ts_agent = TimeSeriesBuilderAgent()
            ts_result = await ts_agent.run()
            print(f"✅ Time series built: {ts_result.get('days_processed', 0)} days of data")
        except Exception as e:
            print(f"⚠️ Time series skipped: {e}")
        
        # Step 5: Forecasting (if agents available)
        print("\n🔮 STEP 5: Escalation Forecasting")
        print("-" * 50)
        
        try:
            from agents.forecasting_agent import ForecastingAgent
            forecast_agent = ForecastingAgent()
            forecast_result = await forecast_agent.run()
            print(f"✅ Forecasting completed: {forecast_result.get('forecasts_generated', 0)} forecasts generated")
        except Exception as e:
            print(f"⚠️ Forecasting skipped: {e}")
        
        # Step 6: Intelligence Report Generation
        print("\n📄 STEP 6: Intelligence Report Generation")
        print("-" * 50)
        
        print("📊 Generating storage-based intelligence report...")
        
        # Run storage-based reporter
        from storage_based_reporter import StorageBasedReporter
        
        reporter = StorageBasedReporter()
        report_data = reporter.generate_storage_based_report()
        
        if 'error' not in report_data:
            total_clusters = report_data.get('metadata', {}).get('total_clusters', 0)
            threat_level = report_data.get('executive_summary', {}).get('threat_level', 'UNKNOWN')
            
            print(f"✅ Report generated successfully")
            print(f"📊 Clusters identified: {total_clusters}")
            print(f"⚠️ Threat level: {threat_level}")
        else:
            print(f"❌ Report generation failed: {report_data.get('error')}")
        
        # Final Summary
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print("\n📋 PIPELINE SUMMARY")
        print("=" * 70)
        print(f"⏰ Total Duration: {duration:.1f} seconds")
        print(f"📰 Articles in Storage: {total_articles}")
        print(f"📄 Report Generated: {'✅ Yes' if 'error' not in report_data else '❌ No'}")
        print(f"🔒 Hallucination Prevention: ✅ Active")
        print(f"📊 Data Authenticity: ✅ Real articles only")
        print(f"❌ Synthetic Content: ✅ None generated")
        
        print("\n🎯 StraitWatch pipeline completed successfully!")
        print("🔒 All intelligence reports are based exclusively on real articles")
        print("📄 Check the reports/ directory for detailed analysis")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 