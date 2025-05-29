#!/usr/bin/env python3
"""
Demo script for StraitWatch Phase 1 Backend ML Pipeline

This script demonstrates the complete pipeline functionality with sample data.
"""
import json
from run_backend_pipeline import run_full_pipeline
from datetime import datetime

def create_demo_articles():
    """Create sample articles for demonstration."""
    return [
        {
            'id': 'demo_001',
            'title': 'Taiwan Military Exercise in Response to Regional Tensions',
            'translated_text': 'Taiwan military forces conducted large-scale naval exercises in the Taiwan Strait following recent diplomatic tensions with mainland China.',
            'source': 'demo_source',
            'date': datetime.now().isoformat()
        },
        {
            'id': 'demo_002',
            'title': 'Diplomatic Summit Scheduled for Peace Talks',
            'translated_text': 'Regional leaders announced a diplomatic summit aimed at reducing tensions and promoting peaceful dialogue.',
            'source': 'demo_source',
            'date': datetime.now().isoformat()
        },
        {
            'id': 'demo_003',
            'title': 'Cyber Security Alert Issued by Government',
            'translated_text': 'Government cybersecurity agencies warned of increased cyber attacks targeting critical infrastructure systems.',
            'source': 'demo_source',
            'date': datetime.now().isoformat()
        },
        {
            'id': 'demo_004',
            'title': 'Economic Cooperation Agreement Signed',
            'translated_text': 'Business leaders from multiple countries signed agreements to enhance economic cooperation and trade partnerships.',
            'source': 'demo_source',
            'date': datetime.now().isoformat()
        },
        {
            'id': 'demo_005',
            'title': 'Naval Forces Increase Patrol Activities',
            'translated_text': 'Naval forces in the region have increased patrol activities near disputed maritime boundaries.',
            'source': 'demo_source',
            'date': datetime.now().isoformat()
        }
    ]

def main():
    """Run the demo pipeline."""
    print("🚀 StraitWatch Backend Pipeline Demo")
    print("=" * 50)
    
    # Create demo articles
    demo_articles = create_demo_articles()
    print(f"📰 Created {len(demo_articles)} demo articles")
    
    # Save demo articles
    with open("data/demo_articles.json", 'w', encoding='utf-8') as f:
        json.dump(demo_articles, f, ensure_ascii=False, indent=2)
    print("💾 Saved demo articles to data/demo_articles.json")
    
    print("\n🔄 Running complete backend pipeline...")
    print("-" * 50)
    
    # Run the pipeline
    results = run_full_pipeline(
        articles=demo_articles,
        skip_forecasting=False  # Try forecasting but expect it to use default
    )
    
    print("\n📊 Pipeline Results Summary:")
    print("-" * 50)
    
    # Display results
    stats = results.get('pipeline_stats', {})
    print(f"Articles processed: {stats.get('articles_loaded', 0)}")
    print(f"Articles tagged: {stats.get('articles_tagged', 0)}")
    print(f"Articles scored: {stats.get('articles_scored', 0)}")
    print(f"Articles embedded: {stats.get('articles_embedded', 0)}")
    print(f"Clusters found: {stats.get('clusters_found', 0)}")
    print(f"Noise points: {stats.get('noise_points', 0)}")
    print(f"Average escalation score: {stats.get('avg_escalation_score', 0):.3f}")
    print(f"High-risk articles: {stats.get('high_risk_articles', 0)}")
    
    if results.get('tomorrow_escalation_forecast'):
        print(f"Tomorrow's escalation forecast: {results['tomorrow_escalation_forecast']:.3f}")
    
    print(f"\nTotal runtime: {results.get('total_runtime_seconds', 0):.1f} seconds")
    
    # Show sample processed articles
    processed_articles = results.get('processed_articles', [])
    if processed_articles:
        print("\n📝 Sample Processed Article:")
        print("-" * 50)
        sample = processed_articles[0]
        print(f"ID: {sample.get('id', 'N/A')}")
        print(f"Title: {sample.get('title', 'N/A')}")
        print(f"Tags: {sample.get('tags', [])}")
        print(f"Escalation Score: {sample.get('escalation_score', 0):.3f}")
        print(f"Cluster ID: {sample.get('cluster_id', 'N/A')}")
        print(f"Has Embedding: {'Yes' if 'embedding' in sample else 'No'}")
    
    print("\n✅ Demo completed successfully!")
    print("\n📁 Output Files:")
    print("- data/demo_articles.json (input)")
    print("- data/tagged_articles.json (after tagging)")
    print("- data/scored_articles.json (after inference)")
    print("- data/embedded_articles.json (after embedding)")
    print("- data/clustered_articles.json (final output)")
    print("- data/pipeline_summary.json (execution summary)")


if __name__ == "__main__":
    main() 