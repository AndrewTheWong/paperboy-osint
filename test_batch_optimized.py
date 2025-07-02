#!/usr/bin/env python3

"""
Test the optimized batch processing pipeline with Supabase batch storage
"""

import sys
import os
sys.path.append('.')

import requests
import time
import json

def test_optimized_batch():
    """Test the new optimized batch processing endpoint"""
    
    # Test articles - mix of different topics and regions
    test_articles = [
        {
            "title": "Taiwan Maritime Security Enhancement 2025",
            "body": """
            <p>Taiwan has announced significant enhancements to its maritime 
            security infrastructure following increased regional tensions. 
            The upgrades include advanced radar systems and cybersecurity 
            measures for port facilities.</p>
            
            <p>Military experts believe these improvements are crucial for 
            maintaining Taiwan's strategic position in the South China Sea.</p>
            """,
            "region": "East Asia",
            "topic": "Maritime Security",
            "source_url": "https://example.com/taiwan-maritime-security"
        },
        {
            "title": "Singapore Cybersecurity Initiative 2025",
            "body": """
            <p>Singapore has launched a comprehensive cybersecurity initiative 
            targeting critical infrastructure protection. The program focuses 
            on maritime shipping networks and financial systems.</p>
            
            <p>The initiative includes partnerships with Malaysia and Indonesia 
            to strengthen regional cyber defenses against sophisticated threats.</p>
            """,
            "region": "Southeast Asia", 
            "topic": "Cybersecurity",
            "source_url": "https://example.com/singapore-cyber-initiative"
        },
        {
            "title": "South China Sea Naval Exercise Alert",
            "body": """
            <p>Multiple nations are conducting naval exercises in disputed 
            waters of the South China Sea. Intelligence reports indicate 
            increased submarine activity and surveillance operations.</p>
            
            <p>The exercises involve coordination between allied naval forces 
            and are seen as a response to regional security concerns.</p>
            """,
            "region": "South China Sea",
            "topic": "Military Operations",
            "source_url": "https://example.com/scs-naval-exercise"
        },
        {
            "title": "Indonesia Port Security Vulnerabilities",
            "body": """
            <p>Security assessments have revealed vulnerabilities in Indonesian 
            port systems that could be exploited by cyber attackers. The 
            vulnerabilities affect cargo tracking and vessel management systems.</p>
            
            <p>Indonesian authorities are working with international partners 
            to address these security gaps and implement stronger protections.</p>
            """,
            "region": "Southeast Asia",
            "topic": "Port Security", 
            "source_url": "https://example.com/indonesia-port-security"
        },
        {
            "title": "Philippines Maritime Border Tensions",
            "body": """
            <p>Border tensions in the Philippines maritime zones have escalated 
            following recent incidents involving foreign vessels. Philippine 
            Coast Guard has increased patrols in disputed areas.</p>
            
            <p>The incidents have prompted discussions about regional maritime 
            security cooperation and intelligence sharing protocols.</p>
            """,
            "region": "Southeast Asia",
            "topic": "Border Security",
            "source_url": "https://example.com/philippines-border-tensions"
        }
    ]
    
    print("🚀 Testing Optimized Batch Processing")
    print("=" * 60)
    
    # Test different batch sizes
    for batch_size in [2, 5, 10]:
        print(f"\n📦 Testing batch_size={batch_size}")
        print("-" * 40)
        
        # Send batch request
        try:
            start_time = time.time()
            
            response = requests.post(
                "http://localhost:8000/ingest/v2/batch-optimized/",
                json=test_articles,
                params={"batch_size": batch_size}
            )
            
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Batch submitted successfully in {request_time:.2f}s")
                print(f"   📊 Article count: {result['article_count']}")
                print(f"   📦 Batch size: {result['batch_size']}")
                print(f"   🗃️  Expected DB batches: {result['expected_db_batches']}")
                print(f"   🎯 Task ID: {result['task_id']}")
                print(f"   ⚡ Performance: {result['performance_note']}")
                
                # Wait a bit and check status
                print(f"   ⏳ Waiting for processing...")
                time.sleep(3)
                
            else:
                print(f"❌ Batch submission failed: {response.status_code}")
                print(f"   Error: {response.text}")
                
        except Exception as e:
            print(f"❌ Error testing batch_size={batch_size}: {e}")
    
    # Compare with regular batch processing
    print(f"\n🔄 Testing Regular Batch Processing (for comparison)")
    print("-" * 50)
    
    try:
        start_time = time.time()
        
        response = requests.post(
            "http://localhost:8000/ingest/v2/batch/",
            json=test_articles
        )
        
        request_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Regular batch submitted in {request_time:.2f}s")
            print(f"   📊 Article count: {result['article_count']}")
            print(f"   🎯 Task ID: {result['task_id']}")
            print(f"   🔄 Pipeline: {result['pipeline']}")
            
        else:
            print(f"❌ Regular batch submission failed: {response.status_code}")
            print(f"   Error: {response.text}")
            
    except Exception as e:
        print(f"❌ Error testing regular batch: {e}")
    
    # Check overall status
    print(f"\n📊 Checking Pipeline Status")
    print("-" * 30)
    
    try:
        status_response = requests.get("http://localhost:8000/ingest/status")
        if status_response.status_code == 200:
            status = status_response.json()
            print(f"   Status: {status.get('status', 'unknown')}")
            print(f"   Total articles: {status.get('total_articles', 0)}")
            print(f"   Processed: {status.get('processed_articles', 0)}")
            print(f"   Unprocessed: {status.get('unprocessed_articles', 0)}")
        else:
            print(f"❌ Status check failed: {status_response.status_code}")
    except Exception as e:
        print(f"❌ Error checking status: {e}")
    
    print(f"\n✅ Batch optimization test complete!")
    print("💡 The optimized batch endpoint should provide significant")
    print("   performance improvements for large article volumes.")

if __name__ == "__main__":
    test_optimized_batch() 