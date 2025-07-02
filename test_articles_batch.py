#!/usr/bin/env python3
"""
Send multiple test articles through the new pipeline
"""

import requests
import time
import json

def send_test_articles():
    """Send 3 test articles to test clustering"""
    
    test_articles = [
        {
            "source_url": "https://example.com/maritime-security-1",
            "title": "South China Sea Naval Exercises",
            "body": "Naval forces conducted joint maritime security exercises in the South China Sea region. The exercises focused on improving coordination and response capabilities.",
            "region": "Southeast Asia",
            "topic": "Maritime Security"
        },
        {
            "source_url": "https://example.com/maritime-security-2", 
            "title": "Strait of Malacca Shipping Lane Protection",
            "body": "Increased security measures are being implemented in the Strait of Malacca shipping lanes to protect commercial vessels from potential threats.",
            "region": "Southeast Asia",
            "topic": "Maritime Security"
        },
        {
            "source_url": "https://example.com/cyber-threat-analysis",
            "title": "Regional Cybersecurity Assessment",
            "body": "Cybersecurity experts analyzed emerging threats targeting critical infrastructure across the Asia-Pacific region. New defensive strategies are being developed.",
            "region": "Asia Pacific",
            "topic": "Cybersecurity"
        }
    ]
    
    print("🚀 Sending test articles through new pipeline...")
    print("=" * 50)
    
    for i, article in enumerate(test_articles, 1):
        print(f"\n📝 Sending Article {i}: {article['title']}")
        
        try:
            response = requests.post(
                "http://localhost:8000/ingest",
                headers={"Content-Type": "application/json"},
                data=json.dumps(article)
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Success: {result['id']} - {result['status']}")
            else:
                print(f"❌ Failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Error sending article {i}: {e}")
        
        time.sleep(1)  # Small delay between requests
    
    print(f"\n🎉 Finished sending {len(test_articles)} test articles!")
    print("💡 Now run clustering to test the complete new flow...")

if __name__ == "__main__":
    send_test_articles() 