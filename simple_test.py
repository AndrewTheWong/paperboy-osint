#!/usr/bin/env python3
"""
Simple test script to check API functionality
"""

import requests
import json

def test_api():
    """Test basic API functionality"""
    
    base_url = "http://localhost:8000"
    
    print("=" * 60)
    print("API FUNCTIONALITY TEST")
    print("=" * 60)
    
    # Test 1: Check if server is running
    try:
        print("🔍 Testing server connectivity...")
        response = requests.get(f"{base_url}/docs")
        print(f"Server status: {response.status_code}")
        if response.status_code == 200:
            print("✅ Server is running!")
        else:
            print("⚠️ Server responded but with unexpected status")
    except requests.exceptions.ConnectionError:
        print("❌ Server is not running or not accessible")
        return
    except Exception as e:
        print(f"❌ Error connecting to server: {e}")
        return
    
    # Test 2: Test ingest endpoint
    try:
        print("\n📤 Testing ingest endpoint...")
        test_article = {
            "title": "Test Article - Taiwan Strait Tensions",
            "content": "Recent developments in the Taiwan Strait have raised concerns about regional stability. Military exercises and diplomatic tensions continue to escalate between major powers in the region.",
            "source": "test_source",
            "url": "https://example.com/test-article",
            "published_at": "2025-07-01T12:00:00Z",
            "language": "en"
        }
        
        response = requests.post(
            f"{base_url}/ingest/",
            json=test_article,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Ingest response status: {response.status_code}")
        print(f"Ingest response: {response.text}")
        
        if response.status_code == 200:
            print("✅ Ingest endpoint is working!")
        else:
            print("❌ Ingest endpoint failed")
            
    except Exception as e:
        print(f"❌ Error testing ingest: {e}")
    
    # Test 3: Test report endpoint
    try:
        print("\n📊 Testing report endpoint...")
        response = requests.get(f"{base_url}/report/today")
        print(f"Report response status: {response.status_code}")
        
        if response.status_code == 200:
            print("✅ Report endpoint is working!")
            try:
                data = response.json()
                print(f"Report data: {json.dumps(data, indent=2)}")
            except:
                print(f"Report text: {response.text}")
        else:
            print("❌ Report endpoint failed")
            
    except Exception as e:
        print(f"❌ Error testing report: {e}")

if __name__ == "__main__":
    test_api() 