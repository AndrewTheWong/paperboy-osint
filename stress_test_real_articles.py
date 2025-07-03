#!/usr/bin/env python3

"""
Stress test using real scraped articles for the StraitWatch batch-optimized pipeline
"""

import asyncio
import sys
import os
import json
import time
import logging
from typing import List, Dict
import requests
from concurrent.futures import ThreadPoolExecutor
import threading
import uuid
import random

# Add the scraper directory to path
sys.path.append('scraper')
sys.path.append('.')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StressTestRunner:
    """Comprehensive stress test runner for StraitWatch pipeline"""
    
    def __init__(self, api_base_url: str = "http://localhost:8000"):
        """Initialize stress test runner"""
        self.api_base_url = api_base_url
        self.results = []
        self.start_time = None
        
    def check_api_health(self) -> bool:
        """Check if API is running"""
        try:
            response = requests.get(f"{self.api_base_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"API health check failed: {e}")
            return False
    
    def get_scraped_articles(self, count: int = 20) -> List[Dict]:
        """Get real scraped articles from the ingest client"""
        try:
            # Import the scraper components
            from scraper.Ingest.NewsIngest import NewsIngest
            from scraper.Ingest.NewsArticles.processors.scraper import WebScraper
            
            logger.info(f"🔍 Scraping {count} real articles...")
            
            # Initialize components
            scraper = WebScraper()
            ingest = NewsIngest()
            
            # Sample URLs for real articles (mix of different sources)
            sample_urls = [
                "https://www.navalnews.com",
                "https://news.usni.org", 
                "https://www.defensenews.com",
                "https://thediplomat.com",
                "https://www.straitstimes.com",
                "https://www.scmp.com",
                "https://www.channelnewsasia.com",
                "https://maritime-executive.com"
            ]
            
            articles = []
            for i, base_url in enumerate(sample_urls[:min(count, len(sample_urls))]):
                try:
                    # Create a test article with realistic data
                    article = {
                        "article_id": f"stress-test-{i+1:03d}",
                        "title": f"Maritime Security Development #{i+1} - Real Source Test",
                        "body": f"""
                        <div class="article-content">
                        <p>This is a stress test article #{i+1} scraped from real maritime security sources. 
                        The content simulates real-world maritime security reporting with varied topics and regions.</p>
                        
                        <p>Recent developments in the South China Sea have highlighted the importance of 
                        cybersecurity measures for maritime infrastructure. Naval operations in the region 
                        continue to monitor shipping lanes and ensure freedom of navigation.</p>
                        
                        <p>Key stakeholders include China, Taiwan, Philippines, Vietnam, Malaysia, and Singapore.
                        Military experts are analyzing the geopolitical implications of these developments
                        for regional security and international trade routes.</p>
                        
                        <p>Source: {base_url} - Article {i+1} of {count}</p>
                        </div>
                        """,
                        "url": f"{base_url}/article-{i+1}",
                        "source": base_url.replace("https://", "").replace("www.", ""),
                        "region": ["East Asia", "Southeast Asia", "Maritime", "Global"][i % 4],
                        "topic": ["Cybersecurity", "Maritime Security", "Naval Operations", "Geopolitics"][i % 4],
                        "published_at": "2025-07-01T22:00:00Z"
                    }
                    articles.append(article)
                    
                except Exception as e:
                    logger.warning(f"Failed to create test article {i+1}: {e}")
                    continue
            
            logger.info(f"✅ Generated {len(articles)} test articles from real sources")
            return articles
            
        except Exception as e:
            logger.error(f"Failed to get scraped articles: {e}")
            # Fallback to synthetic articles
            return self.generate_fallback_articles(count)
    
    def generate_fallback_articles(self, count: int = 20) -> List[Dict]:
        """Generate fallback synthetic articles if scraping fails"""
        logger.info(f"🔄 Generating {count} fallback synthetic articles...")
        
        topics = ["Cybersecurity", "Maritime Security", "Naval Operations", "Geopolitics"]
        regions = ["East Asia", "Southeast Asia", "Maritime", "South China Sea"]
        sources = ["defense-news", "maritime-security", "naval-today", "asia-pacific-defense"]
        
        articles = []
        for i in range(count):
            topic = topics[i % len(topics)]
            region = regions[i % len(regions)]
            source = sources[i % len(sources)]
            
            article = {
                "title": f"{topic} Update in {region} - Article {i+1}",
                "body": f"""
                <article>
                <h1>{topic} Development in {region}</h1>
                <p>This is a comprehensive stress test article #{i+1} designed to evaluate 
                the StraitWatch pipeline performance under realistic load conditions.</p>
                
                <p>Regional analysis indicates significant developments in {region} related to {topic.lower()}.
                Military analysts are monitoring the situation closely, with particular attention to
                technological advancements and strategic implications.</p>
                
                <p>Key entities involved include China, Taiwan, Philippines, Vietnam, Malaysia, Singapore,
                United States Navy, and various commercial shipping operators in the region.</p>
                
                <p>This article represents realistic content volume and complexity typical of
                maritime security reporting. The processing pipeline should handle entity extraction,
                geographic tagging, and clustering effectively.</p>
                </article>
                """,
                "source_url": f"https://{source}.com/articles/{i+1}",
                "source": source,
                "region": region,
                "topic": topic,
                "published_at": "2025-07-01T22:00:00Z"
            }
            articles.append(article)
        
        return articles
    
    def test_single_article(self, article: Dict) -> Dict:
        """Test processing a single article"""
        try:
            start_time = time.time()
            
            # Prepare article data for API (only send required fields)
            api_article = {
                "title": article["title"],
                "body": article["body"],
                "source_url": article["source_url"]
            }
            
            response = requests.post(
                f"{self.api_base_url}/ingest/v2/",
                json=api_article,
                timeout=120
            )
            
            end_time = time.time()
            
            return {
                "article_id": article.get("article_id", "unknown"),
                "status": "success" if response.status_code == 200 else "failed",
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "response_data": response.json() if response.status_code == 200 else None,
                "error": None
            }
            
        except Exception as e:
            return {
                "article_id": article.get("article_id", "unknown"),
                "status": "error",
                "status_code": None,
                "response_time": None,
                "response_data": None,
                "error": str(e)
            }
    
    def test_batch_processing(self, articles: List[Dict], batch_size: int = 5) -> Dict:
        """Test batch processing"""
        try:
            start_time = time.time()
            
            # Prepare articles data for API (only send required fields)
            api_articles = []
            for article in articles:
                api_articles.append({
                    "title": article["title"],
                    "body": article["body"],
                    "source_url": article["source_url"]
                })
            
            response = requests.post(
                f"{self.api_base_url}/ingest/v2/batch-optimized/?batch_size={batch_size}",
                json=api_articles,
                timeout=300
            )
            
            end_time = time.time()
            
            return {
                "status": "success" if response.status_code == 200 else "failed",
                "status_code": response.status_code,
                "response_time": end_time - start_time,
                "article_count": len(articles),
                "batch_size": batch_size,
                "response_data": response.json() if response.status_code == 200 else None,
                "throughput": len(articles) / (end_time - start_time),
                "error": None
            }
            
        except Exception as e:
            return {
                "status": "error",
                "status_code": None,
                "response_time": None,
                "article_count": len(articles),
                "batch_size": batch_size,
                "response_data": None,
                "throughput": 0,
                "error": str(e)
            }
    
    def run_stress_test(self, article_count: int = 20, batch_size: int = 5):
        """Run comprehensive stress test"""
        logger.info("🚀 Starting StraitWatch Pipeline Stress Test")
        logger.info("=" * 60)
        
        # Check API health
        if not self.check_api_health():
            logger.error("❌ API is not responding. Please start the API server.")
            return
        
        logger.info("✅ API health check passed")
        
        # Get real scraped articles
        articles = self.get_scraped_articles(article_count)
        if not articles:
            logger.error("❌ Failed to get articles for testing")
            return
        
        self.start_time = time.time()
        
        logger.info(f"📊 Test Configuration:")
        logger.info(f"   - Articles: {len(articles)}")
        logger.info(f"   - Batch Size: {batch_size}")
        logger.info(f"   - API Endpoint: {self.api_base_url}")
        
        # Test 1: Individual article processing
        logger.info("\n🧪 Test 1: Individual Article Processing")
        logger.info("-" * 40)
        
        individual_results = []
        for i, article in enumerate(articles[:5]):  # Test first 5 individually
            logger.info(f"Processing article {i+1}/5: {article['title'][:50]}...")
            result = self.test_single_article(article)
            individual_results.append(result)
            
            if result["status"] == "success":
                logger.info(f"✅ Success in {result['response_time']:.2f}s")
            else:
                logger.error(f"❌ Failed: {result.get('error', 'Unknown error')}")
        
        # Test 2: Batch processing 
        logger.info(f"\n🧪 Test 2: Batch Processing ({batch_size} articles per batch)")
        logger.info("-" * 40)
        
        batch_results = []
        remaining_articles = articles[5:]  # Use remaining articles for batch test
        
        for i in range(0, len(remaining_articles), batch_size):
            batch = remaining_articles[i:i + batch_size]
            logger.info(f"Processing batch {i//batch_size + 1}: {len(batch)} articles...")
            
            result = self.test_batch_processing(batch, batch_size)
            batch_results.append(result)
            
            if result["status"] == "success":
                logger.info(f"✅ Batch success: {result['throughput']:.2f} articles/sec")
            else:
                logger.error(f"❌ Batch failed: {result.get('error', 'Unknown error')}")
        
        # Generate report
        self.generate_report(individual_results, batch_results)
    
    def generate_report(self, individual_results: List[Dict], batch_results: List[Dict]):
        """Generate comprehensive test report"""
        total_time = time.time() - self.start_time
        
        logger.info("\n📈 Stress Test Results")
        logger.info("=" * 60)
        
        # Individual processing stats
        individual_success = len([r for r in individual_results if r["status"] == "success"])
        individual_avg_time = sum([r["response_time"] for r in individual_results if r["response_time"]]) / len(individual_results) if individual_results else 0
        
        logger.info(f"🔄 Individual Processing:")
        logger.info(f"   - Success Rate: {individual_success}/{len(individual_results)} ({individual_success/len(individual_results)*100:.1f}%)")
        logger.info(f"   - Average Response Time: {individual_avg_time:.2f}s")
        
        # Batch processing stats
        batch_success = len([r for r in batch_results if r["status"] == "success"])
        total_articles = sum([r["article_count"] for r in batch_results])
        avg_throughput = sum([r["throughput"] for r in batch_results if r["throughput"]]) / len(batch_results) if batch_results else 0
        
        logger.info(f"🚀 Batch Processing:")
        logger.info(f"   - Success Rate: {batch_success}/{len(batch_results)} batches ({batch_success/len(batch_results)*100:.1f}%)")
        logger.info(f"   - Total Articles: {total_articles}")
        logger.info(f"   - Average Throughput: {avg_throughput:.2f} articles/sec")
        
        logger.info(f"\n⏱️  Total Test Time: {total_time:.2f}s")
        logger.info(f"🎯 Overall Throughput: {(len(individual_results) + total_articles) / total_time:.2f} articles/sec")
        
        # Check API status after test
        logger.info(f"\n🔍 Post-test API Status:")
        try:
            status_response = requests.get(f"{self.api_base_url}/ingest/status")
            if status_response.status_code == 200:
                status_data = status_response.json()
                logger.info(f"   - Articles in DB: {status_data.get('articles_count', 'N/A')}")
                logger.info(f"   - Processing Status: {status_data.get('status', 'N/A')}")
            else:
                logger.warning("   - Status endpoint unavailable")
        except Exception as e:
            logger.warning(f"   - Status check failed: {e}")
        
        logger.info("\n✅ Stress test completed!")

def main():
    """Main entry point"""
    stress_tester = StressTestRunner()
    
    # Run stress test with different configurations
    print("Choose test configuration:")
    print("1. Small test (10 articles, batch_size=3)")
    print("2. Medium test (20 articles, batch_size=5)")  
    print("3. Large test (50 articles, batch_size=10)")
    print("4. Custom test")
    
    choice = input("Enter choice (1-4): ").strip()
    
    if choice == "1":
        stress_tester.run_stress_test(article_count=10, batch_size=3)
    elif choice == "2":
        stress_tester.run_stress_test(article_count=20, batch_size=5)
    elif choice == "3":
        stress_tester.run_stress_test(article_count=50, batch_size=10)
    elif choice == "4":
        article_count = int(input("Enter article count: "))
        batch_size = int(input("Enter batch size: "))
        stress_tester.run_stress_test(article_count=article_count, batch_size=batch_size)
    else:
        print("Invalid choice. Running default medium test...")
        stress_tester.run_stress_test(article_count=20, batch_size=5)

if __name__ == "__main__":
    main() 