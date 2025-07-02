#!/usr/bin/env python3
"""
Ingest client for scrapers to send articles to StraitWatch backend
"""

import requests
import json
import logging
from typing import Dict, Optional
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IngestClient:
    """Client for sending articles to StraitWatch backend"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        """
        Initialize ingest client
        
        Args:
            base_url: Base URL of the StraitWatch backend
        """
        self.base_url = base_url.rstrip('/')
        self.ingest_url = f"{self.base_url}/ingest"
        self.status_url = f"{self.base_url}/ingest/status"
        
    def send_article(self, title: str, body: str, source_url: str, 
                    region: Optional[str] = None, topic: Optional[str] = None,
                    article_id: Optional[str] = None) -> Dict:
        """
        Send a single article to the backend
        
        Args:
            title: Article title
            body: Article body/content
            source_url: Source URL
            region: Geographic region (optional)
            topic: Article topic (optional)
            article_id: Custom article ID (optional)
            
        Returns:
            Dict: Response from backend
        """
        try:
            payload = {
                "title": title,
                "body": body,
                "source_url": source_url,
                "region": region,
                "topic": topic
            }
            
            if article_id:
                payload["id"] = article_id
            
            logger.info(f"📤 Sending article: {title[:50]}...")
            
            response = requests.post(
                self.ingest_url,
                json=payload,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Article sent successfully: {result.get('id', 'unknown')}")
                return result
            else:
                logger.error(f"❌ Failed to send article: {response.status_code} - {response.text}")
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.RequestException as e:
            logger.error(f"❌ Network error sending article: {e}")
            return {
                "status": "error",
                "error": f"Network error: {str(e)}"
            }
        except Exception as e:
            logger.error(f"❌ Error sending article: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    def send_batch(self, articles: list) -> Dict:
        """
        Send multiple articles in batch
        
        Args:
            articles: List of article dictionaries
            
        Returns:
            Dict: Summary of batch operation
        """
        results = []
        success_count = 0
        error_count = 0
        
        logger.info(f"📤 Sending batch of {len(articles)} articles")
        
        for i, article in enumerate(articles):
            try:
                result = self.send_article(**article)
                results.append(result)
                
                if result.get('status') == 'queued':
                    success_count += 1
                else:
                    error_count += 1
                    
            except Exception as e:
                logger.error(f"❌ Error sending article {i}: {e}")
                results.append({
                    "status": "error",
                    "error": str(e)
                })
                error_count += 1
        
        logger.info(f"📊 Batch complete: {success_count} successful, {error_count} failed")
        
        return {
            "status": "complete",
            "total": len(articles),
            "successful": success_count,
            "failed": error_count,
            "results": results
        }
    
    def get_status(self) -> Dict:
        """
        Get backend ingestion status
        
        Returns:
            Dict: Status information
        """
        try:
            response = requests.get(self.status_url, timeout=10)
            
            if response.status_code == 200:
                return response.json()
            else:
                return {
                    "status": "error",
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            logger.error(f"❌ Error getting status: {e}")
            return {
                "status": "error",
                "error": str(e)
            }

# Global client instance
_default_client = None

def get_client(base_url: str = "http://localhost:8000") -> IngestClient:
    """Get or create default ingest client"""
    global _default_client
    if _default_client is None:
        _default_client = IngestClient(base_url)
    return _default_client

def send_article(title: str, body: str, source_url: str, 
                region: Optional[str] = None, topic: Optional[str] = None,
                article_id: Optional[str] = None) -> Dict:
    """Convenience function to send a single article"""
    client = get_client()
    return client.send_article(title, body, source_url, region, topic, article_id)

def send_batch(articles: list) -> Dict:
    """Convenience function to send multiple articles"""
    client = get_client()
    return client.send_batch(articles) 