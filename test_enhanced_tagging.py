#!/usr/bin/env python3
"""
Test Enhanced Tagging System
Verify the new OSINT tagging system with gazetteer, patterns, and NER
"""

import sys
import os
import uuid

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from app.services.tagger import tag_article, tag_article_batch
from app.services.escalation_score import get_priority_level, get_escalation_emoji

def test_enhanced_tagging():
    """Test the enhanced tagging system"""
    
    print("🧪 Testing Enhanced OSINT Tagging System")
    
    # Test article
    test_article = {
        "title": "PLA Navy Conducts Live Fire Drill Near Spratly Islands",
        "body": """
        The People's Liberation Army Navy conducted a live fire drill near the Spratly Islands 
        in the South China Sea. The exercise involved Type 055 destroyers and J-20 fighter jets. 
        Xi Jinping emphasized the importance of maritime rights protection.
        """
    }
    
    # Test tagging
    result = tag_article(test_article['body'], test_article['title'])
    
    print(f"📊 Confidence Score: {result['confidence_score']:.3f}")
    print(f"🎯 Priority Level: {result['priority_level']}")
    print(f"🏷️ Total Tags: {len(result['tags'])}")
    
    # Show tag categories
    print("\n📋 Tag Categories:")
    for category, values in result['tag_categories'].items():
        if values:
            print(f"  {category}: {', '.join(values[:3])}")
    
    print("\n✅ Enhanced tagging system test completed!")

if __name__ == "__main__":
    test_enhanced_tagging() 