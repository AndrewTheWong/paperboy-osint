#!/usr/bin/env python3
"""
Mock data generator for testing the Paperboy pipeline.
Provides functions to generate sample articles for integration testing.
"""
import json
import random
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('mock_data_generator')

# Sample sources for mock data
SOURCES = [
    "taipei_times",
    "south_china_morning_post",
    "japan_times",
    "korea_herald",
    "straits_times"
]

# Sample topics for mock article titles
TOPICS = [
    "Military exercises near {location}",
    "Diplomatic tensions rise between {country1} and {country2}",
    "Naval incident in {location} raises concerns",
    "Trade agreement signed between {country1} and {country2}",
    "Protests erupt in {country1} over {issue}",
    "New technology partnership announced between {country1} and {country2}",
    "Cyber attack targets {country1} government systems",
    "Maritime dispute in {location} escalates",
    "Election results in {country1} shift regional dynamics",
    "Defense budget increased in {country1} amid regional tensions"
]

# Sample locations, countries, and issues for topic templates
LOCATIONS = ["South China Sea", "East China Sea", "Taiwan Strait", "Korean Peninsula", "Malacca Strait"]
COUNTRIES = ["China", "Japan", "South Korea", "Taiwan", "Philippines", "Vietnam", "Malaysia", "Indonesia", "United States"]
ISSUES = ["sovereignty claims", "defense spending", "trade policies", "territorial disputes", "government reforms"]

# Sample content for mock articles
CONTENT_TEMPLATES = [
    "Regional experts are concerned about the recent developments in {location}. "
    "Officials from {country1} have stated that they are monitoring the situation closely. "
    "Meanwhile, {country2} has called for diplomatic talks to address the issue.",
    
    "The incident marks a significant shift in relations between {country1} and {country2}. "
    "Analysts suggest this could impact regional stability, particularly around {location}. "
    "This comes amid growing tensions over {issue} in the region.",
    
    "Sources indicate that {country1} is considering a formal response to recent actions by {country2}. "
    "The situation in {location} has been deteriorating over the past month, with multiple incidents reported. "
    "International observers are urging restraint from all parties involved."
]

# Tags for classification
TAGS = [
    "military", "naval", "aerospace", "missile", "coast guard", "intelligence", "nuclear",
    "diplomacy", "alliances", "treaty", "summit", "sanctions", "arms sale",
    "protest", "riot", "election", "civil society", "governance", "law",
    "trade", "tariff", "investment", "semiconductors", "technology", "infrastructure",
    "cyber", "surveillance", "espionage", "disinformation", "AI", "hack",
    "maritime", "spratly", "fishing fleet", "air defense zone", "incursion",
    "taiwan", "south china sea", "okinawa", "philippines", "japan", "guam",
    "escalatory", "conciliatory", "neutral", "cooperative", "threatening"
]

def generate_mock_title() -> str:
    """Generate a random mock article title"""
    topic = random.choice(TOPICS)
    location = random.choice(LOCATIONS)
    country1, country2 = random.sample(COUNTRIES, 2)
    issue = random.choice(ISSUES)
    
    return topic.format(
        location=location,
        country1=country1,
        country2=country2,
        issue=issue
    )

def generate_mock_content() -> str:
    """Generate random mock article content"""
    template = random.choice(CONTENT_TEMPLATES)
    location = random.choice(LOCATIONS)
    country1, country2 = random.sample(COUNTRIES, 2)
    issue = random.choice(ISSUES)
    
    content = template.format(
        location=location,
        country1=country1,
        country2=country2,
        issue=issue
    )
    
    # Add a few more paragraphs for realistic content length
    paragraphs = [content]
    for _ in range(random.randint(2, 5)):
        additional = random.choice(CONTENT_TEMPLATES).format(
            location=random.choice(LOCATIONS),
            country1=random.choice(COUNTRIES),
            country2=random.choice(COUNTRIES),
            issue=random.choice(ISSUES)
        )
        paragraphs.append(additional)
    
    return "\n\n".join(paragraphs)

def generate_mock_url(source: str, title: str) -> str:
    """Generate a mock URL based on the source and title"""
    # Create a URL-friendly version of the title
    slug = title.lower().replace(" ", "-").replace(",", "").replace(".", "")
    timestamp = datetime.now().strftime("%Y%m%d")
    return f"https://www.{source.replace('_', '')}.com/{timestamp}/{slug}"

def generate_mock_articles(count: int = 10) -> List[Dict[str, Any]]:
    """
    Generate a list of mock articles for testing
    
    Args:
        count: Number of articles to generate
        
    Returns:
        List of article dictionaries
    """
    articles = []
    
    for _ in range(count):
        # Generate basic article information
        source = random.choice(SOURCES)
        title = generate_mock_title()
        text = generate_mock_content()
        url = generate_mock_url(source, title)
        scraped_at = datetime.now().isoformat()
        
        # Create the article dictionary
        article = {
            "title": title,
            "text": text,
            "url": url,
            "source": source,
            "scraped_at": scraped_at
        }
        
        articles.append(article)
    
    logger.info(f"Generated {len(articles)} mock articles")
    return articles

def generate_translated_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add translations to articles (in this mock case, it's the same text)
    
    Args:
        articles: List of articles
        
    Returns:
        List of articles with translations added
    """
    translated_articles = []
    
    for article in articles:
        # Copy the article and add translation fields
        translated = article.copy()
        translated["translated_text"] = article["text"]  # In mock case, same as original
        translated["language"] = "en"  # Assume English
        
        translated_articles.append(translated)
    
    logger.info(f"Added translations to {len(translated_articles)} articles")
    return translated_articles

def generate_tagged_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add tags to articles, with some needing review
    
    Args:
        articles: List of articles
        
    Returns:
        List of articles with tags added
    """
    tagged_articles = []
    
    for article in articles:
        # Copy the article and add tag fields
        tagged = article.copy()
        
        # Randomly assign 1-3 tags
        num_tags = random.randint(1, 3)
        tagged["tags"] = random.sample(TAGS, num_tags)
        
        # For testing the review process, mark some articles as needing review
        if random.random() < 0.3:  # 30% chance
            tagged["needs_review"] = True
            
            # Small chance of unknown tag to trigger review
            if random.random() < 0.2:  # 20% chance
                tagged["tags"] = ["unknown"]
        else:
            tagged["needs_review"] = False
        
        tagged_articles.append(tagged)
    
    logger.info(f"Added tags to {len(tagged_articles)} articles")
    logger.info(f"{sum(1 for a in tagged_articles if a.get('needs_review', False))} articles need review")
    
    return tagged_articles

def generate_embedded_articles(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Add mock embeddings to articles
    
    Args:
        articles: List of articles
        
    Returns:
        List of articles with embeddings added
    """
    embedded_articles = []
    
    for article in articles:
        # Copy the article and add embedding
        embedded = article.copy()
        
        # Generate a random embedding vector (typically 384 dimensions for sentence-transformers)
        embedding_size = 384
        embedded["embedding"] = [random.uniform(-1, 1) for _ in range(embedding_size)]
        
        embedded_articles.append(embedded)
    
    logger.info(f"Added embeddings to {len(embedded_articles)} articles")
    return embedded_articles

def save_mock_data_to_files(count: int = 10) -> None:
    """
    Generate and save mock data for the complete pipeline
    
    Args:
        count: Number of articles to generate
    """
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Generate mock articles
    articles = generate_mock_articles(count)
    
    # Save raw articles
    with open("data/articles.json", "w", encoding="utf-8") as f:
        json.dump(articles, f, ensure_ascii=False, indent=2)
    
    # Generate and save translated articles
    translated = generate_translated_articles(articles)
    with open("data/translated_articles.json", "w", encoding="utf-8") as f:
        json.dump(translated, f, ensure_ascii=False, indent=2)
    
    # Generate and save tagged articles
    tagged = generate_tagged_articles(translated)
    with open("data/tagged_articles.json", "w", encoding="utf-8") as f:
        json.dump(tagged, f, ensure_ascii=False, indent=2)
    
    # Generate and save embedded articles
    embedded = generate_embedded_articles(tagged)
    with open("data/embedded_articles.json", "w", encoding="utf-8") as f:
        json.dump(embedded, f, ensure_ascii=False, indent=2)
    
    logger.info("Mock data generation complete")
    logger.info(f"Generated {count} articles through the complete pipeline")

if __name__ == "__main__":
    import os
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Generate mock data for Paperboy pipeline testing")
    parser.add_argument("--count", "-c", type=int, default=10, help="Number of articles to generate")
    parser.add_argument("--output", "-o", default="data", help="Output directory")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    # Generate and save mock data
    save_mock_data_to_files(args.count)
    
    print(f"Generated {args.count} mock articles in the {args.output} directory")
    print("Files created:")
    print("  - articles.json")
    print("  - translated_articles.json")
    print("  - tagged_articles.json")
    print("  - embedded_articles.json") 