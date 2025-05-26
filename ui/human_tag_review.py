#!/usr/bin/env python3
"""
Streamlit UI for human review of auto-tagged articles.
"""
import json
import streamlit as st
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import sys
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('human_tag_review')

# Add project root to the path to ensure imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import tag utils to access the tag categories
from tagging.tag_utils import KEYWORD_MAP

# Tagging system - use the same categories as in tag_utils.py
ALL_TAGS = {
    "Military & Security": ["military", "naval", "aerospace", "missile", "coast guard", "intelligence", "nuclear"],
    "Geopolitics & Diplomacy": ["diplomacy", "alliances", "treaty", "summit", "sanctions", "arms sale"],
    "Domestic Affairs": ["protest", "riot", "election", "civil society", "governance", "law"],
    "Economy & Industry": ["trade", "tariff", "investment", "semiconductors", "technology", "infrastructure"],
    "Cyber & Espionage": ["cyber", "surveillance", "espionage", "disinformation", "AI", "hack"],
    "Gray Zone & Maritime": ["maritime", "spratly", "fishing fleet", "air defense zone", "incursion"],
    "Regional Focus": ["taiwan", "south china sea", "okinawa", "philippines", "japan", "guam"],
    "Tone/Signal": ["escalatory", "conciliatory", "neutral", "cooperative", "threatening"]
}

# Flatten the tags for UI presentation
FLAT_TAGS = [f"{category} – {tag}" for category, tags in ALL_TAGS.items() for tag in tags]

# Define file paths using OS-independent path joining
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
TAGGED_ARTICLES_FILE = os.path.join(DATA_DIR, "tagged_articles.json")
REVIEWED_ARTICLES_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "tagging", "reviewed_articles.json")

def load_articles() -> List[Dict[str, Any]]:
    """
    Load articles from the tagged_articles.json file.
    """
    if not os.path.exists(TAGGED_ARTICLES_FILE):
        logger.error(f"Tagged articles file not found: {TAGGED_ARTICLES_FILE}")
        return []
    
    try:
        with open(TAGGED_ARTICLES_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
            logger.info(f"Loaded {len(articles)} articles from {TAGGED_ARTICLES_FILE}")
            return articles
    except json.JSONDecodeError:
        logger.error(f"Error parsing {TAGGED_ARTICLES_FILE}. Please ensure it contains valid JSON.")
        st.error(f"Error parsing {TAGGED_ARTICLES_FILE}. Please ensure it contains valid JSON.")
        return []

def load_reviewed_articles() -> List[Dict[str, Any]]:
    """
    Load previously reviewed articles.
    """
    if not os.path.exists(REVIEWED_ARTICLES_FILE):
        logger.info(f"No reviewed articles file found at {REVIEWED_ARTICLES_FILE}. Creating a new one.")
        return []
    
    try:
        with open(REVIEWED_ARTICLES_FILE, "r", encoding="utf-8") as f:
            articles = json.load(f)
            logger.info(f"Loaded {len(articles)} previously reviewed articles")
            return articles
    except json.JSONDecodeError:
        # If file exists but is not valid JSON, return empty list
        logger.error(f"Error parsing reviewed articles file. Starting with empty list.")
        return []

def save_reviewed_article(article: Dict[str, Any], reviewed_articles: List[Dict[str, Any]]) -> None:
    """
    Save a reviewed article to the reviewed_articles.json file.
    """
    # Check if article already exists in reviewed articles
    article_exists = False
    for i, existing_article in enumerate(reviewed_articles):
        if existing_article.get("url") == article.get("url"):
            # Update existing article
            reviewed_articles[i] = article
            article_exists = True
            break
    
    # If article doesn't exist, append it
    if not article_exists:
        reviewed_articles.append(article)
    
    # Save all reviewed articles
    os.makedirs(os.path.dirname(REVIEWED_ARTICLES_FILE), exist_ok=True)
    with open(REVIEWED_ARTICLES_FILE, "w", encoding="utf-8") as f:
        json.dump(reviewed_articles, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved reviewed article: {article.get('title', 'Unknown')} by {article.get('reviewed_by', 'Unknown')}")

def get_articles_needing_review(articles: List[Dict[str, Any]], reviewed_articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter articles that need review.
    """
    # Get URLs of reviewed articles
    reviewed_urls = {article.get("url") for article in reviewed_articles if not article.get("needs_review", False)}
    
    # Filter articles that need review - improved filter
    articles_to_review = [
        article for article in articles 
        if article.get("url") not in reviewed_urls and 
        (str(article.get("needs_review")).lower() in ["true", "1", "yes"] or article.get("tags") == ["unknown"])
    ]
    
    logger.info(f"Found {len(articles_to_review)} articles that need review out of {len(articles)} total articles")
    return articles_to_review

def display_article(article: Dict[str, Any]) -> None:
    """
    Display article information.
    """
    st.markdown(f"## {article.get('title', 'No Title')}")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Source:** {article.get('source', 'Unknown')}")
        st.markdown(f"**URL:** [{article.get('url', '#')}]({article.get('url', '#')})")
        st.markdown(f"**Scraped at:** {article.get('scraped_at', 'Unknown')}")
    
    with col2:
        st.markdown("**Original tags:**")
        if article.get("tags"):
            st.write(", ".join(article.get("tags", [])))
        else:
            st.write("No tags")
    
    st.markdown("**Content:**")
    if "translated_text" in article and article["translated_text"]:
        st.markdown(article["translated_text"])
    else:
        st.markdown(article.get("title", "No content available"))

def tag_to_flat(tag: str) -> Optional[str]:
    """Convert a raw tag to its flat representation"""
    for category, tags in ALL_TAGS.items():
        if tag in tags:
            return f"{category} – {tag}"
    return None

def flat_to_tag(flat_tag: str) -> str:
    """Extract the raw tag from a flat tag representation"""
    return flat_tag.split(" – ")[1] if " – " in flat_tag else flat_tag

def main():
    """
    Main Streamlit application.
    """
    st.set_page_config(
        page_title="OSINT Article Review",
        page_icon="📰",
        layout="wide"
    )
    
    st.title("OSINT Article Review")
    
    # Sidebar for analyst information
    with st.sidebar:
        st.header("Analyst Information")
        analyst_name = st.text_input("Analyst Name (required)")
        st.markdown("---")
        st.markdown("### Instructions")
        st.markdown("1. Select an article to review")
        st.markdown("2. Review the content and existing tags")
        st.markdown("3. Edit tags as needed")
        st.markdown("4. Submit your review")
    
    # Load articles and reviewed articles
    articles = load_articles()
    reviewed_articles = load_reviewed_articles()
    
    # Filter articles that need review
    articles_to_review = get_articles_needing_review(articles, reviewed_articles)
    
    # Display stats
    st.info(f"📊 Stats: {len(articles)} total articles, {len(articles_to_review)} need review, {len(reviewed_articles)} previously reviewed")
    
    if not articles_to_review:
        st.success("🎉 All articles are tagged! There are no articles that need review.")
        return
    
    # Article selection
    article_titles = [article.get("title", f"No Title (URL: {article.get('url', 'unknown')})") for article in articles_to_review]
    selected_index = st.selectbox("Select an article to review:", range(len(article_titles)), format_func=lambda i: article_titles[i])
    
    # Get the selected article
    selected_article = articles_to_review[selected_index]
    
    # Display article information
    display_article(selected_article)
    
    # Tag selection
    st.markdown("---")
    st.markdown("### Tag Selection")
    
    # Convert existing tags to flat tags for default selection
    default_flat_tags = [tag_to_flat(tag) for tag in selected_article.get("tags", []) if tag_to_flat(tag)]
    
    # Multi-select for tags
    selected_flat_tags = st.multiselect(
        "Select tags for this article:",
        options=FLAT_TAGS,
        default=default_flat_tags
    )
    
    # Convert selected flat tags back to raw tags
    selected_tags = [flat_to_tag(flat_tag) for flat_tag in selected_flat_tags]
    
    # Submit button
    st.markdown("---")
    if st.button("Submit Review", disabled=not (analyst_name and selected_tags)):
        if not analyst_name:
            st.warning("Please enter your name before submitting.")
        elif not selected_tags:
            st.warning("Please select at least one tag before submitting.")
        else:
            # Create a copy of the article with updated fields
            reviewed_article = selected_article.copy()
            reviewed_article["human_tags"] = selected_tags
            reviewed_article["reviewed_by"] = analyst_name
            reviewed_article["needs_review"] = False
            reviewed_article["reviewed_at"] = datetime.now().isoformat()
            
            # Save the reviewed article
            save_reviewed_article(reviewed_article, reviewed_articles)
            
            st.success(f"Article reviewed successfully by {analyst_name}!")
            st.balloons()
            
            # Refresh the page after 2 seconds
            st.markdown(
                """
                <meta http-equiv="refresh" content="2">
                """,
                unsafe_allow_html=True
            )

if __name__ == "__main__":
    main() 