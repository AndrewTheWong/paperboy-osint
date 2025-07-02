#!/usr/bin/env python3
"""
Text cleaning service for article preprocessing
"""

import re
import html
from typing import Optional

def clean_html_text(text: str) -> str:
    """
    Clean HTML tags and entities from text
    
    Args:
        text: Raw text that may contain HTML
        
    Returns:
        str: Cleaned text without HTML
    """
    if not text:
        return ""
    
    # Decode HTML entities
    text = html.unescape(text)
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text

def extract_text_from_html(html_content: str) -> str:
    """
    Extract clean text from HTML content
    
    Args:
        html_content: HTML content
        
    Returns:
        str: Extracted clean text
    """
    # Remove script and style tags
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Clean remaining HTML
    text = clean_html_text(text)
    
    return text

def normalize_text(text: str) -> str:
    """
    Normalize text for better processing
    
    Args:
        text: Input text
        
    Returns:
        str: Normalized text
    """
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
    
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip() 