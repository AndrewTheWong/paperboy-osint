#!/usr/bin/env python3
"""
Utility functions for article caching and deduplication.
"""
import json
import hashlib
from pathlib import Path
from typing import Dict, Any, Set

# Default path for the article cache file
CACHE_FILE = Path("storage/article_cache.json")

def generate_hash(article: Dict[str, Any]) -> str:
    """
    Generate a hash of an article for deduplication.
    
    Args:
        article: The article dictionary
        
    Returns:
        A hash string representing the article content
    """
    # Use the URL as a primary key if available
    if "url" in article:
        return hashlib.md5(article["url"].encode("utf-8")).hexdigest()
    
    # Otherwise use title + source as a key
    key_parts = []
    if "title" in article:
        key_parts.append(article["title"])
    if "source" in article:
        key_parts.append(article["source"])
    
    # If we still don't have anything to hash, use the whole article
    if not key_parts:
        key_parts.append(json.dumps(article, sort_keys=True))
    
    # Generate and return the hash
    key_string = "||".join(key_parts)
    return hashlib.md5(key_string.encode("utf-8")).hexdigest()

def load_cache() -> Set[str]:
    """
    Load the article cache from disk.
    
    Returns:
        A set of article hashes
    """
    try:
        if CACHE_FILE.exists():
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return set(json.load(f))
        return set()
    except (json.JSONDecodeError, FileNotFoundError):
        # Return an empty set if the file doesn't exist or is invalid
        return set()

def save_cache(cache: Set[str]) -> None:
    """
    Save the article cache to disk.
    
    Args:
        cache: The set of article hashes to save
    """
    # Create the directory if it doesn't exist
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the cache
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(list(cache), f) 