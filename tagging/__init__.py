"""
Tagging package for Paperboy OSINT pipeline.
Contains modules for keyword and ML-based article tagging.
"""

# Import key tagging components for easier access
try:
    from .tagging_pipeline import tag_articles, save_tagged_articles
except ImportError:
    pass

try:
    from .tag_utils import extract_tags_from_text, needs_human_review, KEYWORD_MAP
except ImportError:
    pass 