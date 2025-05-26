"""
Pipelines package for Paperboy OSINT data processing.
Contains modules for scraping, translation, tagging, embedding, and storage.
"""

# Import key pipeline components for easier access
try:
    from .embedding_pipeline import embed_articles, save_embedded_articles
except ImportError:
    pass

try:
    from .translation_pipeline import translate_articles
except ImportError:
    pass

try:
    from .supabase_storage import upload_articles_to_supabase
except ImportError:
    pass 