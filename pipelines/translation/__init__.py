"""
Translation package for the comprehensive pipeline.
Contains all translation functionality using NLLB models.
"""

from .translation_pipeline import translate_articles, ArticleTranslator

__all__ = ['translate_articles', 'ArticleTranslator'] 