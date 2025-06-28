#!/usr/bin/env python3
"""
Tagging Module

Provides comprehensive text analysis and tagging capabilities including:
- Named Entity Recognition (NER)
- Relation Extraction
- Event Extraction
- Geographic Tagging
- Sentiment Analysis
- Escalation Prediction
"""

# Core tagging components
from .comprehensive_tagging_pipeline import ComprehensiveTaggingPipeline
from .tagging_pipeline import *

# Enhanced tagging layer with relation and event extraction
try:
    from .enhanced_tagging_layer import (
        EnhancedTaggingLayer,
        process_article_with_enhanced_tagging
    )
    HAS_ENHANCED_TAGGING = True
except ImportError as e:
    print(f"Enhanced tagging layer not available: {e}")
    HAS_ENHANCED_TAGGING = False

__all__ = [
    'ComprehensiveTaggingPipeline',
    'process_article_with_enhanced_tagging',
    'HAS_ENHANCED_TAGGING'
]

if HAS_ENHANCED_TAGGING:
    __all__.extend(['EnhancedTaggingLayer'])

# Import key tagging components for easier access
try:
    from .tag_utils import extract_tags_from_text, needs_human_review, KEYWORD_MAP
except ImportError:
    pass 