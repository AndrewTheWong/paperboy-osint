"""
Analytics Summarization Module

This module provides unified intelligence reporting with comprehensive
analysis of OSINT and GDELT events.
"""

from .unified_intelligence_reporter import generate_intelligence_report, UnifiedIntelligenceReporter

__all__ = [
    'generate_intelligence_report',
    'UnifiedIntelligenceReporter'
] 