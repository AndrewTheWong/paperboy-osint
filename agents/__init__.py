"""
StraitWatch Background Agents Package

This package contains all the background agents that power the StraitWatch system:
- article_ingest_agent: Article scraping and ingestion  
- tagging_agent: NLP processing and tagging
- timeseries_builder_agent: Time series data construction
- forecasting_agent: Model training and forecasting
- report_generator_agent: Intelligence report generation
- orchestrator_agent: Agent coordination and scheduling
"""

__version__ = "1.0.0"
__author__ = "StraitWatch Team"

from .base_agent import BaseAgent
from .article_ingest_agent import ArticleIngestAgent
from .tagging_agent import TaggingAgent  
from .timeseries_builder_agent import TimeSeriesBuilderAgent
from .forecasting_agent import ForecastingAgent
from .report_generator_agent import ReportGeneratorAgent
from .orchestrator_agent import OrchestratorAgent

__all__ = [
    'BaseAgent',
    'ArticleIngestAgent',
    'TaggingAgent', 
    'TimeSeriesBuilderAgent',
    'ForecastingAgent',
    'ReportGeneratorAgent',
    'OrchestratorAgent'
]