"""
StraitWatch Background Agents Package

This package contains all the background agents that power the StraitWatch system:
- ingestion_agent: Article scraping and ingestion
- nlp_agent: NLP processing and tagging
- timeseries_agent: Time series data construction
- forecasting_agent: Model training and forecasting
- report_agent: Intelligence report generation
- training_agent: Model retraining and updates
- orchestrator: Agent coordination and scheduling
"""

__version__ = "1.0.0"
__author__ = "StraitWatch Team"

from .ingestion_agent import IngestionAgent
from .nlp_agent import NLPAgent
from .timeseries_agent import TimeSeriesAgent
from .forecasting_agent import ForecastingAgent
from .report_agent import ReportAgent
from .training_agent import TrainingAgent
from .orchestrator import OrchestratorAgent

__all__ = [
    'IngestionAgent',
    'NLPAgent', 
    'TimeSeriesAgent',
    'ForecastingAgent',
    'ReportAgent',
    'TrainingAgent',
    'OrchestratorAgent'
]