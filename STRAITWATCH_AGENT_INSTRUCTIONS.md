# 🛰️ StraitWatch: Cursor Background Agent Instructions

## Overview
StraitWatch is an AI-powered OSINT system for monitoring Taiwan Strait tensions. This document provides complete instructions for background agents to build, deploy, and maintain all system components.

## 🧱 Agent Architecture

### 1. Database Setup Agent (`db_setup_agent`)

**Objective**: Initialize and maintain the Supabase database schema

**Tasks**:
```sql
-- Enhanced Schema for StraitWatch
CREATE TABLE articles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT,
    published_at TIMESTAMPTZ,
    language TEXT DEFAULT 'en',
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE article_tags (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id) ON DELETE CASCADE,
    tag_type TEXT CHECK (tag_type IN ('entity', 'relation', 'event', 'escalation')),
    tag_value TEXT,
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id),
    event_type TEXT,
    event_date TIMESTAMPTZ,
    location TEXT,
    severity_score FLOAT,
    confidence FLOAT,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE forecasts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT,
    forecast_date DATE,
    prediction_horizon INTEGER,
    escalation_score FLOAT,
    confidence_interval JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);
```

**Implementation**:
- File: `storage/enhanced_schema.sql`
- Create indexes on frequently queried columns
- Set up RLS policies for data security

### 2. Article Ingestion Agent (`ingestion_agent`)

**Objective**: Continuously scrape and ingest articles from news sources

**Tasks**:
- Monitor RSS feeds and news APIs
- Extract article content and metadata
- Store in `articles` table
- Handle duplicates and language detection

**Implementation**:
```python
# File: agents/ingestion_agent.py
import asyncio
from datetime import datetime, timedelta
from pipelines.Ingest.NewsIngest import NewsIngest
from storage.db import supabase

class IngestionAgent:
    def __init__(self):
        self.news_ingest = NewsIngest()
    
    async def run_continuous(self):
        while True:
            try:
                # Fetch articles from last 4 hours
                articles = await self.news_ingest.fetch_recent_articles(hours=4)
                
                # Store in database
                for article in articles:
                    self.store_article(article)
                    
                await asyncio.sleep(3600)  # Run every hour
            except Exception as e:
                logger.error(f"Ingestion error: {e}")
                await asyncio.sleep(300)  # Wait 5 min on error
```

**Schedule**: Every hour

### 3. NLP Processing Agent (`nlp_agent`)

**Objective**: Process untagged articles through NLP pipeline

**Tasks**:
- Named Entity Recognition (NER)
- Relation Extraction
- Event Extraction
- Escalation Classification

**Models**:
- NER: `bert-base-cased` fine-tuned
- Relations: `rebel-large`
- Events: Custom BERT classifier
- Escalation: DistilBERT classifier

**Implementation**:
```python
# File: agents/nlp_agent.py
from tagging.enhanced_tagging_layer import EnhancedTagger
from analytics.ner.named_entity_recognizer import NERProcessor
from analytics.ner.event_extraction import EventExtractor

class NLPAgent:
    def __init__(self):
        self.tagger = EnhancedTagger()
        self.ner = NERProcessor()
        self.event_extractor = EventExtractor()
    
    def process_article(self, article):
        # Extract entities
        entities = self.ner.extract_entities(article['content'])
        
        # Extract events
        events = self.event_extractor.extract_events(article['content'])
        
        # Escalation classification
        escalation_score = self.tagger.predict_escalation(article['content'])
        
        return {
            'entities': entities,
            'events': events,
            'escalation_score': escalation_score
        }
```

**Schedule**: Every 2 hours

### 4. Time Series Builder Agent (`timeseries_agent`)

**Objective**: Build daily escalation time series dataset

**Tasks**:
- Query article_tags and events tables
- Aggregate daily metrics
- Generate `escalation_series.csv`

**Implementation**:
```python
# File: agents/timeseries_agent.py
import pandas as pd
from datetime import datetime, timedelta

class TimeSeriesAgent:
    def build_escalation_series(self, days=180):
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=days)
        
        # Query daily metrics
        query = """
        SELECT 
            DATE(created_at) as date,
            COUNT(*) as event_count,
            AVG(CASE 
                WHEN tag_type = 'escalation' THEN confidence 
                ELSE NULL 
            END) as avg_escalation_score
        FROM article_tags 
        WHERE created_at >= %s AND created_at <= %s
        GROUP BY DATE(created_at)
        ORDER BY date
        """
        
        results = supabase.rpc('execute_query', {'query': query, 'params': [start_date, end_date]})
        
        # Save to CSV
        df = pd.DataFrame(results.data)
        df.to_csv('data/time_series/escalation_series.csv', index=False)
```

**Schedule**: Every 6 hours

### 5. Forecasting Agent (`forecasting_agent`)

**Objective**: Train and run forecasting models

**Models**:
1. **Transformer Forecasting**: PyTorch Transformer
2. **ARIMA Baseline**: statsmodels ARIMA

**Implementation**:
```python
# File: agents/forecasting_agent.py
from analytics.time_series.forecasting_engine import ForecastingEngine
from analytics.inference.ensemble_predictor import EnsemblePredictor

class ForecastingAgent:
    def __init__(self):
        self.forecasting_engine = ForecastingEngine()
        self.ensemble = EnsemblePredictor()
    
    def run_forecasting(self):
        # Load time series data
        df = pd.read_csv('data/time_series/escalation_series.csv')
        
        # Train transformer model
        transformer_forecast = self.forecasting_engine.train_transformer(df)
        
        # Train ARIMA model
        arima_forecast = self.forecasting_engine.train_arima(df)
        
        # Ensemble prediction
        final_forecast = self.ensemble.predict([transformer_forecast, arima_forecast])
        
        # Store results
        self.store_forecast(final_forecast)
```

**Schedule**: Daily at 00:00 UTC

### 6. Report Generation Agent (`report_agent`)

**Objective**: Generate daily intelligence reports

**Implementation**:
```python
# File: agents/report_agent.py
from analytics.summarization.unified_intelligence_reporter import IntelligenceReporter

class ReportAgent:
    def __init__(self):
        self.reporter = IntelligenceReporter()
    
    def generate_daily_report(self):
        # Get latest 48 hours of data
        recent_events = self.get_recent_events(hours=48)
        latest_forecast = self.get_latest_forecast()
        
        # Generate report sections
        report = {
            'executive_summary': self.reporter.generate_summary(recent_events),
            'key_developments': self.reporter.analyze_developments(recent_events),
            'forecast': latest_forecast,
            'early_warnings': self.reporter.detect_warnings(recent_events)
        }
        
        # Save report
        timestamp = datetime.now().strftime('%Y-%m-%d')
        filename = f'reports/{timestamp}_straitwatch_report.md'
        self.save_report(report, filename)
```

**Schedule**: Daily at 06:00 UTC

### 7. Model Training Agent (`training_agent`)

**Objective**: Periodically retrain all models

**Tasks**:
- Collect new labeled data
- Fine-tune NLP models
- Retrain forecasting models
- Evaluate performance
- Deploy best models

**Implementation**:
```python
# File: agents/training_agent.py
from models.model_improvement_pipeline import ModelTrainer

class TrainingAgent:
    def __init__(self):
        self.trainer = ModelTrainer()
    
    def retrain_all_models(self):
        # Retrain NER model
        self.trainer.retrain_ner()
        
        # Retrain event extraction
        self.trainer.retrain_event_extraction()
        
        # Retrain escalation classifier
        self.trainer.retrain_escalation_classifier()
        
        # Retrain forecasting models
        self.trainer.retrain_forecasting_models()
        
        # Evaluate and deploy
        self.trainer.evaluate_and_deploy()
```

**Schedule**: Weekly on Sundays

### 8. Orchestration Agent (`orchestrator_agent`)

**Objective**: Coordinate all other agents

**Implementation**:
```python
# File: agents/orchestrator.py
import asyncio
import schedule
from agents.ingestion_agent import IngestionAgent
from agents.nlp_agent import NLPAgent
from agents.timeseries_agent import TimeSeriesAgent
from agents.forecasting_agent import ForecastingAgent
from agents.report_agent import ReportAgent

class OrchestratorAgent:
    def __init__(self):
        self.agents = {
            'ingestion': IngestionAgent(),
            'nlp': NLPAgent(),
            'timeseries': TimeSeriesAgent(),
            'forecasting': ForecastingAgent(),
            'report': ReportAgent()
        }
    
    def setup_schedules(self):
        # Ingestion every hour
        schedule.every().hour.do(self.agents['ingestion'].run)
        
        # NLP every 2 hours
        schedule.every(2).hours.do(self.agents['nlp'].run)
        
        # Time series every 6 hours
        schedule.every(6).hours.do(self.agents['timeseries'].run)
        
        # Forecasting daily
        schedule.every().day.at("00:00").do(self.agents['forecasting'].run)
        
        # Reports daily
        schedule.every().day.at("06:00").do(self.agents['report'].run)
    
    async def run_forever(self):
        while True:
            schedule.run_pending()
            await asyncio.sleep(60)
```

## 🚀 Deployment Instructions

### Step 1: Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Set environment variables
export SUPABASE_URL="your_supabase_url"
export SUPABASE_KEY="your_supabase_key"
```

### Step 2: Initialize Database
```bash
python -c "from agents.db_setup_agent import setup_database; setup_database()"
```

### Step 3: Deploy Agents
```bash
# Start orchestrator (runs all agents)
python agents/orchestrator.py
```

## 📊 Monitoring & Metrics

### Key Performance Indicators
- Articles processed per hour
- Tagging accuracy
- Forecast accuracy (RMSE, MAE)
- Model confidence scores
- System uptime

### Logging
- All agents log to `logs/straitwatch_{date}.log`
- Errors trigger alerts
- Performance metrics saved to database

## 🔧 Maintenance

### Daily
- Check agent health status
- Review error logs
- Validate data quality

### Weekly
- Model performance review
- Database cleanup
- Security updates

### Monthly
- Full system backup
- Performance optimization
- Feature updates

## ✅ Success Criteria

StraitWatch is successfully deployed when:
1. All agents running without errors
2. New articles automatically processed within 2 hours
3. Daily reports generated consistently
4. Forecast accuracy > 75%
5. System uptime > 99%