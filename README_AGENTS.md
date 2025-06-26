# StraitWatch Background Agents

This directory contains the background agents that power the StraitWatch OSINT system. Each agent is responsible for a specific aspect of the intelligence pipeline.

## 🤖 Agent Overview

### 1. **Ingestion Agent** (`ingestion_agent.py`)
- **Purpose**: Continuously scrapes and ingests articles from news sources
- **Schedule**: Every hour
- **Output**: New articles stored in `articles` table
- **Sources**: RSS feeds, Taiwan-specific news sites, keyword-based searches

### 2. **NLP Agent** (`nlp_agent.py`)
- **Purpose**: Processes untagged articles through NLP pipeline
- **Schedule**: Every 2 hours
- **Components**:
  - Named Entity Recognition (NER)
  - Event Extraction
  - Escalation Classification
  - Basic keyword tagging
- **Output**: Tags and structured data in `article_tags` and `events` tables

### 3. **Time Series Agent** (`timeseries_agent.py`)
- **Purpose**: Builds daily escalation time series dataset
- **Schedule**: Every 6 hours
- **Output**: `data/time_series/escalation_series.csv`

### 4. **Forecasting Agent** (`forecasting_agent.py`)
- **Purpose**: Trains and runs forecasting models
- **Schedule**: Daily at midnight
- **Models**: Transformer and ARIMA forecasting
- **Output**: Forecasts stored in `forecasts` table

### 5. **Report Agent** (`report_agent.py`)
- **Purpose**: Generates daily intelligence reports
- **Schedule**: Daily at 6:00 AM UTC
- **Output**: Intelligence reports in `reports/` directory

### 6. **Training Agent** (`training_agent.py`)
- **Purpose**: Periodically retrains all models
- **Schedule**: Weekly
- **Output**: Updated model files and performance metrics

### 7. **Orchestrator** (`orchestrator.py`)
- **Purpose**: Coordinates all other agents and manages scheduling
- **Features**: Health monitoring, error handling, automated scheduling

## 🚀 Quick Start

### 1. Deploy the System
```bash
python deploy_straitwatch.py
```

### 2. Start the Orchestrator
```bash
# Run with automatic scheduling
python agents/orchestrator.py --mode schedule

# Run all agents once
python agents/orchestrator.py --mode run-once

# Check system status
python agents/orchestrator.py --mode status

# Run specific agent
python agents/orchestrator.py --mode run-once --agent ingestion
```

### 3. Monitor the System
```bash
# Check logs
tail -f logs/straitwatch_$(date +%Y-%m-%d).log

# Check agent status
python agents/orchestrator.py --mode status
```

## 📊 Agent Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  News Sources   │───▶│ Ingestion Agent  │───▶│   Articles DB   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Intelligence   │◀───│    NLP Agent     │◀───│  Untagged Arts  │
│    Reports      │    └──────────────────┘    └─────────────────┘
└─────────────────┘             │                       │
         ▲                      ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Report Agent   │    │ TimeSeries Agent │    │ Tags & Events   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         ▲                      │
         │                      ▼
┌─────────────────┐    ┌──────────────────┐
│Forecasting Agent│◀───│ escalation.csv   │
└─────────────────┘    └──────────────────┘
```

## ⚙️ Configuration

### Environment Variables
```bash
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_key
LOG_LEVEL=INFO
```

### Sources Configuration (`config/sources_config.json`)
```json
{
  "rss_feeds": [
    "https://feeds.reuters.com/reuters/world",
    "https://rss.cnn.com/rss/edition.world.rss"
  ],
  "taiwan_sources": [
    "https://focustaiwan.tw/rss/news.xml"
  ],
  "keywords": [
    "taiwan strait", "china taiwan"
  ]
}
```

## 🔧 Development

### Adding a New Agent

1. **Create agent class** inheriting from `BaseAgent`:
```python
from .base_agent import BaseAgent

class MyAgent(BaseAgent):
    def __init__(self):
        super().__init__("my_agent")
    
    async def run(self) -> Dict[str, Any]:
        # Your agent logic here
        return {"status": "completed"}
```

2. **Add to orchestrator** in `orchestrator.py`:
```python
from .my_agent import MyAgent
self.agents['my_agent'] = MyAgent()
```

3. **Add scheduling** if needed:
```python
schedule.every(4).hours.do(self.run_agent_safe, 'my_agent')
```

### Testing Agents

```bash
# Test individual agent
python -c "
from agents.ingestion_agent import IngestionAgent
import asyncio
async def test():
    agent = IngestionAgent()
    result = await agent.safe_run()
    print(result)
asyncio.run(test())
"

# Test with orchestrator
python agents/orchestrator.py --mode run-once --agent ingestion
```

## 📈 Monitoring

### Health Checks
Each agent implements:
- **Health check**: Database connectivity and basic functionality
- **Status reporting**: Current state, last run time, error count
- **Metrics logging**: Performance and error metrics to database

### Log Files
- Main log: `logs/straitwatch_YYYY-MM-DD.log`
- Agent-specific logs include timestamps, levels, and structured data

### Database Monitoring
Monitor these tables:
- `agent_runs`: Agent execution history and metrics
- `articles`: New article ingestion
- `article_tags`: NLP processing progress
- `events`: Extracted events
- `forecasts`: Forecasting model outputs

## 🚨 Troubleshooting

### Common Issues

**Agent won't start:**
```bash
# Check dependencies
pip install -r requirements.txt

# Check database connection
python -c "from utils.supabase_client import get_supabase; print(get_supabase().table('articles').select('count').limit(1).execute())"
```

**No articles being ingested:**
- Check RSS feed URLs in `config/sources_config.json`
- Verify network connectivity
- Check ingestion agent logs

**NLP processing fails:**
- Ensure ML model files are present
- Check available memory/CPU resources
- Verify model dependencies are installed

**Scheduler stops:**
- Check for unhandled exceptions in logs
- Verify system resources
- Restart orchestrator with: `python agents/orchestrator.py --mode schedule`

### Production Deployment

For production use:
1. Set up systemd service: `sudo cp straitwatch.service /etc/systemd/system/`
2. Enable auto-start: `sudo systemctl enable straitwatch`
3. Start service: `sudo systemctl start straitwatch`
4. Monitor: `sudo systemctl status straitwatch`
5. View logs: `sudo journalctl -u straitwatch -f`

## 🔮 Future Enhancements

- **Agent scaling**: Horizontal scaling for high-volume processing
- **Model versioning**: Automated A/B testing of model updates
- **Alert system**: Real-time notifications for high-priority events
- **API interface**: RESTful API for external integrations
- **Dashboard**: Real-time monitoring dashboard
- **Backup agents**: Redundancy for critical components