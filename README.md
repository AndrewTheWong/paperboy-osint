# StraitWatch Backend

A comprehensive intelligence analysis backend system for monitoring Taiwan Strait tensions and regional security developments. This system processes real news articles from verified sources to generate actionable intelligence reports.

## 🎯 Overview

StraitWatch Backend is a production-ready intelligence analysis pipeline that:

- **Ingests real articles** from verified news sources
- **Performs NLP analysis** including tagging, sentiment analysis, and entity extraction
- **Builds time series** for escalation tracking
- **Generates forecasts** using ML models
- **Creates intelligence reports** with threat assessments and recommendations
- **Prevents hallucination** by using only real article data

## 🏗️ Architecture

```
StraitWatch-Backend/
├── agents/                    # Agent system for pipeline orchestration
├── analytics/                 # ML components (clustering, inference, time series)
├── cache/                     # Cache storage
├── config/                    # Configuration files
├── data/                      # Data storage
├── models/                    # Trained ML models
├── pipelines/                 # Pipeline components
├── reports/                   # Generated intelligence reports
├── storage/                   # Database schemas and data processing
├── supabase/                  # Database migrations
├── utils/                     # Core utilities
├── run_complete_straitwatch_pipeline.py  # Main entry point
└── storage_based_reporter.py  # Report generation
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Supabase account and project
- Required API keys (see Configuration section)

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd StraitWatch-Backend
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your Supabase credentials and API keys
   ```

4. **Run the complete pipeline**
   ```bash
   python run_complete_straitwatch_pipeline.py
   ```

## 📋 Pipeline Components

### 1. Article Ingestion Agent
- Scrapes articles from verified news sources
- Handles paywalls and dynamic content
- Performs initial content validation

### 2. NLP Tagging Agent
- Extracts entities, events, and relationships
- Performs sentiment analysis
- Tags articles with escalation indicators

### 3. Time Series Builder Agent
- Builds escalation time series from tagged articles
- Aggregates data by country/region
- Prepares data for forecasting

### 4. Forecasting Agent
- Generates escalation forecasts using XGBoost
- Provides confidence intervals
- Supports multiple forecasting horizons

### 5. Report Generator Agent
- Creates comprehensive intelligence reports
- Includes threat assessments and recommendations
- Generates both summary and detailed reports

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# Supabase Configuration
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key

# API Keys (optional)
NEWS_API_KEY=your_news_api_key
OPENAI_API_KEY=your_openai_api_key

# Database Configuration
DATABASE_URL=your_database_url
```

### Source Configuration

Edit `config/sources_config.json` to configure news sources:

```json
{
  "sources": {
    "scmp": {
      "url": "https://www.scmp.com",
      "enabled": true,
      "priority": "high"
    },
    "defensenews": {
      "url": "https://www.defensenews.com",
      "enabled": true,
      "priority": "high"
    }
  }
}
```

## 📊 Output

### Intelligence Reports

Reports are generated in the `reports/` directory with the following structure:

- **Executive Summary** - High-level threat assessment
- **Key Developments** - Thematic clusters of significant events
- **Threat Assessment** - Military, diplomatic, and economic analysis
- **Regional Analysis** - Geographic focus and actor analysis
- **Recommendations** - Actionable intelligence insights

### Data Storage

- **Articles** - Stored in Supabase with full metadata
- **Tags** - Entity and event tags with confidence scores
- **Time Series** - Escalation metrics by country/region
- **Forecasts** - Predicted escalation levels with confidence intervals

## 🔒 Security & Data Authenticity

- **Real Articles Only** - No synthetic content generation
- **Verified Sources** - All sources are validated and monitored
- **Hallucination Prevention** - Multiple validation layers
- **Data Integrity** - Comprehensive logging and validation

## 🧪 Testing

The pipeline includes comprehensive error handling and graceful degradation:

- **Dependency Checks** - Validates required components
- **Source Validation** - Ensures data quality
- **Model Validation** - Verifies ML model performance
- **Database Health** - Monitors connection status

## 📈 Performance

Typical pipeline execution:
- **Ingestion**: 2-5 minutes for 100+ articles
- **Analysis**: 3-7 minutes for NLP processing
- **Forecasting**: 1-2 minutes for predictions
- **Report Generation**: 30-60 seconds

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the logs in the `reports/` directory
2. Verify your configuration in `.env`
3. Ensure all dependencies are installed
4. Check Supabase connection status

## 🔄 Updates

The system automatically:
- Updates article data from sources
- Retrains models with new data
- Regenerates forecasts
- Updates intelligence reports

---

**StraitWatch Backend** - Real Intelligence from Real Sources 