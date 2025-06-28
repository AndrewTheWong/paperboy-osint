# News Ingest Pipeline

Unified news scraping and ingestion system for the Paperboy project.

## Overview

The Ingest module provides a clean, unified interface for scraping news articles from multiple sources with advanced features like paywall bypass, proxy rotation, and intelligent retry mechanisms.

## Structure

```
pipelines/Ingest/
├── __init__.py          # Module exports
├── NewsScraper.py       # Core scraper with advanced features
├── runner.py            # Pipeline orchestration and runners
├── sources.json         # Source configurations
└── README.md           # This file
```

## Key Features

- **Unified Scraper**: Single `NewsScraper` class handles all sources
- **Advanced Paywall Bypass**: Google Cache, Wayback Machine, undetected Chrome
- **Proxy Rotation**: Automatic proxy switching with health checking
- **Cookie Persistence**: Session management for subscription sites
- **Human Behavior Simulation**: Realistic browsing patterns
- **Comprehensive Logging**: Full audit trail and performance metrics
- **Flexible Configuration**: JSON-based source management

## Usage

### Quick Start

```python
from pipelines.Ingest import run_news_ingestion_pipeline

# Run full pipeline
report = await run_news_ingestion_pipeline()

# Run specific categories
report = await run_news_ingestion_pipeline(
    categories=['us_major_outlets'],
    max_workers=5,
    articles_per_source=10
)
```

### Direct Scraper Usage

```python
from pipelines.Ingest import NewsScraper, ScrapingConfig

# Create scraper with custom config
config = ScrapingConfig(
    max_workers=10,
    articles_per_source=20,
    enable_paywall_bypass=True
)

scraper = NewsScraper(config)
articles = await scraper.scrape_all_sources()
```

### Runner Functions

```python
from pipelines.Ingest import (
    run_quick_scraping,
    run_full_production_pipeline,
    run_specific_categories
)

# Quick test (limited articles, no paywall bypass)
report = await run_quick_scraping(['us_major_outlets'])

# Full production pipeline
report = await run_full_production_pipeline()

# Specific categories
report = await run_specific_categories(['international_outlets'])
```

## Configuration

### Source Categories

The system organizes sources into categories:

- `us_major_outlets`: NYT, WaPo, WSJ, Bloomberg, Reuters, CNN, Fox, Politico
- `international_outlets`: BBC, FT, Al Jazeera, SCMP, Guardian, DW, Nikkei
- `chinese_state_media`: Xinhua, Global Times, China Daily
- `taiwan_media`: Taipei Times, Focus Taiwan

### Scraping Configuration

```python
@dataclass
class ScrapingConfig:
    max_workers: int = 10              # Concurrent workers
    max_retries: int = 3               # Retry attempts
    articles_per_source: int = 20      # Articles per source
    enable_paywall_bypass: bool = True # Paywall handling
    use_proxy_rotation: bool = True    # Proxy rotation
    simulate_human_behavior: bool = True # Human-like browsing
```

## Command Line Usage

Use the comprehensive scraper runner:

```bash
# Test mode (quick test)
python run_comprehensive_scraper.py test

# Specific regions
python run_comprehensive_scraper.py us
python run_comprehensive_scraper.py international
python run_comprehensive_scraper.py china
python run_comprehensive_scraper.py taiwan

# Full production pipeline
python run_comprehensive_scraper.py full
```

## Output

### Article Structure

Each scraped article contains:

```json
{
  "url": "https://example.com/article",
  "source": "Source Name",
  "source_url": "https://example.com",
  "title": "Article Title",
  "content": "Article content...",
  "publish_date": "2025-06-22T08:00:00",
  "language": "en",
  "region": "US",
  "category": "major_outlet",
  "scraped_at": "2025-06-22T08:00:00",
  "processed_at": "2025-06-22T08:00:00"
}
```

### Pipeline Report

```json
{
  "pipeline_execution": {
    "start_time": "2025-06-22T08:00:00",
    "end_time": "2025-06-22T08:30:00",
    "duration_seconds": 1800,
    "status": "completed"
  },
  "scraping_results": {
    "total_sources_attempted": 20,
    "successful_sources": 18,
    "failed_sources": 2,
    "total_articles_scraped": 360
  },
  "storage_results": {
    "saved_to_file": true,
    "file_path": "data/scraped_articles_20250622_080000.json",
    "uploaded_to_supabase": true,
    "supabase_new": 340,
    "supabase_updated": 20
  }
}
```

## Advanced Features

### Paywall Bypass

1. **Google Cache**: Attempts to retrieve cached versions
2. **Wayback Machine**: Falls back to archived snapshots
3. **Undetected Chrome**: Uses stealth browser for JavaScript sites
4. **Cookie Persistence**: Maintains sessions for subscription sites

### Proxy Support

- Automatic proxy rotation
- Health checking and performance tracking
- Fallback to direct connections
- Support for HTTP, HTTPS, and SOCKS proxies

### Error Handling

- Exponential backoff retry logic
- Graceful degradation on failures
- Comprehensive error logging
- Proxy failure detection and switching

## Migration from Old System

The old `pipelines.scrapers` module is deprecated. Update imports:

```python
# Old
from pipelines.scrapers import scrape_all_dynamic

# New  
from pipelines.Ingest import run_news_ingestion_pipeline
```

## Dependencies

- `aiohttp`: Async HTTP requests
- `beautifulsoup4`: HTML parsing
- `undetected-chromedriver`: Stealth browser automation
- `selenium`: Web driver support
- `supabase`: Database integration

## Environment Variables

```bash
# Supabase (optional)
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key

# Proxy list (optional)
PROXY_LIST=proxy1.com:8080,proxy2.com:8080

# Proxy API (optional)
PROXY_API_URL=https://api.proxyservice.com/list
``` 