# Paperboy OSINT Data Pipeline

Paperboy is an OSINT (Open Source Intelligence) data pipeline that scrapes, translates, tags, and stores articles from various news sources focusing on geopolitical events in East Asia.

## Architecture

The pipeline consists of four main stages:

1. **Scraping**: Collect articles from various news sources
2. **Translation**: Translate non-English articles to English
3. **Auto-tagging**: Add tags and identify articles needing human review
4. **Storage**: Store articles in Supabase for search and analysis

## Directory Structure

```
Paperboy/
├── .streamlit/              # Streamlit configuration
├── config/                  # Configuration files
├── dashboard/               # Dashboard visualizations
├── data/                    # Data storage
│   ├── model_ready/         # Processed data for models
│   └── ucdp/                # UCDP conflict data
├── docs/                    # Documentation
├── ingestion/               # Data ingestion scripts
├── logs/                    # Log files
├── models/                  # ML models
│   └── training_scripts/    # Model training scripts
├── notebooks/               # Jupyter notebooks
├── pipelines/               # Pipeline components
│   ├── config/              # Pipeline configuration
│   ├── data/                # Pipeline data
│   ├── scrapers/            # Old scraper location (deprecated)
│   └── utils/               # Pipeline utilities
├── scraping/                # Scraping architecture
│   ├── config/              # Scraper configuration
│   └── sources/             # Source-specific scrapers
├── storage/                 # Storage utilities
├── tagging/                 # Article tagging
├── tests/                   # Test suite
├── ui/                      # User interfaces
└── utils/                   # Utility functions
```

## Running the Pipeline

To run the complete pipeline:

```bash
python run_pipeline.py
```

The pipeline will:

1. Scrape articles from configured sources
2. Translate non-English articles to English
3. Add tags and identify articles needing human review
4. Upload articles to Supabase with deduplication

## Scraper Architecture

The scraping system uses a flexible architecture:

1. Source-specific scrapers in `scraping/sources/`
2. Universal fallback scraper in `scraping/universal_scraper.py`
3. Runner that manages scraping in `scraping/runner.py`

Configuration is stored in `scraping/config/sources_config.json`.

Each article must include:
- `title`: Article title
- `url`: Source URL
- `source`: Source name
- `scraped_at`: Timestamp
- `language`: Language code

## Human Review UI

A Streamlit-based review interface is available for articles flagged as needing human review:

```bash
streamlit run ui/human_tag_review.py
```

## Test Suite

Run the test suite with:

```bash
python run_all_tests.py
```

Tests are organized in the `/tests` directory and use Python's unittest framework.

## Dependencies

- Python 3.8+
- BeautifulSoup4 (for scraping)
- Streamlit (for UI)
- Supabase Python client (for storage)

See `requirements.txt` for a full list of dependencies.

## Configuration

Configure the pipeline using:

1. `.env` file for secrets (see `template.env`)
2. `config/sources_config.json` for scraper configuration
3. `config/tagging_config.json` for tagging rules

## License

Copyright (c) 2025 Paperboy Project 