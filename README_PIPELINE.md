# Paperboy Pipeline

This module implements the full Paperboy backend pipeline for article processing, tagging, and storage.

## Pipeline Flow

The pipeline consists of the following steps:

1. **Scrape**: Collect articles from various news sources
2. **Translate**: Translate non-English articles to English
3. **Tag**: Apply automatic content tagging to articles
4. **Embed**: Generate vector embeddings for similarity search
5. **Store**: Save processed articles to Supabase
6. **Review UI**: Launch Streamlit dashboard for human review if needed

## Setup

Before running the pipeline, make sure to:

1. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Set up the `.env` file with your Supabase credentials:
   ```
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```

3. Install Streamlit for the human review UI:
   ```bash
   pip install streamlit
   ```

## Running the Pipeline

### Production Pipeline

To run the full pipeline with real data:

```bash
python run_pipeline.py
```

This will:
- Scrape articles from configured sources
- Translate non-English articles
- Tag articles based on content
- Generate embeddings
- Upload to Supabase
- Launch Streamlit for human review if any articles need it

### Testing with Mock Data

For testing purposes, you can run the pipeline with mock data:

```bash
python run_pipeline_with_mocks.py --count 10
```

Options:
- `--count` or `-c`: Number of mock articles to generate (default: 10)
- `--use-existing` or `-e`: Use existing mock data files if available

This allows testing the pipeline flow without making real external API calls.

### Running All Tests

To run all the tests for the pipeline:

```bash
python run_all_tests.py
```

This will:
1. Check if Streamlit is installed
2. Run all unit tests
3. Perform a full pipeline test with mock data
4. Provide a summary of test results

## Mock Data Generation

You can also generate mock data separately:

```bash
python pipelines/mock_data_generator.py --count 20
```

This creates:
- `data/articles.json`: Raw scraped articles
- `data/translated_articles.json`: Articles with translations
- `data/tagged_articles.json`: Articles with tags and review flags
- `data/embedded_articles.json`: Articles with embeddings

## Human Review UI

The Streamlit human review UI is launched automatically when articles need review. 
You can also launch it manually:

```bash
streamlit run ui/human_tag_review.py
```

## Folder Structure

- `run_pipeline.py`: Main pipeline script
- `run_pipeline_with_mocks.py`: Testing version with mock data
- `test_run_pipeline.py`: Unit tests for the pipeline
- `run_all_tests.py`: Test runner for all tests
- `pipelines/`: Pipeline modules
  - `dynamic_scraper.py`: Article scraping
  - `translation_pipeline.py`: Translation module
  - `embedding_pipeline.py`: Embedding generation
  - `supabase_storage.py`: Supabase integration
  - `mock_data_generator.py`: Mock data for testing
- `tagging/`: Tagging system
  - `tagging_pipeline.py`: Automatic tagging
  - `tag_utils.py`: Tagging utilities
- `ui/`: User interfaces
  - `human_tag_review.py`: Streamlit review dashboard
- `data/`: Data storage (created when running the pipeline)

## Error Handling

The pipeline is designed to handle errors gracefully:
- Each step is wrapped in try/except blocks
- Errors are logged but don't crash the entire pipeline
- If Streamlit is not available, the error is logged with instructions 