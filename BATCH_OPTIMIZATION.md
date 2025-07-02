# Batch Optimization for StraitWatch Pipeline

## Overview

The StraitWatch pipeline has been optimized with batch processing capabilities to dramatically improve Supabase storage performance. Instead of storing articles individually (which was very slow), articles are now processed and stored in configurable batches.

## Performance Improvements

### Before (Individual Storage)
- Each article stored separately via individual Celery tasks
- High database connection overhead
- Slow throughput: ~1-2 articles/second
- Many individual HTTP requests to Supabase

### After (Batch Storage)
- Articles processed in batches then stored via single Supabase call
- Minimal database connection overhead
- Fast throughput: ~10-50 articles/second (depending on batch size)
- Single HTTP request per batch to Supabase

## New API Endpoints

### 1. Optimized Batch Processing
**Endpoint:** `POST /ingest/v2/batch-optimized/`

**Parameters:**
- `articles`: List of article objects
- `batch_size`: Number of articles to store per Supabase batch (default: 10)

**Example:**
```bash
curl -X POST "http://localhost:8000/ingest/v2/batch-optimized/?batch_size=5" \
  -H "Content-Type: application/json" \
  -d '[
    {
      "title": "Test Article",
      "body": "Article content...",
      "region": "East Asia",
      "topic": "Cybersecurity", 
      "source_url": "https://example.com/article"
    }
  ]'
```

**Response:**
```json
{
  "status": "optimized_batch_started",
  "article_count": 5,
  "batch_size": 5,
  "expected_db_batches": 1,
  "task_id": "abc123...",
  "articles": [...],
  "pipeline": "batch_process → batch_tag → batch_embed_cluster → batch_store_supabase",
  "performance_note": "Articles will be stored to Supabase in batches of 5 for optimal performance"
}
```

### 2. Regular Batch Processing (Parallel)
**Endpoint:** `POST /ingest/v2/batch/`

Processes articles in parallel but stores individually (existing behavior).

## New Celery Tasks

### `store_batch_to_supabase(articles)`
Stores multiple articles to Supabase in a single database operation.

**Benefits:**
- Single database transaction
- Reduced connection overhead
- Better error handling
- Faster execution

### `process_article_batch(articles, batch_size=10)`
Processes multiple articles through the complete pipeline and stores in optimized batches.

**Pipeline Steps:**
1. **Preprocess** - HTML cleaning, text normalization
2. **NER Tagging** - Geographic and security entity extraction
3. **Embedding & Clustering** - Generate embeddings and assign clusters
4. **Batch Storage** - Store to Supabase in configurable batches

## Batch Size Recommendations

| Articles | Recommended Batch Size | Expected Performance |
|----------|----------------------|---------------------|
| 1-10     | 5-10                | ~5-10 articles/sec  |
| 11-50    | 10-15               | ~10-20 articles/sec |
| 51-100   | 15-20               | ~15-30 articles/sec |
| 100+     | 20-25               | ~20-50 articles/sec |

## Usage Examples

### Test Script
```bash
# Test optimized batch processing
python test_batch_optimized.py
```

### Python Client
```python
import requests

articles = [
    {
        "title": "Article 1",
        "body": "Content...",
        "region": "East Asia",
        "topic": "Maritime Security",
        "source_url": "https://example.com/1"
    },
    # ... more articles
]

# Optimized batch with batch_size=10
response = requests.post(
    "http://localhost:8000/ingest/v2/batch-optimized/",
    json=articles,
    params={"batch_size": 10}
)
```

## Performance Monitoring

Check pipeline status:
```bash
curl http://localhost:8000/ingest/status
```

Monitor Celery worker logs for batch processing metrics:
```
💾 Batch storing 10 articles to Supabase
✅ Batch stored 10 articles to Supabase with DB IDs: [123, 124, 125...]
✅ Batch processing complete: 10 articles stored in 1 batches
```

## Configuration

### Celery Worker
Ensure batch tasks are registered:
```python
from app.tasks.pipeline_tasks import (
    store_batch_to_supabase, 
    process_article_batch
)
```

### Batch Size Tuning
- **Small batches (2-5):** Lower memory usage, more database calls
- **Medium batches (10-15):** Balanced performance and reliability
- **Large batches (20+):** Best performance but higher memory usage

Choose based on your system resources and performance requirements.

## Error Handling

The batch processing includes robust error handling:
- Individual article processing errors don't stop the entire batch
- Failed batches are retried with exponential backoff
- Detailed logging for debugging batch operations
- Graceful degradation if batch storage fails

## Migration from Individual Processing

To migrate existing code:

1. **Replace individual calls:**
   ```python
   # Old: Individual processing
   for article in articles:
       run_article_pipeline.delay(article)
   
   # New: Batch processing
   process_article_batch.delay(articles, batch_size=10)
   ```

2. **Update API calls:**
   ```python
   # Old: Individual submissions
   for article in articles:
       requests.post("/ingest/v2/", json=article)
   
   # New: Batch submission
   requests.post("/ingest/v2/batch-optimized/", 
                 json=articles, 
                 params={"batch_size": 10})
   ```

## Monitoring and Troubleshooting

### Check Batch Performance
Monitor logs for batch timing:
```
🔄 Processing batch of 10 articles
✅ Batch processing complete: 10 articles stored
```

### Tune Batch Size
If experiencing:
- **Memory issues:** Reduce batch_size
- **Slow performance:** Increase batch_size
- **Database timeouts:** Reduce batch_size

### Debug Failed Batches
Check Celery worker logs for detailed error information about batch processing failures. 