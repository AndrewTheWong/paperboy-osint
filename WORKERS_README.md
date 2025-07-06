# Paperboy Celery Workers

A distributed pipeline system using Celery workers for OSINT article processing.

## 🏗️ Architecture

The pipeline consists of 6 dedicated workers:

1. **Scraper** (`scrape` queue) - Web scraping and article extraction
2. **Translator** (`translate` queue) - Language detection and translation
3. **Tagger** (`tag` queue) - NER tagging and entity extraction
4. **Embedder** (`embed` queue) - Text embedding generation
5. **Cluster** (`cluster` queue) - Article clustering and grouping
6. **Orchestrator** (`orchestrate` queue) - Pipeline coordination

## 🚀 Quick Start

### 1. Start Redis
```bash
redis-server
```

### 2. Start All Workers
```powershell
.\start_workers.ps1
```

### 3. Run Pipeline
```python
from workers.orchestrator import run_full_pipeline
result = run_full_pipeline.delay()
print(result.get())
```

## 📋 Worker Details

### Scraper Worker (`workers/scraper.py`)
- **Queue**: `scrape`
- **Tasks**: `run_async_scraper`, `run_continuous_scraper`, `run_stress_test_scraper`
- **Function**: High-speed web scraping with async support
- **Performance**: 2+ articles/second target

### Translator Worker (`workers/translator.py`)
- **Queue**: `translate`
- **Tasks**: `translate_single_article`, `translate_articles_batch`, `translate_from_queue`, `detect_languages_batch`
- **Function**: Language detection and translation to English
- **Supported**: Chinese, Japanese, Korean, English

### Tagger Worker (`workers/tagger.py`)
- **Queue**: `tag`
- **Tasks**: `tag_single_article`, `tag_articles_batch`, `tag_from_queue`, `extract_entities_only`, `tag_with_custom_categories`
- **Function**: NER tagging and entity extraction
- **Features**: Batch processing, custom categories

### Embedder Worker (`workers/embedder.py`)
- **Queue**: `embed`
- **Tasks**: `embed_single_article`, `embed_articles_batch`, `embed_from_queue`, `store_embeddings_to_faiss`, `embed_and_store_batch`, `similarity_search`
- **Function**: Text embedding generation and Faiss storage
- **Features**: TF-IDF fallback, similarity search

### Cluster Worker (`workers/cluster.py`)
- **Queue**: `cluster`
- **Tasks**: `run_clustering`, `cluster_articles_batch`, `cluster_from_queue`, `store_clusters_to_database`, `maybe_trigger_clustering`
- **Function**: Article clustering and similarity grouping
- **Features**: Topic-based clustering, minimum 3 articles per cluster

### Orchestrator Worker (`workers/orchestrator.py`)
- **Queue**: `orchestrate`
- **Tasks**: `run_full_pipeline`, `run_pipeline_from_queue`, `run_continuous_pipeline`, `run_pipeline_step`, `monitor_pipeline_health`
- **Function**: Pipeline coordination and monitoring
- **Features**: Full pipeline orchestration, health monitoring

## 🔧 Configuration

### Celery Configuration (`config/celery_config.py`)
- **Broker**: Redis
- **Result Backend**: Redis
- **Task Routing**: Dedicated queues per worker type
- **Worker Pool**: Solo (Windows compatibility)

### Worker Script (`start_workers.ps1`)
```powershell
# Start all workers
.\start_workers.ps1

# Start specific workers
.\start_workers.ps1 -Workers "scraper,translator"

# Start with custom concurrency
.\start_workers.ps1 -Concurrency 2
```

## 📊 Pipeline Flow

```
Scraping → Translation → Tagging → Embedding → Clustering → Storage
    ↓           ↓           ↓           ↓           ↓           ↓
  scrape    translate     tag        embed      cluster    storage
  queue      queue       queue      queue      queue      queue
```

### Queue Chain
1. **scraping_queue** → Articles from scrapers
2. **translation_queue** → Translated articles
3. **tagging_queue** → Tagged articles
4. **embedding_queue** → Embedded articles
5. **clustering_queue** → Clustered articles
6. **storage_queue** → Articles ready for storage

## 🧪 Testing

### Run Test Suite
```python
python test_workers.py
```

### Test Individual Workers
```python
# Test translation
from workers.translator import translate_single_article
result = translate_single_article.delay(test_article).get()

# Test tagging
from workers.tagger import tag_single_article
result = tag_single_article.delay(article).get()

# Test embedding
from workers.embedder import embed_single_article
result = embed_single_article.delay(article).get()

# Test clustering
from workers.cluster import cluster_articles_batch
result = cluster_articles_batch.delay(articles).get()
```

## 📈 Monitoring

### Celery Flower (Web UI)
```bash
celery -A celery_worker flower
```
Access at: http://localhost:5555

### Health Monitoring
```python
from workers.orchestrator import monitor_pipeline_health
health = monitor_pipeline_health.delay().get()
print(health)
```

## 🔄 Continuous Processing

### Start Continuous Pipeline
```python
from workers.orchestrator import run_continuous_pipeline
result = run_continuous_pipeline.delay(interval_minutes=30)
```

### Queue-based Processing
```python
from workers.orchestrator import run_pipeline_from_queue
result = run_pipeline_from_queue.delay()
```

## 🛠️ Troubleshooting

### Common Issues

1. **Redis Connection Error**
   ```bash
   redis-server
   ```

2. **Worker Not Starting**
   ```bash
   celery -A celery_worker worker --loglevel=info -Q scrape -n Scraper@%h
   ```

3. **Task Not Executing**
   - Check queue routing in `config/celery_config.py`
   - Verify worker is listening to correct queue

4. **Memory Issues**
   - Reduce concurrency: `.\start_workers.ps1 -Concurrency 1`
   - Monitor with: `celery -A celery_worker flower`

### Logs
- Worker logs appear in console
- Use `--loglevel=debug` for detailed logs
- Check Redis for task results

## 📝 API Reference

### Orchestrator Tasks
```python
# Full pipeline
run_full_pipeline(sources=None, max_articles_per_source=10)

# Queue-based pipeline
run_pipeline_from_queue()

# Continuous pipeline
run_continuous_pipeline(interval_minutes=30)

# Single step
run_pipeline_step(step_name, articles=None)

# Health check
monitor_pipeline_health()
```

### Translation Tasks
```python
# Single article
translate_single_article(article)

# Batch articles
translate_articles_batch(articles)

# From queue
translate_from_queue(queue_name="translation_queue")

# Language detection only
detect_languages_batch(articles)
```

### Tagging Tasks
```python
# Single article
tag_single_article(article)

# Batch articles
tag_articles_batch(articles)

# From queue
tag_from_queue(queue_name="tagging_queue")

# Entities only
extract_entities_only(articles)

# Custom categories
tag_with_custom_categories(articles, custom_categories)
```

### Embedding Tasks
```python
# Single article
embed_single_article(article)

# Batch articles
embed_articles_batch(articles)

# From queue
embed_from_queue(queue_name="embedding_queue")

# Store to Faiss
store_embeddings_to_faiss(articles)

# Embed and store
embed_and_store_batch(articles)

# Similarity search
similarity_search(query_text, top_k=10)
```

### Clustering Tasks
```python
# Batch clustering
cluster_articles_batch(articles)

# From queue
cluster_from_queue(queue_name="clustering_queue")

# Store clusters
store_clusters_to_database(articles)

# Trigger clustering
maybe_trigger_clustering()
```

## 🚀 Performance Tips

1. **Concurrency**: Adjust based on CPU cores
2. **Batch Size**: Process articles in batches for efficiency
3. **Queue Monitoring**: Use Flower to monitor queue sizes
4. **Memory**: Monitor memory usage with high concurrency
5. **Redis**: Ensure Redis has enough memory for task results

## 🔐 Security

- Workers run with minimal permissions
- No sensitive data in task results
- Redis should be secured in production
- Use environment variables for API keys

## 📚 Dependencies

- Celery
- Redis
- Supabase
- Faiss
- Sentence Transformers
- spaCy
- Requests

## 🤝 Contributing

1. Add new tasks to appropriate worker files
2. Update task routing in `config/celery_config.py`
3. Add tests in `test_workers.py`
4. Update this README with new features 