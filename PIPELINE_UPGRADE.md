# StraitWatch Pipeline Upgrade

## 🚀 Upgraded Pipeline Architecture

The new StraitWatch pipeline has been upgraded from the original queue-based system to a streamlined **Celery chain-based pipeline** with enhanced NER (Named Entity Recognition) capabilities.

### Pipeline Flow

```
[Preprocess] → [NER Tag] → [Embed+Cluster] → [Store to Local Supabase]
```

### Key Improvements

- ✅ **Consolidated Tasks**: All pipeline tasks in one file (`app/tasks/pipeline_tasks.py`)
- ✅ **NER Tagging**: Automatic extraction of geographic and security entities
- ✅ **Chain-based Processing**: Celery chains for sequential task execution
- ✅ **Windows Compatibility**: Uses `--pool=solo` for Windows Celery workers
- ✅ **Direct Storage**: Articles stored to Supabase with tags and entities
- ✅ **Batch Processing**: Support for processing multiple articles at once
- ✅ **Enhanced Logging**: Detailed logging with emojis for easy monitoring

## 📋 Pipeline Tasks

### 1. Preprocess Article (`preprocess_article`)
- Cleans HTML tags and normalizes whitespace
- Removes special characters while preserving punctuation
- Prepares text for NER and embedding

### 2. NER Tagging (`tag_article_ner`)
- **Geographic Entities**: Taiwan, China, South China Sea, Strait of Malacca, etc.
- **Security Entities**: Naval Operations, Cybersecurity, Intelligence, etc.
- **Auto-classification**: Determines region and topic based on content
- **Tag Format**: `GEO:Taiwan`, `SEC:Cybersecurity`

### 3. Embed and Cluster (`embed_and_cluster_article`)
- Generates 384-dimensional embeddings using SentenceTransformers
- Assigns preliminary cluster IDs based on region/topic similarity
- Uses `all-MiniLM-L6-v2` model for embeddings

### 4. Store to Supabase (`store_to_supabase`)
- Stores processed article with all metadata
- Includes tags, entities, embeddings, and cluster assignments
- Provides comprehensive article data for intelligence analysis

## 🛠️ Setup and Usage

### Prerequisites

1. **Redis Server**: Required for Celery task queue
   ```bash
   redis-server
   ```

2. **Supabase Local**: Required for data storage
   ```bash
   supabase start
   ```

3. **Python Dependencies**: Install required packages
   ```bash
   pip install celery redis sentence-transformers hdbscan fastapi uvicorn
   ```

### Quick Start

#### Option 1: Use the Startup Script
```bash
python start_upgraded_pipeline.py
```

#### Option 2: Manual Startup
```bash
# Terminal 1: Start API Server
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2: Start Celery Worker (Windows)
celery -A app.celery_worker worker --loglevel=info --pool=solo
```

## 🌐 API Endpoints

### Legacy Pipeline (v1)
- `POST /ingest/` - Original pipeline

### Upgraded Pipeline (v2)
- `POST /ingest/v2/` - Single article processing
- `POST /ingest/v2/batch/` - Batch article processing
- `GET /ingest/status` - Pipeline status

### Example API Usage

#### Single Article
```bash
curl -X POST "http://localhost:8000/ingest/v2/" \
-H "Content-Type: application/json" \
-d '{
  "title": "China Naval Exercises in South China Sea",
  "body": "The Chinese Navy conducted military exercises...",
  "region": "East Asia",
  "topic": "Maritime Security",
  "source_url": "https://example.com/news"
}'
```

#### Batch Articles
```bash
curl -X POST "http://localhost:8000/ingest/v2/batch/" \
-H "Content-Type: application/json" \
-d '[
  {
    "title": "Article 1",
    "body": "Content 1...",
    "region": "Southeast Asia",
    "topic": "Cybersecurity",
    "source_url": "https://example.com/1"
  },
  {
    "title": "Article 2", 
    "body": "Content 2...",
    "region": "East Asia",
    "topic": "Maritime Security",
    "source_url": "https://example.com/2"
  }
]'
```

## 🧪 Testing

### Test Scripts

1. **API Testing**:
   ```bash
   python test_upgraded_pipeline.py
   ```

2. **Programmatic Usage**:
   ```bash
   python pipeline_main.py
   ```

### Expected Results

After processing, articles will have:
- ✅ Cleaned text
- ✅ Extracted entities (Geographic & Security)
- ✅ Auto-assigned tags (`GEO:China`, `SEC:Naval Operations`)
- ✅ 384-dimensional embeddings
- ✅ Cluster assignments
- ✅ Complete metadata in Supabase

## 📊 Monitoring

### Celery Worker Logs
Monitor task execution with emojis:
```
🧹 Preprocessing article abc-123
🏷️  Tagging article abc-123
🔢 Embedding and clustering article abc-123
💾 Storing article abc-123 to Supabase
✅ Pipeline complete for article abc-123
```

### Supabase Database
Check the `articles` table for:
- `tags` array: Geographic and security tags
- `entities` array: Extracted entity names
- `embedding` array: 384-dimensional vectors
- `cluster_id`: Assigned cluster number
- `region` & `topic`: Auto-classified metadata

## 🔧 Configuration

### Model Configuration
Models are loaded once per worker:
- **SentenceTransformer**: `all-MiniLM-L6-v2`
- **HDBSCAN**: `min_cluster_size=2, min_samples=1`

### Supabase Schema
Ensure your `articles` table includes:
```sql
ALTER TABLE articles ADD COLUMN IF NOT EXISTS tags TEXT[];
ALTER TABLE articles ADD COLUMN IF NOT EXISTS entities TEXT[];
ALTER TABLE articles ADD COLUMN IF NOT EXISTS embedding VECTOR(384);
ALTER TABLE articles ADD COLUMN IF NOT EXISTS cluster_id INTEGER;
```

## 🚨 Troubleshooting

### Common Issues

1. **Celery Windows Errors**: Use `--pool=solo` instead of `--concurrency=1`
2. **Import Errors**: Ensure all tasks are imported in `app/celery_worker.py`
3. **Model Loading**: SentenceTransformers may take time to download on first run
4. **Redis Connection**: Ensure Redis is running on `localhost:6379`
5. **Supabase Connection**: Verify Supabase is accessible on `localhost:54321`

### Task Failures
Tasks have automatic retry with:
- Max retries: 3
- Retry countdown: 60 seconds
- Error logging with full tracebacks

## 📈 Performance

### Throughput
- **Single Article**: ~15-30 seconds (including model loading)
- **Batch Processing**: Parallel task execution
- **Embedding Generation**: ~0.1 seconds per article (after model load)

### Scaling
- Increase Celery workers for higher throughput
- Use Redis clustering for larger deployments
- Consider GPU acceleration for embedding generation

## 🔄 Migration from v1

The upgraded pipeline coexists with the legacy pipeline:
- Legacy endpoints remain functional
- New v2 endpoints use the upgraded pipeline
- Gradual migration recommended for production systems

---

**🎯 Ready to process maritime intelligence with enhanced NER capabilities!** 