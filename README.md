# Paperboy - Article Processing Pipeline

A production-ready distributed article processing pipeline that scrapes, translates, tags, embeds, and clusters news articles using Celery workers and Redis.

## Architecture

The pipeline consists of 6 main stages:

1. **Ingest Articles** (`services/scraper.py`) - Scrapes articles from configured sources
2. **Clean + Translate** (`services/cleaner.py`, `services/translator.py`) - Cleans and translates content
3. **Tagging NER + Heuristics** (`services/tagger.py`) - Extracts entities and applies tags
4. **Embed Text with SBERT** (`services/embedder.py`) - Creates embeddings using sentence-transformers
5. **Cluster using FAISS/HDBSCAN** (`services/clusterer.py`) - Groups similar articles
6. **Write to Database** (`db/supabase_client.py`) - Stores processed articles and clusters

## Quick Start

### Using Docker Compose (Recommended)

1. **Set environment variables:**
   ```bash
   export SUPABASE_URL="your_supabase_url"
   export SUPABASE_KEY="your_supabase_key"
   ```

2. **Deploy with the deployment script:**
   ```bash
   chmod +x deploy.sh
   ./deploy.sh
   ```

3. **Or start manually:**
   ```bash
   docker-compose up -d
   ```

4. **Access the application:**
   - API: http://localhost:8000
   - API docs: http://localhost:8000/docs
   - Monitoring: http://localhost:5555

### Manual Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   python -m spacy download en_core_web_sm
   ```

2. **Start Redis:**
   ```bash
   redis-server
   ```

3. **Start Celery workers:**
   ```bash
   # Start each worker in separate terminals
   celery -A celery_worker worker --loglevel=info --pool=solo -Q scraper -n Scraper@%h
   celery -A celery_worker worker --loglevel=info --pool=solo -Q translate -n Translator@%h
   celery -A celery_worker worker --loglevel=info --pool=solo -Q tag -n Tagger@%h
   celery -A celery_worker worker --loglevel=info --pool=solo -Q embed -n Embedder@%h
   celery -A celery_worker worker --loglevel=info --pool=solo -Q cluster -n Clusterer@%h
   celery -A celery_worker worker --loglevel=info --pool=solo -Q orchestrate -n Orchestrator@%h
   ```

4. **Start the API:**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

## API Endpoints

- `GET /health` - Health check
- `POST /ingest` - Trigger article ingestion
- `GET /report` - Get processing report

## Configuration

### Environment Variables

- `SUPABASE_URL` - Database project URL
- `SUPABASE_KEY` - Database service key
- `REDIS_URL` - Redis connection URL (default: redis://localhost:6379/0)

### Sources Configuration

Edit `sources/master_sources.json` to configure article sources.

## Project Structure

```
├── app/                    # FastAPI application
│   ├── api/               # API endpoints
│   └── main.py           # FastAPI app
├── services/              # Core pipeline services
│   ├── scraper.py        # Article scraping
│   ├── cleaner.py        # Content cleaning
│   ├── translator.py     # Translation service
│   ├── tagger.py         # NER and tagging
│   ├── embedder.py       # Text embedding
│   └── clusterer.py      # Clustering
├── workers/               # Celery task workers
│   ├── scraper.py        # Scraping tasks
│   ├── translator.py     # Translation tasks
│   ├── tagger.py         # Tagging tasks
│   ├── embedder.py       # Embedding tasks
│   ├── cluster.py        # Clustering tasks
│   └── orchestrator.py   # Pipeline orchestration
├── db/                    # Database clients
│   ├── supabase_client.py # Database client
│   └── redis_queue.py    # Redis queue utilities
├── config/               # Configuration
│   ├── celery_config.py  # Celery settings
│   └── supabase_client.py # Database config
├── models/               # Data models
└── utils/                # Utility functions
```

## Development

### Adding New Sources

1. Add source configuration to `sources/master_sources.json`
2. Update scraping logic in `services/scraper.py` if needed

### Monitoring

- **Monitoring**: http://localhost:5555 (Celery monitoring)
- **API Docs**: http://localhost:8000/docs (FastAPI docs)

## Deployment

### Docker (Recommended)

```bash
# Quick deployment
./deploy.sh

# Manual deployment
docker-compose up -d

# View logs
docker-compose logs -f

# Scale workers
docker-compose up -d --scale scraper-worker=2

# Stop services
docker-compose down
```

### Kubernetes

```bash
# Create namespace
kubectl apply -f k8s/namespace.yaml

# Create secrets (replace with your values)
kubectl create secret generic paperboy-secrets \
  --from-literal=supabase-url=your_supabase_url \
  --from-literal=supabase-key=your_supabase_key \
  -n paperboy

# Deploy services
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/redis.yaml
kubectl apply -f k8s/api.yaml
kubectl apply -f k8s/workers.yaml

# Check status
kubectl get pods -n paperboy
```

### Production Checklist

- [ ] Set environment variables (`SUPABASE_URL`, `SUPABASE_KEY`)
- [ ] Configure sources in `sources/master_sources.json`
- [ ] Set up monitoring (logs)
- [ ] Configure backup strategy for Redis data
- [ ] Set up SSL/TLS for production API
- [ ] Configure resource limits and scaling policies

## License

MIT License 