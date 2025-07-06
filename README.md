# Paperboy - Intelligent News Pipeline

A sophisticated news scraping, processing, and clustering pipeline built with FastAPI, Celery, Redis, and Supabase.

## 🚀 Features

- **Intelligent Scraping**: Multi-source news scraping with Trifiltura and Newspaper3k
- **AI Processing**: Article cleaning, tagging, and embedding generation
- **Smart Clustering**: HDBSCAN-based article clustering for topic discovery
- **Real-time Dashboard**: Beautiful web interface for monitoring and control
- **Scalable Architecture**: Redis queue with Celery workers for distributed processing

## 🏗️ Architecture

```
┌─────────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│   Frontend      │    │   FastAPI    │    │   Redis      │    │   Supabase   │
│   Dashboard     │◄──►│   Backend    │◄──►│   Queue      │◄──►│   Database   │
└─────────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
                              │                     │
                              ▼                     ▼
                       ┌──────────────┐    ┌──────────────┐
                       │   Celery     │    │   Celery     │
                       │   Workers    │    │   Workers    │
                       │ (Scraper,    │    │ (Preprocess, │
                       │  Preprocess, │    │  Cluster)    │
                       │  Cluster)    │    │              │
                       └──────────────┘    └──────────────┘
```

## 📋 Prerequisites

- Python 3.8+
- Docker (for Redis)
- Supabase CLI (for local development)

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/paperboy.git
cd paperboy
```

### 2. Set Up Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Additional Dependencies
```bash
pip install trafilatura newspaper3k
pip install supabase==1.2.0 gotrue==1.3.1 postgrest==0.10.8 httpx==0.24.1 realtime==1.0.2 websockets==11.0
```

### 5. Start Redis (Docker)
```bash
docker run -d --name redis-server -p 6379:6379 redis:7-alpine
```

### 6. Start Local Supabase
```bash
# Install Supabase CLI if not already installed
npm install -g supabase

# Start local Supabase
supabase start
```

## 🚀 Quick Start

### 1. Start the FastAPI Server
```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 2. Start Celery Workers
```bash
# Terminal 1 - Scraper Worker
celery -A app.celery_worker worker --loglevel=info --pool=solo -n Scraper@%h -Q scrape

# Terminal 2 - Preprocess Worker  
celery -A app.celery_worker worker --loglevel=info --pool=solo -n Preprocess@%h -Q preprocess

# Terminal 3 - Cluster Worker
celery -A app.celery_worker worker --loglevel=info --pool=solo -n Cluster@%h -Q clustering
```

### 3. Access the Dashboard
Open your browser and navigate to: http://localhost:8000

## 🎛️ Usage

### Web Dashboard
The dashboard provides:
- **Real-time Statistics**: Articles scraped, processed, clusters created
- **Control Buttons**: Start scraper, run clustering, generate summaries
- **Live Logs**: Real-time pipeline activity monitoring
- **Status Monitoring**: Pipeline status and queue information

### API Endpoints

#### Scraper
```bash
# Start scraping with default sources
curl -X POST http://localhost:8000/scraper/run \
  -H "Content-Type: application/json" \
  -d '{"use_default_sources": true, "max_articles_per_source": 5}'
```

#### Ingest
```bash
# Ingest a single article
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Article Title",
    "body": "Article content...",
    "source_url": "https://example.com/article"
  }'
```

#### Clustering
```bash
# Run clustering on processed articles
curl -X POST http://localhost:8000/cluster/run
```

#### Status
```bash
# Get pipeline status
curl http://localhost:8000/ingest/status
```

## 🔧 Configuration

### Environment Variables
Create a `.env` file in the root directory:

```env
SUPABASE_URL=http://127.0.0.1:54321
SUPABASE_KEY=your_supabase_anon_key
REDIS_URL=redis://localhost:6379
```

### Scraper Sources
Configure news sources in `app/services/scraper_service.py`:

```python
DEFAULT_SOURCES = [
    "https://www.bbc.com/news",
    "https://www.reuters.com",
    "https://www.theguardian.com"
]
```

## 📊 Pipeline Flow

1. **Scraping**: Celery scraper worker fetches articles from configured sources
2. **Ingestion**: Articles are cleaned and stored in Supabase
3. **Preprocessing**: Articles are tagged, embedded, and queued for clustering
4. **Clustering**: HDBSCAN groups similar articles into clusters
5. **Reporting**: Generate summaries and intelligence reports

## 🐛 Troubleshooting

### Common Issues

#### Redis Connection Error
```bash
# Check if Redis is running
docker ps | grep redis

# Start Redis if not running
docker start redis-server
```

#### Supabase Client Errors
```bash
# Reinstall compatible versions
pip install --force-reinstall supabase==1.2.0 gotrue==1.3.1 postgrest==0.10.8 httpx==0.24.1 realtime==1.0.2 websockets==11.0
```

#### Celery Worker Issues
```bash
# Check worker status
celery -A app.celery_worker inspect active

# Restart workers
taskkill /F /IM python.exe  # Windows
# or
pkill -f celery  # Linux/macOS
```

### Logs
- **FastAPI**: Check terminal running uvicorn
- **Celery Workers**: Check individual worker terminals
- **Redis**: `docker logs redis-server`

## 📁 Project Structure

```
paperboy/
├── app/
│   ├── api/                 # FastAPI endpoints
│   ├── services/            # Core services (scraper, embedding, etc.)
│   ├── tasks/              # Celery tasks
│   ├── utils/              # Utilities and helpers
│   └── static/             # Frontend dashboard
├── storage/                # Database schemas and migrations
├── supabase/              # Supabase configuration
└── logs/                  # Application logs
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Celery](https://celeryproject.org/) for task queue management
- [Supabase](https://supabase.com/) for the database
- [Trafilatura](https://trafilatura.readthedocs.io/) for content extraction
- [HDBSCAN](https://hdbscan.readthedocs.io/) for clustering 