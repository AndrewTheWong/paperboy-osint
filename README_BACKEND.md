# StraitWatch Phase 1 Backend ML Pipeline

A comprehensive machine learning pipeline for processing OSINT articles and predicting escalation likelihood in the Taiwan Strait region.

## 🎯 Overview

The StraitWatch backend pipeline processes translated articles through multiple ML models to provide:
- **Automated tagging** (keyword + ML-based)
- **Escalation prediction** (binary classification)
- **Article clustering** (HDBSCAN topic clustering)
- **Forecasting** (XGBoost time-series prediction)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Input Articles│    │  Tagging        │    │  Escalation     │
│   (Translated)  │───▶│  Pipeline       │───▶│  Inference      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                                        │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Final Output  │    │   HDBSCAN       │    │    SBERT        │
│  (Clustered)    │◀───│  Clustering     │◀───│  Embeddings     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                       ┌─────────────────┐
                       │    XGBoost      │
                       │   Forecasting   │
                       └─────────────────┘
```

## 📁 Core Components

### 1. Inference Pipeline (`pipelines/inference_pipeline.py`)
- **Model**: Pre-trained XGBoost classifier
- **Input**: Raw article text (title + content)
- **Output**: Escalation probability [0, 1]
- **Features**: Tag-based feature extraction (7 conflict tags + confidence)

### 2. Clustering Pipeline (`pipelines/cluster_articles.py`)
- **Algorithm**: HDBSCAN (density-based clustering)
- **Input**: SBERT embeddings (384-dimensional)
- **Output**: Cluster assignments + noise detection
- **Parameters**: `min_cluster_size=5`, `min_samples=3`

### 3. Forecasting Model (`train_forecasting_model.py`)
- **Model**: XGBoost Regressor
- **Input**: Daily event counts (GDELT/ACLED) + lagged features
- **Output**: Next-day escalation prediction
- **Features**: Lag features (t-1, t-2, t-3), rolling means, temporal features

### 4. Tagging Pipeline (`tagging/tagging_pipeline.py`)
- **Hybrid approach**: Keyword matching + ML classification
- **Models**: Hugging Face BART-MNLI (zero-shot classification)
- **Output**: Conflict-related tags + review flags

### 5. Model Evaluation (`evaluate_models.py`)
- **Metrics**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Forecasting**: MAE, RMSE, R² score
- **Clustering**: Silhouette score, cluster quality metrics

## 🚀 Quick Start

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from pipelines.inference_pipeline import predict_escalation; print('Setup complete!')"
```

### Basic Usage

#### 1. Run Complete Pipeline
```bash
# Process articles from default location
python run_backend_pipeline.py

# Process specific file
python run_backend_pipeline.py --input-file data/my_articles.json

# Skip certain steps
python run_backend_pipeline.py --skip-tagging --skip-clustering
```

#### 2. Individual Components

**Escalation Prediction**
```python
from pipelines.inference_pipeline import predict_escalation

score = predict_escalation("Military exercises conducted near Taiwan strait")
print(f"Escalation probability: {score:.3f}")
```

**Article Clustering**
```python
from pipelines.cluster_articles import cluster_articles

clustered_articles = cluster_articles(
    input_path="data/embedded_articles.json",
    min_cluster_size=5
)
```

**Model Evaluation**
```bash
python evaluate_models.py
```

#### 3. Demo Script
```bash
# Run demonstration with sample data
python demo_backend_pipeline.py
```

## 📊 Expected Input/Output

### Input Format
```json
[
  {
    "id": "article_001",
    "title": "Taiwan Military Exercise",
    "translated_text": "Taiwan conducts naval exercises...",
    "source": "news_source",
    "date": "2024-01-15T10:30:00Z"
  }
]
```

### Output Format
```json
[
  {
    "id": "article_001",
    "title": "Taiwan Military Exercise",
    "translated_text": "Taiwan conducts naval exercises...",
    "tags": ["military", "naval", "exercise"],
    "ml_tags": ["conflict", "military"],
    "escalation_score": 0.75,
    "cluster_id": 2,
    "embedding": [0.1, 0.2, ...],
    "needs_review": false
  }
]
```

## 🛠️ Configuration

### Pipeline Parameters
```python
# Clustering settings
MIN_CLUSTER_SIZE = 5
MIN_SAMPLES = 3

# Escalation thresholds
HIGH_RISK_THRESHOLD = 0.7
NEUTRAL_THRESHOLD = 0.5

# Feature dimensions
EMBEDDING_DIM = 384  # SBERT all-MiniLM-L6-v2
TAG_FEATURES = 7     # Conflict-related tags
```

### Model Paths
```
models/
├── xgboost_conflict_model_*.pkl    # Escalation classifier
├── xgboost_forecast_*.pkl          # Forecasting model
└── latest_forecast.json            # Model metadata
```

## 📈 Performance Metrics

### Current Performance (as of Phase 1)
- **Inference Speed**: ~0.01s per article
- **Clustering Quality**: Silhouette score varies by data
- **Memory Usage**: ~2GB for full pipeline
- **Throughput**: ~100 articles/minute

### Model Metrics
```json
{
  "inference_pipeline": {
    "avg_prediction_score": 0.018,
    "avg_processing_time": 0.001,
    "predictions_in_range": true
  },
  "clustering": {
    "n_clusters": 3,
    "noise_ratio": 0.2,
    "silhouette_score": 0.45
  }
}
```

## 🧪 Testing

### Run Test Suite
```bash
# All tests
python -m pytest tests/test_backend_pipeline.py -v

# Specific component tests
python -m pytest tests/test_backend_pipeline.py::TestInferencePipeline -v
python -m pytest tests/test_backend_pipeline.py::TestClusteringPipeline -v
```

### Test Coverage
- ✅ Inference pipeline functionality
- ✅ Clustering with/without embeddings
- ✅ Tagging pipeline integration
- ✅ Model evaluation metrics
- ✅ End-to-end pipeline execution

## 📂 File Structure

```
├── pipelines/
│   ├── inference_pipeline.py      # Escalation prediction
│   └── cluster_articles.py        # HDBSCAN clustering
├── tagging/
│   ├── tagging_pipeline.py        # Hybrid tagging system
│   └── tag_utils.py              # Tagging utilities
├── models/                        # Trained models
├── data/                         # Input/output data
├── logs/                         # Execution logs
├── tests/                        # Test suite
├── run_backend_pipeline.py      # Main pipeline orchestrator
├── train_forecasting_model.py   # XGBoost forecasting
├── evaluate_models.py           # Model evaluation
└── demo_backend_pipeline.py     # Demonstration script
```

## 🔧 Troubleshooting

### Common Issues

**1. Model Loading Errors**
```bash
# Check model availability
ls models/
# Ensure at least one XGBoost model exists
```

**2. Memory Issues**
```python
# Reduce batch size for embeddings
encoder.encode(texts, batch_size=16)
```

**3. Clustering No Results**
```python
# Reduce minimum cluster size
cluster_articles(min_cluster_size=3, min_samples=2)
```

**4. Import Errors**
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 🚧 Limitations & Future Work

### Current Limitations
- Forecasting model requires trained model file
- Small article sets may not cluster effectively  
- Tag vocabulary is fixed (7 conflict categories)
- No real-time data ingestion

### Phase 2 Enhancements
- Real-time Supabase integration
- Dynamic tag vocabulary expansion
- Advanced forecasting with external data
- API endpoints for real-time inference
- Distributed processing for scale

## 📞 Support

For technical issues or questions:
1. Check the troubleshooting section above
2. Review test cases in `tests/test_backend_pipeline.py`
3. Run the demo script: `python demo_backend_pipeline.py`
4. Check logs in `logs/` directory for detailed error information

---

**StraitWatch Phase 1** - Built for efficient, scalable OSINT analysis and conflict prediction. 