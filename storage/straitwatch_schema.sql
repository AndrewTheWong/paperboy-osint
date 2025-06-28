-- StraitWatch Complete Schema
-- Taiwan Strait Early Warning and Intelligence System

-- Enable necessary extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 1. Articles table for ingested OSINT articles
CREATE TABLE IF NOT EXISTS articles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT,
    url TEXT,
    title TEXT,
    published_at TIMESTAMPTZ,
    content TEXT,
    language TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 2. Article tags for NLP analysis results
CREATE TABLE IF NOT EXISTS article_tags (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id) ON DELETE CASCADE,
    tag_type TEXT CHECK (tag_type IN ('entity', 'relation', 'event', 'escalation')),
    tag_data JSONB,
    confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 3. Named entities extracted from articles
CREATE TABLE IF NOT EXISTS entities (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    confidence FLOAT DEFAULT 0.0,
    position_start INTEGER,
    position_end INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 4. Events extracted from articles
CREATE TABLE IF NOT EXISTS events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id) ON DELETE CASCADE,
    event_type TEXT,
    event_date TIMESTAMPTZ,
    location TEXT,
    severity_score FLOAT DEFAULT 0.0,
    confidence FLOAT DEFAULT 0.0,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 5. Article sentiment analysis
CREATE TABLE IF NOT EXISTS article_sentiment (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id) ON DELETE CASCADE,
    sentiment_score FLOAT DEFAULT 0.0,
    sentiment_label TEXT,
    confidence FLOAT DEFAULT 0.0,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 6. Forecasts from ML models
CREATE TABLE IF NOT EXISTS forecasts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT,
    forecast_date DATE,
    prediction_horizon INTEGER,
    escalation_score FLOAT,
    confidence_interval JSONB,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 7. Intelligence reports
CREATE TABLE IF NOT EXISTS intelligence_reports (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    report_date DATE,
    report_type TEXT DEFAULT 'daily',
    content TEXT,
    summary JSONB,
    threat_level TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 8. Agent run logs for monitoring
CREATE TABLE IF NOT EXISTS agent_runs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_name TEXT,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_seconds FLOAT,
    success BOOLEAN,
    result_data JSONB,
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- 9. Article embeddings for vector search (optional)
CREATE TABLE IF NOT EXISTS article_embeddings (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id) ON DELETE CASCADE,
    embedding vector(384),
    model_name TEXT DEFAULT 'sentence-transformers',
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_articles_published_at ON articles(published_at);
CREATE INDEX IF NOT EXISTS idx_articles_source ON articles(source);
CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);

CREATE INDEX IF NOT EXISTS idx_article_tags_article_id ON article_tags(article_id);
CREATE INDEX IF NOT EXISTS idx_article_tags_tag_type ON article_tags(tag_type);
CREATE INDEX IF NOT EXISTS idx_article_tags_created_at ON article_tags(created_at);

CREATE INDEX IF NOT EXISTS idx_entities_article_id ON entities(article_id);
CREATE INDEX IF NOT EXISTS idx_entities_entity_type ON entities(entity_type);

CREATE INDEX IF NOT EXISTS idx_events_article_id ON events(article_id);
CREATE INDEX IF NOT EXISTS idx_events_event_date ON events(event_date);
CREATE INDEX IF NOT EXISTS idx_events_event_type ON events(event_type);

CREATE INDEX IF NOT EXISTS idx_forecasts_forecast_date ON forecasts(forecast_date);
CREATE INDEX IF NOT EXISTS idx_forecasts_model_name ON forecasts(model_name);

CREATE INDEX IF NOT EXISTS idx_agent_runs_agent_name ON agent_runs(agent_name);
CREATE INDEX IF NOT EXISTS idx_agent_runs_start_time ON agent_runs(start_time);

-- Vector index for embeddings (if using pgvector)
CREATE INDEX IF NOT EXISTS idx_article_embeddings_vector ON article_embeddings USING ivfflat (embedding vector_cosine_ops); 