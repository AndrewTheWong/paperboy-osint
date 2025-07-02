-- StraitWatch Schema - Simple tables for article ingestion and tagging

-- Articles table for storing ingested articles (using expected name)
CREATE TABLE osint_articles (
    id SERIAL PRIMARY KEY,
    source TEXT,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT,
    published_at TIMESTAMP,
    scraped_at TIMESTAMP DEFAULT NOW(),
    language TEXT DEFAULT 'en',
    relevant BOOLEAN DEFAULT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Also create the articles table for compatibility
CREATE TABLE articles (
    id SERIAL PRIMARY KEY,
    source TEXT,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT,
    published_at TIMESTAMP,
    language TEXT DEFAULT 'en',
    relevant BOOLEAN DEFAULT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Article tags table for storing NLP analysis results
CREATE TABLE article_tags (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    tag_type TEXT NOT NULL,
    tag_data JSONB,
    confidence REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Entities table for named entities
CREATE TABLE entities (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    confidence REAL DEFAULT 0.0,
    start_char INTEGER,
    end_char INTEGER,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Events table for extracted events (FIXED field names)
CREATE TABLE events (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    event_type TEXT,
    description TEXT,  -- FIXED: Use 'description' not 'event_description'
    escalation_score REAL DEFAULT 0.0,
    escalation_analysis TEXT,
    participants JSONB,
    location TEXT,
    datetime TEXT,  -- FIXED: Use 'datetime' not 'date_extracted'
    severity_rating TEXT DEFAULT 'medium',
    keywords JSONB,
    confidence_score REAL DEFAULT 0.0,  -- FIXED: Use 'confidence_score' not 'confidence'
    model_used TEXT DEFAULT 'enhanced-event-classifier',
    context_sentence TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Article sentiment table
CREATE TABLE article_sentiment (
    id SERIAL PRIMARY KEY,
    article_id INTEGER REFERENCES articles(id) ON DELETE CASCADE,
    sentiment_score REAL DEFAULT 0.0,
    sentiment_label TEXT,
    confidence REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Escalation series table for time series data
CREATE TABLE escalation_series (
    id SERIAL PRIMARY KEY,
    date DATE UNIQUE,
    event_count INTEGER DEFAULT 0,
    avg_escalation_score REAL DEFAULT 0.0,
    max_escalation_score REAL DEFAULT 0.0,
    avg_sentiment REAL DEFAULT 0.0,
    high_risk_events INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Forecasts table for model predictions
CREATE TABLE forecasts (
    id SERIAL PRIMARY KEY,
    forecast_type TEXT DEFAULT 'escalation_forecast',
    model_name TEXT,
    model_used TEXT,
    forecast_data JSONB,
    forecast_horizon INTEGER DEFAULT 7,
    confidence REAL DEFAULT 0.0,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Agent runs table for monitoring
CREATE TABLE agent_runs (
    id SERIAL PRIMARY KEY,
    agent_name TEXT,
    start_time TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds REAL,
    success BOOLEAN,
    result_data JSONB,
    error_message TEXT,
    error_count INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Intelligence reports table for storing daily assessments
CREATE TABLE intelligence_reports (
    id SERIAL PRIMARY KEY,
    report_date DATE,
    report_type TEXT DEFAULT 'daily_intelligence_assessment',
    threat_level TEXT,
    escalation_risk TEXT,
    report_data JSONB,
    file_path TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Clusters table for storing thematic clusters
CREATE TABLE clusters (
    id SERIAL PRIMARY KEY,
    theme TEXT NOT NULL,
    summary TEXT,
    article_ids INTEGER[],
    status TEXT DEFAULT 'processing',
    priority_level TEXT DEFAULT 'MEDIUM',
    escalation_score REAL DEFAULT 0.0,
    region TEXT,
    topic TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_osint_articles_url ON osint_articles(url);
CREATE INDEX idx_osint_articles_relevant ON osint_articles(relevant);
CREATE INDEX idx_osint_articles_created_at ON osint_articles(created_at);

CREATE INDEX idx_articles_url ON articles(url);
CREATE INDEX idx_articles_relevant ON articles(relevant);
CREATE INDEX idx_articles_created_at ON articles(created_at);

CREATE INDEX idx_article_tags_article_id ON article_tags(article_id);
CREATE INDEX idx_article_tags_tag_type ON article_tags(tag_type);

CREATE INDEX idx_entities_article_id ON entities(article_id);
CREATE INDEX idx_entities_entity_type ON entities(entity_type);

CREATE INDEX idx_events_article_id ON events(article_id);
CREATE INDEX idx_events_escalation_score ON events(escalation_score);

CREATE INDEX idx_escalation_series_date ON escalation_series(date);

CREATE INDEX idx_forecasts_forecast_type ON forecasts(forecast_type);
CREATE INDEX idx_forecasts_model_name ON forecasts(model_name);
CREATE INDEX idx_forecasts_created_at ON forecasts(created_at);

CREATE INDEX idx_agent_runs_agent_name ON agent_runs(agent_name);
CREATE INDEX idx_agent_runs_created_at ON agent_runs(created_at);

CREATE INDEX idx_clusters_status ON clusters(status);
CREATE INDEX idx_clusters_created_at ON clusters(created_at);
CREATE INDEX idx_clusters_priority_level ON clusters(priority_level);
