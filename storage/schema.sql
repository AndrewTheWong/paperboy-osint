-- Paperboy Database Schema
-- Unified schema for GDELT and OSINT data with consistent tagging/embedding support

-- GDELT Events Table
CREATE TABLE IF NOT EXISTS gdelt_events (
    id BIGSERIAL PRIMARY KEY,
    
    -- Core Event Data
    event_date DATE NOT NULL,
    event_id TEXT UNIQUE,
    event_code TEXT,
    actor1_country TEXT,
    actor2_country TEXT,
    goldstein_score FLOAT,
    avg_tone FLOAT,
    num_mentions INTEGER,
    num_articles INTEGER,
    quad_class INTEGER,
    
    -- Geographic Data
    location_name TEXT,
    country_code TEXT,
    latitude FLOAT,
    longitude FLOAT,
    primary_country TEXT,
    primary_city TEXT,
    primary_region TEXT,
    coordinates POINT,
    
    -- Processed Content
    title TEXT,
    content TEXT,
    summary TEXT,
    source_url TEXT,
    
    -- Tagging & Analysis
    keyword_tags JSONB DEFAULT '[]'::jsonb,
    semantic_tags JSONB DEFAULT '[]'::jsonb,
    geographic_tags JSONB DEFAULT '[]'::jsonb,
    sentiment_score FLOAT,
    escalation_score FLOAT,
    significance_score FLOAT,
    
    -- Embeddings
    title_embedding VECTOR(384),
    content_embedding VECTOR(384),
    
    -- Metadata
    language TEXT DEFAULT 'en',
    source TEXT DEFAULT 'GDELT',
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Indexes
    CONSTRAINT gdelt_events_event_date_check CHECK (event_date >= '2000-01-01'),
    CONSTRAINT gdelt_events_goldstein_check CHECK (goldstein_score BETWEEN -10 AND 10)
);

-- OSINT Articles Table  
CREATE TABLE IF NOT EXISTS osint_articles (
    id BIGSERIAL PRIMARY KEY,
    
    -- Core Article Data
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    url TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    
    -- Geographic Data
    primary_country TEXT,
    primary_city TEXT,
    primary_region TEXT,
    all_locations JSONB DEFAULT '[]'::jsonb,
    geographic_confidence FLOAT,
    coordinates POINT,
    
    -- Tagging & Analysis
    keyword_tags JSONB DEFAULT '[]'::jsonb,
    semantic_tags JSONB DEFAULT '[]'::jsonb,
    geographic_tags JSONB DEFAULT '[]'::jsonb,
    sentiment_score FLOAT,
    escalation_score FLOAT,
    significance_score FLOAT,
    
    -- Embeddings
    title_embedding VECTOR(384),
    content_embedding VECTOR(384),
    
    -- Temporal Data
    published_at TIMESTAMP WITH TIME ZONE,
    scraped_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Metadata
    language TEXT DEFAULT 'en',
    article_type TEXT DEFAULT 'news',
    word_count INTEGER,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT osint_articles_url_check CHECK (url ~ '^https?://'),
    CONSTRAINT osint_articles_word_count_check CHECK (word_count >= 0)
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_gdelt_events_date ON gdelt_events(event_date);
CREATE INDEX IF NOT EXISTS idx_gdelt_events_country ON gdelt_events(actor1_country);
CREATE INDEX IF NOT EXISTS idx_gdelt_events_goldstein ON gdelt_events(goldstein_score);
CREATE INDEX IF NOT EXISTS idx_gdelt_events_escalation ON gdelt_events(escalation_score);
CREATE INDEX IF NOT EXISTS idx_gdelt_events_tags ON gdelt_events USING GIN(keyword_tags);

CREATE INDEX IF NOT EXISTS idx_osint_articles_source ON osint_articles(source);
CREATE INDEX IF NOT EXISTS idx_osint_articles_published ON osint_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_osint_articles_country ON osint_articles(primary_country);
CREATE INDEX IF NOT EXISTS idx_osint_articles_escalation ON osint_articles(escalation_score);
CREATE INDEX IF NOT EXISTS idx_osint_articles_tags ON osint_articles USING GIN(keyword_tags);

-- Update triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_gdelt_events_updated_at BEFORE UPDATE ON gdelt_events 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_osint_articles_updated_at BEFORE UPDATE ON osint_articles 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(); 