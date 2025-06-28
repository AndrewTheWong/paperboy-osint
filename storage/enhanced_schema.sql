-- StraitWatch Enhanced Schema

CREATE TABLE IF NOT EXISTS articles (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source TEXT NOT NULL,
    url TEXT UNIQUE,
    title TEXT,
    content TEXT,
    published_at TIMESTAMPTZ,
    language TEXT DEFAULT 'en',
    embedding vector(384),
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS article_tags (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id) ON DELETE CASCADE,
    tag_type TEXT CHECK (tag_type IN ('entity', 'relation', 'event', 'escalation')),
    tag_value TEXT,
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS events (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    article_id uuid REFERENCES articles(id),
    event_type TEXT,
    event_date TIMESTAMPTZ,
    location TEXT,
    severity_score FLOAT,
    confidence FLOAT,
    description TEXT,
    created_at TIMESTAMPTZ DEFAULT now()
);

CREATE TABLE IF NOT EXISTS forecasts (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name TEXT,
    forecast_date DATE,
    prediction_horizon INTEGER,
    escalation_score FLOAT,
    confidence_interval JSONB,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Enhanced Tagging Schema for Paperboy
-- Tables for storing relation extraction and event extraction data

-- Named Entities Table
CREATE TABLE IF NOT EXISTS named_entities (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES osint_articles(id) ON DELETE CASCADE,
    
    -- Entity Information
    entity_text TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    entity_subtype TEXT,
    confidence_score FLOAT DEFAULT 0.0,
    
    -- Geographic Information (if applicable)
    country TEXT,
    city TEXT,
    region TEXT,
    coordinates POINT,
    
    -- Metadata
    start_position INTEGER,
    end_position INTEGER,
    sentence_context TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT named_entities_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    CONSTRAINT named_entities_position_check CHECK (start_position >= 0 AND end_position > start_position)
);

-- Relations Table
CREATE TABLE IF NOT EXISTS entity_relations (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES osint_articles(id) ON DELETE CASCADE,
    
    -- Relation Information
    relation_type TEXT NOT NULL,
    relation_subtype TEXT,
    confidence_score FLOAT DEFAULT 0.0,
    
    -- Entity References
    entity1_id BIGINT REFERENCES named_entities(id) ON DELETE CASCADE,
    entity2_id BIGINT REFERENCES named_entities(id) ON DELETE CASCADE,
    entity1_text TEXT NOT NULL,
    entity2_text TEXT NOT NULL,
    entity1_type TEXT NOT NULL,
    entity2_type TEXT NOT NULL,
    
    -- Relation Context
    sentence_context TEXT,
    extraction_method TEXT, -- 'dependency_parsing', 'pattern_based', 'transformer'
    
    -- Metadata
    source_sentence TEXT,
    start_position INTEGER,
    end_position INTEGER,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT entity_relations_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    CONSTRAINT entity_relations_entities_different CHECK (entity1_id != entity2_id)
);

-- Events Table
CREATE TABLE IF NOT EXISTS extracted_events (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES osint_articles(id) ON DELETE CASCADE,
    
    -- Event Information
    event_type TEXT NOT NULL,
    event_subtype TEXT,
    event_title TEXT,
    event_description TEXT,
    confidence_score FLOAT DEFAULT 0.0,
    
    -- Temporal Information
    event_date DATE,
    event_time TIMESTAMP WITH TIME ZONE,
    date_confidence FLOAT DEFAULT 0.0,
    
    -- Geographic Information
    location_country TEXT,
    location_city TEXT,
    location_region TEXT,
    coordinates POINT,
    location_confidence FLOAT DEFAULT 0.0,
    
    -- Event Context
    source_sentences TEXT[],
    extraction_method TEXT, -- 'rule_based', 'ml_model', 'hybrid'
    
    -- Metadata
    event_importance FLOAT DEFAULT 0.0,
    escalation_indicator BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT extracted_events_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    CONSTRAINT extracted_events_importance_check CHECK (event_importance BETWEEN 0.0 AND 1.0)
);

-- Event Participants Table
CREATE TABLE IF NOT EXISTS event_participants (
    id BIGSERIAL PRIMARY KEY,
    event_id BIGINT REFERENCES extracted_events(id) ON DELETE CASCADE,
    entity_id BIGINT REFERENCES named_entities(id) ON DELETE CASCADE,
    
    -- Participant Information
    participant_role TEXT NOT NULL, -- 'attacker', 'victim', 'participant', 'signatory', etc.
    participant_type TEXT, -- 'country', 'organization', 'person', 'military_unit'
    confidence_score FLOAT DEFAULT 0.0,
    
    -- Context
    role_context TEXT,
    sentence_context TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT event_participants_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0)
);

-- Event Relations Table (for event-to-event relationships)
CREATE TABLE IF NOT EXISTS event_relations (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES osint_articles(id) ON DELETE CASCADE,
    
    -- Event References
    event1_id BIGINT REFERENCES extracted_events(id) ON DELETE CASCADE,
    event2_id BIGINT REFERENCES extracted_events(id) ON DELETE CASCADE,
    
    -- Relation Information
    relation_type TEXT NOT NULL, -- 'causes', 'follows', 'triggers', 'escalates', etc.
    confidence_score FLOAT DEFAULT 0.0,
    
    -- Context
    relation_context TEXT,
    extraction_method TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT event_relations_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    CONSTRAINT event_relations_events_different CHECK (event1_id != event2_id)
);

-- Enhanced Analytics Summary Table
CREATE TABLE IF NOT EXISTS enhanced_analytics_summary (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES osint_articles(id) ON DELETE CASCADE,
    
    -- Summary Statistics
    total_entities INTEGER DEFAULT 0,
    total_relations INTEGER DEFAULT 0,
    total_events INTEGER DEFAULT 0,
    total_participants INTEGER DEFAULT 0,
    
    -- Key Metrics
    escalation_score FLOAT DEFAULT 0.0,
    conflict_indicators INTEGER DEFAULT 0,
    tension_indicators INTEGER DEFAULT 0,
    cooperation_indicators INTEGER DEFAULT 0,
    
    -- Key Actors
    key_actors JSONB DEFAULT '[]'::jsonb, -- Array of most important entities
    key_relations JSONB DEFAULT '[]'::jsonb, -- Array of most important relations
    key_events JSONB DEFAULT '[]'::jsonb, -- Array of most important events
    
    -- Analysis Results
    insights JSONB DEFAULT '[]'::jsonb, -- Array of generated insights
    recommendations JSONB DEFAULT '[]'::jsonb, -- Array of recommendations
    
    -- Processing Metadata
    processing_time_seconds FLOAT,
    model_versions JSONB DEFAULT '{}'::jsonb, -- Versions of models used
    extraction_methods JSONB DEFAULT '[]'::jsonb, -- Methods used for extraction
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT enhanced_analytics_escalation_check CHECK (escalation_score BETWEEN 0.0 AND 1.0)
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_named_entities_article ON named_entities(article_id);
CREATE INDEX IF NOT EXISTS idx_named_entities_type ON named_entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_named_entities_text ON named_entities(entity_text);
CREATE INDEX IF NOT EXISTS idx_named_entities_confidence ON named_entities(confidence_score);

CREATE INDEX IF NOT EXISTS idx_entity_relations_article ON entity_relations(article_id);
CREATE INDEX IF NOT EXISTS idx_entity_relations_type ON entity_relations(relation_type);
CREATE INDEX IF NOT EXISTS idx_entity_relations_entities ON entity_relations(entity1_id, entity2_id);
CREATE INDEX IF NOT EXISTS idx_entity_relations_confidence ON entity_relations(confidence_score);

CREATE INDEX IF NOT EXISTS idx_extracted_events_article ON extracted_events(article_id);
CREATE INDEX IF NOT EXISTS idx_extracted_events_type ON extracted_events(event_type);
CREATE INDEX IF NOT EXISTS idx_extracted_events_date ON extracted_events(event_date);
CREATE INDEX IF NOT EXISTS idx_extracted_events_escalation ON extracted_events(escalation_indicator);
CREATE INDEX IF NOT EXISTS idx_extracted_events_confidence ON extracted_events(confidence_score);

CREATE INDEX IF NOT EXISTS idx_event_participants_event ON event_participants(event_id);
CREATE INDEX IF NOT EXISTS idx_event_participants_entity ON event_participants(entity_id);
CREATE INDEX IF NOT EXISTS idx_event_participants_role ON event_participants(participant_role);

CREATE INDEX IF NOT EXISTS idx_event_relations_article ON event_relations(article_id);
CREATE INDEX IF NOT EXISTS idx_event_relations_events ON event_relations(event1_id, event2_id);
CREATE INDEX IF NOT EXISTS idx_event_relations_type ON event_relations(relation_type);

CREATE INDEX IF NOT EXISTS idx_enhanced_analytics_article ON enhanced_analytics_summary(article_id);
CREATE INDEX IF NOT EXISTS idx_enhanced_analytics_escalation ON enhanced_analytics_summary(escalation_score);

-- Update triggers
CREATE OR REPLACE FUNCTION update_enhanced_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_named_entities_updated_at BEFORE UPDATE ON named_entities 
    FOR EACH ROW EXECUTE FUNCTION update_enhanced_updated_at_column();

CREATE TRIGGER update_entity_relations_updated_at BEFORE UPDATE ON entity_relations 
    FOR EACH ROW EXECUTE FUNCTION update_enhanced_updated_at_column();

CREATE TRIGGER update_extracted_events_updated_at BEFORE UPDATE ON extracted_events 
    FOR EACH ROW EXECUTE FUNCTION update_enhanced_updated_at_column();

CREATE TRIGGER update_event_participants_updated_at BEFORE UPDATE ON event_participants 
    FOR EACH ROW EXECUTE FUNCTION update_enhanced_updated_at_column();

CREATE TRIGGER update_event_relations_updated_at BEFORE UPDATE ON event_relations 
    FOR EACH ROW EXECUTE FUNCTION update_enhanced_updated_at_column();

CREATE TRIGGER update_enhanced_analytics_updated_at BEFORE UPDATE ON enhanced_analytics_summary 
    FOR EACH ROW EXECUTE FUNCTION update_enhanced_updated_at_column();

-- Main Events Table
CREATE TABLE IF NOT EXISTS events (
    id BIGSERIAL PRIMARY KEY,
    
    -- Core Article Data
    title TEXT NOT NULL,
    content TEXT,
    summary TEXT,
    url TEXT UNIQUE NOT NULL,
    source TEXT NOT NULL,
    source_name TEXT,
    language_original TEXT DEFAULT 'en',
    
    -- Geographic Data
    primary_country TEXT,
    primary_city TEXT,
    primary_region TEXT,
    all_locations JSONB DEFAULT '[]'::jsonb,
    geographic_confidence FLOAT,
    coordinates POINT,
    
    -- Temporal Data
    published_at TIMESTAMP WITH TIME ZONE,
    scraped_at TIMESTAMP WITH TIME ZONE NOT NULL,
    extracted_datetime TEXT, -- From article content
    
    -- Embeddings
    title_embedding VECTOR(384),
    content_embedding VECTOR(384),
    
    -- Metadata
    word_count INTEGER,
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT events_url_check CHECK (url ~ '^https?://'),
    CONSTRAINT events_word_count_check CHECK (word_count >= 0)
);

-- Entities Table
CREATE TABLE IF NOT EXISTS entities (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES events(id) ON DELETE CASCADE,
    
    -- Entity Data
    entity TEXT NOT NULL,
    entity_type TEXT NOT NULL, -- PERSON, ORG, GPE, LOCATION, FACILITY, AIRSPACE, MARITIME_ZONE, EQUIPMENT, PLATFORM, CYBERTOOL, DATETIME
    linked_id TEXT, -- Wikidata ID or future KB ID
    normalized_name TEXT,
    
    -- Context
    context_sentence TEXT,
    confidence_score FLOAT DEFAULT 0.8,
    
    -- Metadata
    model_used TEXT DEFAULT 'spacy-en_core_web_sm',
    human_reviewed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT entities_type_check CHECK (entity_type IN ('PERSON', 'ORG', 'GPE', 'LOCATION', 'FACILITY', 'AIRSPACE', 'MARITIME_ZONE', 'EQUIPMENT', 'PLATFORM', 'CYBERTOOL', 'DATETIME')),
    CONSTRAINT entities_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0)
);

-- Relations Table
CREATE TABLE IF NOT EXISTS relations (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES events(id) ON DELETE CASCADE,
    
    -- Relation Data
    subject TEXT NOT NULL,
    predicate TEXT NOT NULL, -- deployed_to, conducted, met_with, threatened, violated, entered, announced, tracked_by, detected_by
    object TEXT NOT NULL,
    
    -- Context
    context_sentence TEXT,
    confidence_score FLOAT DEFAULT 0.8,
    
    -- Metadata
    model_used TEXT DEFAULT 'custom-relation-extractor',
    human_reviewed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT relations_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0)
);

-- Event Tags Table
CREATE TABLE IF NOT EXISTS event_tags (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES events(id) ON DELETE CASCADE,
    
    -- Event Classification
    event_type TEXT NOT NULL, -- military_movement, live_fire_exercise, gray_zone_operation, airspace_intrusion, naval_patrol, diplomatic_statement, summit_or_meeting, arms_sale, cyber_attack, economic_signal, legislative_action, disinfo_campaign, incident_reported
    
    -- Event Details
    participants JSONB DEFAULT '[]'::jsonb, -- Array of entity names
    location TEXT,
    datetime TEXT, -- Extracted from article
    severity_rating TEXT, -- low, medium, high, critical
    
    -- Confidence and Metadata
    confidence_score FLOAT DEFAULT 0.8,
    model_used TEXT DEFAULT 'custom-event-classifier',
    human_reviewed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT event_tags_type_check CHECK (event_type IN ('military_movement', 'live_fire_exercise', 'gray_zone_operation', 'airspace_intrusion', 'naval_patrol', 'diplomatic_statement', 'summit_or_meeting', 'arms_sale', 'cyber_attack', 'economic_signal', 'legislative_action', 'disinfo_campaign', 'incident_reported')),
    CONSTRAINT event_tags_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0),
    CONSTRAINT event_tags_severity_check CHECK (severity_rating IN ('low', 'medium', 'high', 'critical'))
);

-- Article Sentiment Table
CREATE TABLE IF NOT EXISTS article_sentiment (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES events(id) ON DELETE CASCADE,
    
    -- Sentiment Analysis
    sentiment_toward_Taiwan TEXT, -- hostile, neutral, supportive
    escalation_level TEXT, -- low, medium, high
    intent_signal TEXT, -- coercive, defensive, deterrent, symbolic
    strategic_posture_change BOOLEAN DEFAULT FALSE,
    info_warfare_detected BOOLEAN DEFAULT FALSE,
    
    -- Confidence
    confidence_score FLOAT DEFAULT 0.8,
    model_used TEXT DEFAULT 'custom-sentiment-analyzer',
    human_reviewed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT article_sentiment_taiwan_check CHECK (sentiment_toward_Taiwan IN ('hostile', 'neutral', 'supportive')),
    CONSTRAINT article_sentiment_escalation_check CHECK (escalation_level IN ('low', 'medium', 'high')),
    CONSTRAINT article_sentiment_intent_check CHECK (intent_signal IN ('coercive', 'defensive', 'deterrent', 'symbolic')),
    CONSTRAINT article_sentiment_confidence_check CHECK (confidence_score BETWEEN 0.0 AND 1.0)
);

-- Article Tags (Joiner Table for Quick Reference)
CREATE TABLE IF NOT EXISTS article_tags (
    id BIGSERIAL PRIMARY KEY,
    article_id BIGINT REFERENCES events(id) ON DELETE CASCADE,
    
    -- Quick Reference Tags
    tag_type TEXT NOT NULL, -- entity, relation, event, sentiment
    tag_value TEXT NOT NULL,
    tag_metadata JSONB DEFAULT '{}'::jsonb,
    
    -- Indexing
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT article_tags_type_check CHECK (tag_type IN ('entity', 'relation', 'event', 'sentiment'))
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_events_source ON events(source);
CREATE INDEX IF NOT EXISTS idx_events_published ON events(published_at);
CREATE INDEX IF NOT EXISTS idx_events_country ON events(primary_country);
CREATE INDEX IF NOT EXISTS idx_events_created ON events(created_at);

CREATE INDEX IF NOT EXISTS idx_entities_article ON entities(article_id);
CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);
CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(entity);
CREATE INDEX IF NOT EXISTS idx_entities_confidence ON entities(confidence_score);

CREATE INDEX IF NOT EXISTS idx_relations_article ON relations(article_id);
CREATE INDEX IF NOT EXISTS idx_relations_predicate ON relations(predicate);
CREATE INDEX IF NOT EXISTS idx_relations_subject ON relations(subject);
CREATE INDEX IF NOT EXISTS idx_relations_object ON relations(object);
CREATE INDEX IF NOT EXISTS idx_relations_confidence ON relations(confidence_score);

CREATE INDEX IF NOT EXISTS idx_event_tags_article ON event_tags(article_id);
CREATE INDEX IF NOT EXISTS idx_event_tags_type ON event_tags(event_type);
CREATE INDEX IF NOT EXISTS idx_event_tags_severity ON event_tags(severity_rating);
CREATE INDEX IF NOT EXISTS idx_event_tags_confidence ON event_tags(confidence_score);

CREATE INDEX IF NOT EXISTS idx_article_sentiment_article ON article_sentiment(article_id);
CREATE INDEX IF NOT EXISTS idx_article_sentiment_taiwan ON article_sentiment(sentiment_toward_Taiwan);
CREATE INDEX IF NOT EXISTS idx_article_sentiment_escalation ON article_sentiment(escalation_level);

CREATE INDEX IF NOT EXISTS idx_article_tags_article ON article_tags(article_id);
CREATE INDEX IF NOT EXISTS idx_article_tags_type ON article_tags(tag_type);
CREATE INDEX IF NOT EXISTS idx_article_tags_value ON article_tags(tag_value);

-- Update triggers
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_events_updated_at BEFORE UPDATE ON events 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column(); 