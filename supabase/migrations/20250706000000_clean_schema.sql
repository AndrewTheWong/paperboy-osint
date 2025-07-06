-- StraitWatch Clean Schema Migration
-- Single migration with articles and clusters tables only
-- Migration: 20250706000000_clean_schema.sql

-- Drop existing tables if they exist
DROP TABLE IF EXISTS clusters CASCADE;
DROP TABLE IF EXISTS articles CASCADE;

-- Create articles table
CREATE TABLE articles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  title TEXT NOT NULL,
  url TEXT UNIQUE NOT NULL,
  body TEXT,                              -- Raw article content
  cleaned TEXT,                           -- Cleaned article content
  source_name TEXT,                       -- Source name
  published_at TIMESTAMP,
  embedding REAL[],                       -- Article embedding (384 dimensions as array)
  tags JSONB DEFAULT '[]'::jsonb,        -- List: ["GEO:China", "CMD:Naval Operations", ...]
  entities JSONB DEFAULT '[]'::jsonb,    -- List: ["China", "Naval Operations", ...]
  cluster_id INTEGER,                     -- Cluster assignment
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);

-- Create clusters table
CREATE TABLE clusters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cluster_id INTEGER UNIQUE NOT NULL,     -- FAISS cluster ID
  top_tags JSONB DEFAULT '[]'::jsonb,    -- Top tags from cluster members
  top_entities JSONB DEFAULT '[]'::jsonb, -- Top entities from cluster members
  summary TEXT,                           -- Generated summary of cluster content
  representative_article_id UUID REFERENCES articles(id), -- Representative article
  member_count INTEGER DEFAULT 0,         -- Number of articles in cluster
  topic TEXT,                             -- Main topic of cluster
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX idx_articles_cluster_id ON articles(cluster_id);
CREATE INDEX idx_articles_published_at ON articles(published_at);
CREATE INDEX idx_articles_source_name ON articles(source_name);
CREATE INDEX idx_articles_created_at ON articles(created_at);

-- Create indexes for clusters table
CREATE INDEX idx_clusters_cluster_id ON clusters(cluster_id);
CREATE INDEX idx_clusters_representative_article_id ON clusters(representative_article_id);
CREATE INDEX idx_clusters_member_count ON clusters(member_count);
CREATE INDEX idx_clusters_created_at ON clusters(created_at);

-- Create indexes on JSONB fields
CREATE INDEX idx_articles_tags ON articles USING GIN(tags);
CREATE INDEX idx_articles_entities ON articles USING GIN(entities);
CREATE INDEX idx_clusters_top_tags ON clusters USING GIN(top_tags);
CREATE INDEX idx_clusters_top_entities ON clusters USING GIN(top_entities);

-- Enable RLS (Row Level Security)
ALTER TABLE articles ENABLE ROW LEVEL SECURITY;
ALTER TABLE clusters ENABLE ROW LEVEL SECURITY;

-- Create policies for public access
CREATE POLICY "Allow all access to articles" ON articles
    FOR ALL USING (true);

CREATE POLICY "Allow all access to clusters" ON clusters
    FOR ALL USING (true);

-- Add trigger to update updated_at timestamp for articles
CREATE OR REPLACE FUNCTION update_articles_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_articles_updated_at
    BEFORE UPDATE ON articles
    FOR EACH ROW
    EXECUTE FUNCTION update_articles_updated_at();

-- Add trigger to update updated_at timestamp for clusters
CREATE OR REPLACE FUNCTION update_clusters_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_clusters_updated_at
    BEFORE UPDATE ON clusters
    FOR EACH ROW
    EXECUTE FUNCTION update_clusters_updated_at();

-- Add comments
COMMENT ON TABLE articles IS 'Articles table for OSINT clustering system';
COMMENT ON COLUMN articles.embedding IS '384-dimensional vector embedding for similarity search';
COMMENT ON COLUMN articles.tags IS 'JSON array of tags with prefixes (GEO:, CMD:, ACT:, etc.)';
COMMENT ON COLUMN articles.entities IS 'JSON array of named entities extracted from article';
COMMENT ON COLUMN articles.cluster_id IS 'Cluster assignment from FAISS clustering';

COMMENT ON TABLE clusters IS 'Clusters table for OSINT clustering system';
COMMENT ON COLUMN clusters.cluster_id IS 'FAISS cluster identifier';
COMMENT ON COLUMN clusters.top_tags IS 'Most common tags from cluster members';
COMMENT ON COLUMN clusters.top_entities IS 'Most common entities from cluster members';
COMMENT ON COLUMN clusters.summary IS 'Generated summary of cluster content';
COMMENT ON COLUMN clusters.representative_article_id IS 'Article closest to cluster centroid';
COMMENT ON COLUMN clusters.member_count IS 'Number of articles in this cluster';
COMMENT ON COLUMN clusters.topic IS 'Main topic of the cluster'; 