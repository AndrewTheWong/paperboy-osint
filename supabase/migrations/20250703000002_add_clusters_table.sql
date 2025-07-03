-- Add clusters table and missing relevant column
-- Migration: 20250703000002_add_clusters_table.sql

-- Add relevant column to articles table
ALTER TABLE articles 
ADD COLUMN IF NOT EXISTS relevant BOOLEAN DEFAULT true;

-- Create clusters table
CREATE TABLE IF NOT EXISTS clusters (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  cluster_id TEXT UNIQUE NOT NULL,           -- FAISS cluster ID
  theme TEXT,                                -- Cluster theme/topic
  summary TEXT,                              -- Cluster summary
  article_ids TEXT[],                        -- Array of article IDs in this cluster
  region TEXT,                               -- Primary region for this cluster
  topic TEXT,                                -- Primary topic for this cluster
  priority_level TEXT DEFAULT 'MEDIUM',      -- Priority level
  escalation_score FLOAT DEFAULT 0.0,       -- Escalation score
  status TEXT DEFAULT 'pending',             -- Status: pending, completed, archived
  created_at TIMESTAMP DEFAULT now(),
  updated_at TIMESTAMP DEFAULT now()
);

-- Create indexes for clusters table
CREATE INDEX IF NOT EXISTS idx_clusters_cluster_id ON clusters(cluster_id);
CREATE INDEX IF NOT EXISTS idx_clusters_region ON clusters(region);
CREATE INDEX IF NOT EXISTS idx_clusters_topic ON clusters(topic);
CREATE INDEX IF NOT EXISTS idx_clusters_status ON clusters(status);
CREATE INDEX IF NOT EXISTS idx_clusters_created_at ON clusters(created_at);

-- Enable RLS on clusters table
ALTER TABLE clusters ENABLE ROW LEVEL SECURITY;

-- Create policy for public access
CREATE POLICY "Allow all access to clusters" ON clusters
    FOR ALL USING (true);

-- Add comments
COMMENT ON COLUMN articles.relevant IS 'Whether article is relevant for analysis';
COMMENT ON COLUMN clusters.cluster_id IS 'FAISS cluster identifier';
COMMENT ON COLUMN clusters.theme IS 'Human-readable cluster theme';
COMMENT ON COLUMN clusters.summary IS 'Generated summary of cluster content';
COMMENT ON COLUMN clusters.article_ids IS 'Array of article UUIDs in this cluster';
COMMENT ON COLUMN clusters.region IS 'Primary geographic region for cluster';
COMMENT ON COLUMN clusters.topic IS 'Primary topic category for cluster';
COMMENT ON COLUMN clusters.priority_level IS 'Priority level (HIGH/MEDIUM/LOW)';
COMMENT ON COLUMN clusters.escalation_score IS 'Escalation score (0-10)';
COMMENT ON COLUMN clusters.status IS 'Processing status'; 