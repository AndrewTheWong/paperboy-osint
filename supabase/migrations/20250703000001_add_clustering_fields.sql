-- Add clustering fields to articles table
-- Migration: 20250703000001_add_clustering_fields.sql

-- Add clustering-related columns
ALTER TABLE articles 
ADD COLUMN IF NOT EXISTS embedding float8[],
ADD COLUMN IF NOT EXISTS cluster_label TEXT,
ADD COLUMN IF NOT EXISTS cluster_description TEXT,
ADD COLUMN IF NOT EXISTS tag_categories JSONB DEFAULT '{}',
ADD COLUMN IF NOT EXISTS priority_level TEXT DEFAULT 'LOW';

-- Create index on embedding for similarity search
CREATE INDEX IF NOT EXISTS idx_articles_embedding ON articles USING gin(embedding);

-- Add comment to document the new fields
COMMENT ON COLUMN articles.embedding IS 'Article embedding vector for similarity search';
COMMENT ON COLUMN articles.cluster_label IS 'Human-readable cluster label';
COMMENT ON COLUMN articles.cluster_description IS 'Detailed cluster description';
COMMENT ON COLUMN articles.tag_categories IS 'Enhanced tagging categories with confidence scores';
COMMENT ON COLUMN articles.priority_level IS 'Article priority level (HIGH/MEDIUM/LOW)'; 