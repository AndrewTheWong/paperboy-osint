-- Fix missing columns in articles table
-- Migration: 20250703000003_fix_missing_columns.sql

-- Add missing created_at column (alias for inserted_at)
ALTER TABLE articles 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT now();

-- Update existing rows to have created_at = inserted_at
UPDATE articles 
SET created_at = inserted_at 
WHERE created_at IS NULL;

-- Add any other missing columns that might be needed
ALTER TABLE articles 
ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT now(),
ADD COLUMN IF NOT EXISTS processed_at TIMESTAMP,
ADD COLUMN IF NOT EXISTS status TEXT DEFAULT 'pending';

-- Create indexes for new columns
CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);
CREATE INDEX IF NOT EXISTS idx_articles_updated_at ON articles(updated_at);
CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);

-- Add comments
COMMENT ON COLUMN articles.created_at IS 'Article creation timestamp (alias for inserted_at)';
COMMENT ON COLUMN articles.updated_at IS 'Last update timestamp';
COMMENT ON COLUMN articles.processed_at IS 'When article was processed by pipeline';
COMMENT ON COLUMN articles.status IS 'Processing status (pending, processed, error)'; 