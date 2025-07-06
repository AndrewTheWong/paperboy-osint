-- Add translation fields to articles table
-- Migration: 20250706000001_add_translation_fields.sql

-- Add translation-related columns to articles table
ALTER TABLE articles 
ADD COLUMN title_original TEXT,
ADD COLUMN content_original TEXT,
ADD COLUMN title_language TEXT DEFAULT 'en',
ADD COLUMN content_language TEXT DEFAULT 'en';

-- Add comments for new columns
COMMENT ON COLUMN articles.title_original IS 'Original title before translation';
COMMENT ON COLUMN articles.content_original IS 'Original content before translation';
COMMENT ON COLUMN articles.title_language IS 'Language code of original title (en, zh, ja, ko, etc.)';
COMMENT ON COLUMN articles.content_language IS 'Language code of original content (en, zh, ja, ko, etc.)';

-- Create indexes for language fields
CREATE INDEX idx_articles_title_language ON articles(title_language);
CREATE INDEX idx_articles_content_language ON articles(content_language); 