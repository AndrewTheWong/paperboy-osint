-- StraitWatch Articles Schema for Local Supabase
-- Updated to match production schema

CREATE TABLE articles (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  original_id UUID,                  -- From upstream pipeline
  title TEXT NOT NULL,
  url TEXT UNIQUE NOT NULL,
  source TEXT,
  published_at TIMESTAMP,
  region TEXT,
  topic TEXT,
  content TEXT,
  cleaned TEXT,
  tags JSONB,                        -- List: ["GEO:China", "SEC:Surveillance", ...]
  entities JSONB,                    -- List: ["China", "Naval Operations", ...]
  cluster_id TEXT,                   -- Can be numeric or string label
  confidence_score FLOAT,
  embedding_dimensions INT,
  processed_by TEXT,
  inserted_at TIMESTAMP DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX idx_articles_region ON articles(region);
CREATE INDEX idx_articles_topic ON articles(topic);
CREATE INDEX idx_articles_cluster_id ON articles(cluster_id);
CREATE INDEX idx_articles_inserted_at ON articles(inserted_at);
CREATE INDEX idx_articles_published_at ON articles(published_at);

-- Create index on tags JSONB field
CREATE INDEX idx_articles_tags ON articles USING GIN(tags);
CREATE INDEX idx_articles_entities ON articles USING GIN(entities);

-- Enable RLS (Row Level Security)
ALTER TABLE articles ENABLE ROW LEVEL SECURITY;

-- Create policy for public access (adjust as needed for production)
CREATE POLICY "Allow all access to articles" ON articles
    FOR ALL USING (true); 