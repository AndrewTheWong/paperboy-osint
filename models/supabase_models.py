from sqlalchemy import Column, String, Text, DateTime, JSON, ForeignKey, Float, Integer
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.ext.declarative import declarative_base
import datetime
import uuid

Base = declarative_base()

class Article(Base):
    """Raw scraped articles table - stores only unprocessed data"""
    __tablename__ = 'articles'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    source_url = Column(Text, nullable=False)
    title = Column(Text)
    raw_html = Column(Text)
    text_content = Column(Text, nullable=False)
    language = Column(String(10))
    published_date = Column(DateTime)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)

class ArticleProcessed(Base):
    """Processed articles table - stores enrichment data linked to raw articles"""
    __tablename__ = 'articles_processed'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    article_id = Column(UUID(as_uuid=True), ForeignKey('articles.id'), unique=True, nullable=False)
    
    # Translation data
    translated_text = Column(Text)
    translated_title = Column(Text)
    source_language = Column(String(10))
    target_language = Column(String(10))
    
    # Embedding data
    embedding = Column(ARRAY(Float))
    
    # Tagging and NER data
    tags = Column(JSON)
    entities = Column(JSON)
    tag_categories = Column(JSON)
    
    # Clustering data
    cluster_id = Column(UUID(as_uuid=True), ForeignKey('clusters.id'))
    
    # Summary data
    summary = Column(Text)
    
    # Processing metadata
    processing_status = Column(String(50), default='pending')  # pending, processing, complete, failed
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)

class Cluster(Base):
    """Clusters table - stores clustering metadata and summaries"""
    __tablename__ = 'clusters'

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow)
    
    # Cluster metadata
    label = Column(String(255))
    description = Column(Text)
    center_embedding = Column(ARRAY(Float))
    
    # Cluster statistics
    member_count = Column(Integer, default=0)
    top_tags = Column(JSON)
    top_entities = Column(JSON)
    
    # Cluster summary
    summary = Column(Text)
    representative_article_id = Column(UUID(as_uuid=True), ForeignKey('articles_processed.id'))
    
    # Processing status
    status = Column(String(50), default='pending')  # pending, processing, complete, failed 