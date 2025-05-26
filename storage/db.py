from supabase import create_client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def insert_osint_entry(source_url: str, content: str):
    """Insert a new OSINT entry into the database."""
    return supabase.table("osint_raw_data").insert({
        "source_url": source_url,
        "content": content
    }).execute()

def get_unprocessed_osint_entries(limit: int = 10):
    """Get a list of OSINT entries that haven't been tagged yet."""
    return supabase.table("osint_raw_data") \
        .select("*") \
        .filter("tags", "is", "null") \
        .order("ingested_at", {"ascending": False}) \
        .limit(limit).execute().data

def update_osint_tags(entry_id: str, tags: list, confidence: float):
    """Update the tags and confidence score for an OSINT entry."""
    return supabase.table("osint_raw_data").update({
        "tags": tags,
        "confidence_score": confidence,
        "manual_review": False
    }).eq("id", entry_id).execute()

def insert_prediction(osint_id, event_type, region, score, model):
    """Insert a new prediction based on an OSINT entry."""
    return supabase.table("predictions").insert({
        "osint_id": osint_id,
        "event_type": event_type,
        "region": region,
        "likelihood_score": score,
        "model_used": model
    }).execute()

def create_tables():
    """Create necessary tables if they don't exist yet."""
    # Create osint_raw_data table
    supabase.table("osint_raw_data").create({
        "id": "uuid primary key default uuid_generate_v4()",
        "source_url": "text",
        "content": "text",
        "ingested_at": "timestamptz default now()",
        "tags": "text[]",
        "confidence_score": "float",
        "manual_review": "boolean default true"
    })
    
    # Create predictions table
    supabase.table("predictions").create({
        "id": "uuid primary key default uuid_generate_v4()",
        "osint_id": "uuid references osint_raw_data(id)",
        "event_type": "text",
        "region": "text",
        "likelihood_score": "float",
        "model_used": "text",
        "generated_at": "timestamptz default now()"
    }) 