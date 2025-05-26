# test_supabase.py

import os
from pathlib import Path
from dotenv import load_dotenv
from supabase import create_client

# === Load environment variables from .env ===
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

if not SUPABASE_URL or not SUPABASE_KEY:
    print("❌ ERROR: Environment variables not loaded.")
    print(f"SUPABASE_URL: {SUPABASE_URL}")
    print(f"SUPABASE_KEY: {SUPABASE_KEY}")
    exit(1)

# === Create Supabase client ===
try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase client initialized.")
except Exception as e:
    print(f"❌ Failed to create Supabase client: {e}")
    exit(1)

# === Try querying a table ===
try:
    response = supabase.table("osint_raw_data").select("*").limit(1).execute()
    print("✅ Supabase query succeeded.")
    print("Data:", response.data)
except Exception as e:
    print(f"❌ Failed to query Supabase: {e}")
