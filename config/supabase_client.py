from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def get_supabase() -> Client:
    """
    Get Supabase client using environment variables only.
    Raises an error if not set.
    """
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in the environment.")
    # Clear any proxy environment variables that might interfere
    proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
    for var in proxy_vars:
        if var in os.environ:
            del os.environ[var]
    client = create_client(url, key)
    return client

def test_connection() -> bool:
    """
    Test the Supabase connection.
    """
    try:
        supabase = get_supabase()
        # Try to query a table to test connection
        result = supabase.table('articles').select('count').execute()
        print("Supabase connection successful!")
        return True
    except Exception as e:
        print(f"Supabase connection failed: {e}")
        return False

def get_supabase_client() -> Client:
    return get_supabase() 