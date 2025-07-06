from supabase import create_client, Client
import os

def get_supabase() -> Client:
    """
    Get Supabase client for local development.
    Uses local Supabase instance running on default ports.
    """
    # Local Supabase configuration
    url = "http://127.0.0.1:54321"
    key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZS1kZW1vIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImV4cCI6MTk4MzgxMjk5Nn0.EGIM96RAZx35lJzdJsyH-qQwv8Hdp7fsn3W0YpN81IU"
    
    try:
        # Clear any proxy environment variables that might interfere
        proxy_vars = ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']
        for var in proxy_vars:
            if var in os.environ:
                del os.environ[var]
        
        client = create_client(url, key)
        return client
    except TypeError as e:
        if "proxy" in str(e):
            # Try alternative initialization without proxy
            print(f"Warning: Proxy configuration issue detected: {e}")
            print("Attempting alternative initialization...")
            try:
                # Import and use the client directly
                from supabase.client import Client as SupabaseClient
                client = SupabaseClient(url, key)
                return client
            except Exception as e2:
                print(f"Alternative initialization failed: {e2}")
                raise
        else:
            raise
    except Exception as e:
        print(f"Failed to create Supabase client: {e}")
        raise
        
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
    """Alias for get_supabase for backward compatibility"""
    return get_supabase() 