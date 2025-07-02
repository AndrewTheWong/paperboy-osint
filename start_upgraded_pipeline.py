#!/usr/bin/env python3
"""
🚀 Startup Script for Upgraded StraitWatch Pipeline
Launches FastAPI server and Celery worker with Windows compatibility
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def start_api_server():
    """Start the FastAPI server"""
    print("🌐 Starting FastAPI server...")
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app", 
        "--host", "0.0.0.0", 
        "--port", "8000", 
        "--reload"
    ]
    
    return subprocess.Popen(cmd, cwd=Path.cwd())

def start_celery_worker():
    """Start Celery worker with Windows-compatible solo pool"""
    print("⚙️  Starting Celery worker (Windows Solo Pool)...")
    cmd = [
        "celery", "-A", "app.celery_worker", "worker",
        "--loglevel=info",
        "--pool=solo"  # Windows compatibility
    ]
    
    return subprocess.Popen(cmd, cwd=Path.cwd())

def main():
    """Start both API server and Celery worker"""
    print("🚀 Starting StraitWatch Upgraded Pipeline")
    print("=" * 60)
    print("Pipeline: [Preprocess] → [NER Tag] → [Embed+Cluster] → [Store to Supabase]")
    print()
    
    # Check if Redis is available
    print("🔍 Checking Redis connection...")
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("💡 Make sure Redis is running: redis-server")
        return
    
    # Check if Supabase is available
    print("🔍 Checking Supabase connection...")
    try:
        from utils.supabase_client import get_supabase_client
        supabase = get_supabase_client()
        # Simple health check
        supabase.table('articles').select('id').limit(1).execute()
        print("✅ Supabase connection successful")
    except Exception as e:
        print(f"❌ Supabase connection failed: {e}")
        print("💡 Make sure Supabase is running: supabase start")
        return
    
    processes = []
    
    try:
        # Start API server
        api_process = start_api_server()
        processes.append(api_process)
        time.sleep(3)  # Give API time to start
        
        # Start Celery worker
        worker_process = start_celery_worker()
        processes.append(worker_process)
        time.sleep(2)
        
        print("\n✅ All services started successfully!")
        print("\n🌐 API Endpoints:")
        print("   - Health: http://localhost:8000/health")
        print("   - Legacy Pipeline: POST http://localhost:8000/ingest/")
        print("   - Upgraded Pipeline: POST http://localhost:8000/ingest/v2/")
        print("   - Batch Pipeline: POST http://localhost:8000/ingest/v2/batch/")
        print("   - Status: GET http://localhost:8000/ingest/status")
        print("   - Docs: http://localhost:8000/docs")
        
        print("\n🧪 Test the pipeline:")
        print("   python test_upgraded_pipeline.py")
        print("   python pipeline_main.py")
        
        print("\n💡 Press Ctrl+C to stop all services")
        
        # Wait for processes
        while True:
            time.sleep(1)
            # Check if any process has died
            for i, process in enumerate(processes):
                if process.poll() is not None:
                    print(f"⚠️  Process {i} has terminated with code {process.returncode}")
                    
    except KeyboardInterrupt:
        print("\n🛑 Shutting down services...")
        
    finally:
        # Cleanup processes
        for i, process in enumerate(processes):
            if process.poll() is None:  # Process is still running
                print(f"🔄 Terminating process {i}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(f"⚡ Force killing process {i}...")
                    process.kill()
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 