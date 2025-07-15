#!/usr/bin/env python3
"""Startup script for Paperboy pipeline"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_redis():
    """Check if Redis is running"""
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis is running")
        return True
    except:
        print("❌ Redis is not running")
        return False

def start_workers():
    """Start Celery workers"""
    print("🚀 Starting Celery workers...")
    
    workers = [
        ("Scraper", "scraper"),
        ("Translator", "translate"), 
        ("Tagger", "tag"),
        ("Embedder", "embed"),
        ("Clusterer", "cluster"),
        ("Orchestrator", "orchestrate")
    ]
    
    processes = []
    
    for name, queue in workers:
        cmd = [
            "celery", "-A", "celery_worker", "worker",
            "--loglevel=info", "--pool=solo",
            "-Q", queue, "-n", f"{name}@%h"
        ]
        
        try:
            process = subprocess.Popen(cmd)
            processes.append((name, process))
            print(f"✅ Started {name} worker (PID: {process.pid})")
        except Exception as e:
            print(f"❌ Failed to start {name} worker: {e}")
    
    return processes

def start_api():
    """Start FastAPI application"""
    print("🌐 Starting FastAPI application...")
    
    try:
        cmd = ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
        process = subprocess.Popen(cmd)
        print(f"✅ Started API server (PID: {process.pid})")
        return process
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        return None

def main():
    """Main startup function"""
    print("🎯 PAPERBOY PIPELINE STARTUP")
    print("=" * 40)
    
    # Check Redis
    if not check_redis():
        print("Please start Redis first: redis-server")
        return
    
    # Start workers
    worker_processes = start_workers()
    
    # Start API
    api_process = start_api()
    
    if worker_processes and api_process:
        print(f"\n✅ Started {len(worker_processes)} workers and API server")
        print("\n📊 Services running:")
        for name, process in worker_processes:
            print(f"  - {name} worker (PID: {process.pid})")
        print(f"  - API server (PID: {api_process.pid})")
        
        print("\n🌐 Access points:")
        print("  - API: http://localhost:8000")
        print("  - API docs: http://localhost:8000/docs")
        
        print("\n🛑 Press Ctrl+C to stop all services")
        
        try:
            # Keep running
            while True:
                time.sleep(1)
                
                # Check if any processes died
                for name, process in worker_processes[:]:
                    if process.poll() is not None:
                        print(f"⚠️ {name} worker stopped")
                        worker_processes.remove((name, process))
                
                if api_process and api_process.poll() is not None:
                    print("⚠️ API server stopped")
                    break
                    
        except KeyboardInterrupt:
            print("\n🛑 Stopping services...")
            
            # Stop workers
            for name, process in worker_processes:
                process.terminate()
                print(f"Stopped {name} worker")
            
            # Stop API
            if api_process:
                api_process.terminate()
                print("Stopped API server")
            
            print("✅ All services stopped")

if __name__ == "__main__":
    main() 