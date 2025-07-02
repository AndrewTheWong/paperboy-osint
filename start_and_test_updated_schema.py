#!/usr/bin/env python3

"""
Start StraitWatch services and test the updated schema pipeline
"""

import subprocess
import time
import sys
import signal
import os
from typing import List

def run_command_background(command: str, name: str) -> subprocess.Popen:
    """Run a command in the background"""
    print(f"🚀 Starting {name}...")
    
    if sys.platform == "win32":
        # Windows
        process = subprocess.Popen(
            command,
            shell=True,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
        )
    else:
        # Unix/Linux/Mac
        process = subprocess.Popen(
            command,
            shell=True,
            preexec_fn=os.setsid
        )
    
    print(f"✅ {name} started (PID: {process.pid})")
    return process

def check_service_health() -> bool:
    """Check if services are healthy"""
    import requests
    
    try:
        # Check API
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code != 200:
            return False
        
        # Check Redis
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        
        return True
        
    except Exception:
        return False

def main():
    """Main startup and test routine"""
    print("🔧 StraitWatch Updated Schema Test Runner")
    print("=" * 50)
    
    processes: List[subprocess.Popen] = []
    
    try:
        # Start FastAPI server
        api_process = run_command_background(
            "python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload",
            "FastAPI Server"
        )
        processes.append(api_process)
        
        time.sleep(5)  # Let API start
        
        # Start Celery worker with Windows-compatible solo pool
        celery_process = run_command_background(
            "celery -A app.celery_worker worker --loglevel=info --pool=solo",
            "Celery Worker (Solo Pool)"
        )
        processes.append(celery_process)
        
        time.sleep(10)  # Let Celery start and load models
        
        # Check service health
        print("\n🔍 Checking service health...")
        if check_service_health():
            print("✅ All services are healthy")
        else:
            print("❌ Service health check failed")
            return
        
        print("\n📋 Services running:")
        print("   - FastAPI: http://localhost:8000")
        print("   - Celery Worker: Solo pool (Windows compatible)")
        print("   - Redis: localhost:6379")
        
        # Run the updated schema test
        print("\n🧪 Running updated schema test...")
        try:
            test_result = subprocess.run([
                sys.executable, "test_updated_schema.py"
            ], timeout=120, capture_output=False)
            
            if test_result.returncode == 0:
                print("\n✅ Schema test completed successfully!")
            else:
                print("\n⚠️ Schema test completed with issues")
                
        except subprocess.TimeoutExpired:
            print("\n⏰ Schema test timed out")
        except Exception as e:
            print(f"\n❌ Schema test error: {e}")
        
        # Offer to run stress test
        print("\n🤔 Would you like to run the stress test with real articles?")
        print("Options:")
        print("1. Run stress test")
        print("2. Keep services running for manual testing")
        print("3. Shutdown")
        
        choice = input("Enter choice (1-3): ").strip()
        
        if choice == "1":
            print("\n🚀 Running stress test...")
            try:
                subprocess.run([sys.executable, "stress_test_real_articles.py"], timeout=300)
            except subprocess.TimeoutExpired:
                print("⏰ Stress test timed out")
            except Exception as e:
                print(f"❌ Stress test error: {e}")
                
        elif choice == "2":
            print("\n🌐 Services are running!")
            print("💡 Test endpoints:")
            print("   - POST http://localhost:8000/ingest/v2/ (single article)")
            print("   - POST http://localhost:8000/ingest/v2/batch-optimized/ (batch)")
            print("   - GET http://localhost:8000/ingest/status (status)")
            print("   - GET http://localhost:8000/docs (API docs)")
            print("\n⏸️  Press Ctrl+C to shutdown...")
            
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutdown requested...")
        
    except KeyboardInterrupt:
        print("\n🛑 Interrupted by user...")
        
    finally:
        # Cleanup processes
        print("\n🧹 Shutting down services...")
        for i, process in enumerate(processes):
            try:
                if sys.platform == "win32":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    
                print(f"🔄 Terminating process {i}...")
                process.wait(timeout=10)
                
            except subprocess.TimeoutExpired:
                print(f"⚡ Force killing process {i}...")
                process.kill()
            except Exception as e:
                print(f"⚠️ Error terminating process {i}: {e}")
        
        print("✅ All services stopped")

if __name__ == "__main__":
    main() 