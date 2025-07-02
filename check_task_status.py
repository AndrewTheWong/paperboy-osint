#!/usr/bin/env python3
"""
Check task execution status for latest pipeline tests
"""

import redis
import json
import requests

def check_recent_task_execution():
    """Check if our recent pipeline tasks executed"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Latest test task IDs
        task_ids = [
            "7b828da0-0b1f-44c3-907f-0281400df97b",  # Single article
            "c6f973d4-8820-4936-9b42-6a73016e0f69",  # Batch
        ]
        
        print(f"🔍 Checking Recent Pipeline Tasks")
        print("=" * 50)
        
        for i, task_id in enumerate(task_ids, 1):
            task_key = f"celery-task-meta-{task_id}"
            task_type = "Single Article" if i == 1 else "Batch Processing"
            
            print(f"\n{i}. {task_type} Task: {task_id}")
            
            if r.exists(task_key):
                task_result = r.get(task_key)
                if task_result:
                    try:
                        result_data = json.loads(task_result.decode('utf-8'))
                        status = result_data.get('status', 'Unknown')
                        result = result_data.get('result', 'None')
                        
                        print(f"   📋 Status: {status}")
                        if status == 'SUCCESS':
                            print(f"   ✅ Result: {result}")
                        elif status == 'FAILURE':
                            print(f"   ❌ Error: {result}")
                            if 'traceback' in result_data:
                                print(f"   📜 Traceback: {result_data['traceback'][:200]}...")
                        else:
                            print(f"   🔄 Result: {result}")
                            
                    except json.JSONDecodeError:
                        print(f"   📋 Raw data: {task_result[:100]}...")
            else:
                print(f"   ❌ Task metadata not found!")
        
        # Check current pipeline status
        print(f"\n📊 Current Pipeline Status")
        print("=" * 30)
        try:
            response = requests.get("http://localhost:8000/ingest/status", timeout=10)
            if response.status_code == 200:
                status = response.json()
                total = status.get('total_articles', 0)
                processed = status.get('processed_articles', 0)
                unprocessed = status.get('unprocessed_articles', 0)
                
                print(f"   Total Articles: {total}")
                print(f"   Processed: {processed}")
                print(f"   Unprocessed: {unprocessed}")
                print(f"   Status: {status.get('status', 'Unknown')}")
                
                if processed > 0:
                    print(f"   🎉 Articles are being processed successfully!")
                else:
                    print(f"   ⏳ No articles processed yet")
                    
            else:
                print(f"   ❌ API Error: {response.status_code}")
        except Exception as e:
            print(f"   ❌ Connection Error: {e}")
        
        # Check for any active tasks
        print(f"\n🔄 Queue Status")
        print("=" * 15)
        queues_to_check = ['celery', 'preprocessing', 'clustering_queue', 'summarization']
        
        all_empty = True
        for queue_name in queues_to_check:
            length = r.llen(queue_name)
            if length > 0:
                print(f"   📋 {queue_name}: {length} items")
                all_empty = False
        
        if all_empty:
            print(f"   ✅ All queues are empty")
        
        # Get very recent task statuses
        print(f"\n📈 Recent Task Activity (Last 10)")
        print("=" * 35)
        
        all_task_keys = r.keys('celery-task-meta-*')
        recent_tasks = sorted(all_task_keys, reverse=True)[:10]
        
        success_count = 0
        failure_count = 0
        pending_count = 0
        
        for task_key in recent_tasks:
            task_id = task_key.decode('utf-8').replace('celery-task-meta-', '')
            task_data = r.get(task_key)
            if task_data:
                try:
                    data = json.loads(task_data.decode('utf-8'))
                    status = data.get('status', 'Unknown')
                    if status == 'SUCCESS':
                        success_count += 1
                    elif status == 'FAILURE':
                        failure_count += 1
                    else:
                        pending_count += 1
                except:
                    pass
        
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failures: {failure_count}")  
        print(f"   🔄 Pending: {pending_count}")
        
    except Exception as e:
        print(f"❌ Error checking task status: {e}")

if __name__ == "__main__":
    check_recent_task_execution() 