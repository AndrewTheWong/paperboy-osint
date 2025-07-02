#!/usr/bin/env python3
"""
Check the status of chained pipeline tasks
"""

import redis
import json

def check_chained_tasks():
    """Check if the chained pipeline tasks executed"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Chained task IDs from the successful main tasks
        chained_task_ids = [
            "a8fee25a-969b-4f5f-a273-193e8c504706",  # Single article chain
            "ef1e90ea-b0ea-4eb3-852b-0b8bc39393ed",  # Batch 1
            "b8b54a71-ae02-48a1-8d27-21d43f0bcf9a",  # Batch 2  
            "fffa1bb5-86cc-4a93-9e13-683aa50a043",   # Batch 3
        ]
        
        print(f"🔍 Checking Chained Pipeline Tasks")
        print("=" * 40)
        
        for i, task_id in enumerate(chained_task_ids, 1):
            task_key = f"celery-task-meta-{task_id}"
            
            if i == 1:
                task_type = "Single Article Chain"
            else:
                task_type = f"Batch Article {i-1}"
                
            print(f"\n{i}. {task_type}: {task_id}")
            
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
                                print(f"   📜 Traceback: {result_data['traceback'][:300]}...")
                        elif status == 'PENDING':
                            print(f"   ⏳ Task is still running...")
                        else:
                            print(f"   🔄 Status: {status}, Result: {result}")
                            
                    except json.JSONDecodeError:
                        print(f"   📋 Raw data: {task_result[:100]}...")
            else:
                print(f"   ❌ Task metadata not found (task may not have started)")
        
        # Check if any tasks are currently running
        print(f"\n🔄 Current Processing Status")
        print("=" * 30)
        
        # Look for any tasks with PENDING status
        all_task_keys = r.keys('celery-task-meta-*')
        
        pending_tasks = []
        for task_key in all_task_keys:
            task_data = r.get(task_key)
            if task_data:
                try:
                    data = json.loads(task_data.decode('utf-8'))
                    if data.get('status') == 'PENDING':
                        task_id = task_key.decode('utf-8').replace('celery-task-meta-', '')
                        pending_tasks.append(task_id)
                except:
                    pass
        
        if pending_tasks:
            print(f"   🔄 {len(pending_tasks)} tasks still pending:")
            for task_id in pending_tasks[:5]:  # Show first 5
                print(f"      - {task_id}")
        else:
            print(f"   ✅ No pending tasks found")
        
        # Summary
        print(f"\n📊 Summary")
        print("=" * 10)
        
        found_count = 0
        success_count = 0
        failure_count = 0
        pending_count = 0
        
        for task_id in chained_task_ids:
            task_key = f"celery-task-meta-{task_id}"
            if r.exists(task_key):
                found_count += 1
                task_data = r.get(task_key)
                if task_data:
                    try:
                        data = json.loads(task_data.decode('utf-8'))
                        status = data.get('status', 'Unknown')
                        if status == 'SUCCESS':
                            success_count += 1
                        elif status == 'FAILURE':
                            failure_count += 1
                        elif status == 'PENDING':
                            pending_count += 1
                    except:
                        pass
        
        print(f"   📋 Found: {found_count}/{len(chained_task_ids)} chained tasks")
        print(f"   ✅ Success: {success_count}")
        print(f"   ❌ Failures: {failure_count}")
        print(f"   ⏳ Pending: {pending_count}")
        print(f"   ❓ Missing: {len(chained_task_ids) - found_count}")
        
        if found_count == 0:
            print(f"\n⚠️  No chained tasks found - this suggests:")
            print(f"      1. Tasks haven't started yet (registration issue)")
            print(f"      2. Tasks completed so fast metadata expired")
            print(f"      3. Task registration failed in Celery worker")
        
    except Exception as e:
        print(f"❌ Error checking chained tasks: {e}")

if __name__ == "__main__":
    check_chained_tasks() 