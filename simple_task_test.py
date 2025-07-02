#!/usr/bin/env python3
"""
Simple test to verify Celery worker execution
"""

from app.tasks.preprocess import preprocess_and_enqueue
import time

def test_simple_execution():
    """Test if Celery worker can execute a simple task"""
    
    print("🔬 Simple Celery Task Execution Test")
    print("=" * 40)
    
    # Try executing task with .delay()
    print("📤 Submitting task with .delay()...")
    try:
        task = preprocess_and_enqueue.delay(
            'simple-test', 
            'Simple Test Title', 
            'Simple test body content',
            'Test Region',
            'Test Topic', 
            'http://simple.test'
        )
        print(f"✅ Task submitted: {task.id}")
        print(f"📊 Initial status: {task.status}")
        
        # Monitor task for 20 seconds
        for i in range(4):
            time.sleep(5)
            print(f"[{i+1}/4] Task status: {task.status}")
            if task.status == 'SUCCESS':
                print(f"🎉 Task completed successfully!")
                print(f"📋 Result: {task.result}")
                break
            elif task.status == 'FAILURE':
                print(f"❌ Task failed!")
                print(f"📋 Error: {task.result}")
                break
        else:
            print("⏰ Task did not complete within 20 seconds")
            
    except Exception as e:
        print(f"❌ Error submitting task: {e}")
    
    print("\n🏁 Simple test complete")

if __name__ == "__main__":
    test_simple_execution() 