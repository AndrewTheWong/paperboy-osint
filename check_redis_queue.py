#!/usr/bin/env python3
"""
Check Redis queue status
"""

import redis
import json

def check_redis_queues():
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        
        # Check clustering queue
        queue_length = r.llen('clustering_queue')
        print(f"📊 Articles in clustering queue: {queue_length}")
        
        if queue_length > 0:
            print("📄 Sample articles in queue:")
            for i in range(min(3, queue_length)):
                item = r.lindex('clustering_queue', i)
                if item:
                    try:
                        # Try to decode as JSON if it's JSON data
                        if item.startswith(b'{'):
                            article_data = json.loads(item.decode('utf-8'))
                            print(f"  {i+1}. ID: {article_data.get('article_id', 'Unknown')}")
                        else:
                            print(f"  {i+1}. Raw: {item.decode('utf-8')[:100]}...")
                    except:
                        print(f"  {i+1}. Raw: {str(item)[:100]}...")
        
        # Check all keys in Redis
        all_keys = r.keys('*')
        print(f"\n🔑 All Redis keys: {len(all_keys)}")
        for key in all_keys:
            if isinstance(key, bytes):
                key = key.decode('utf-8')
            key_type = r.type(key).decode('utf-8')
            if key_type == 'list':
                length = r.llen(key)
                print(f"  📋 {key} (list): {length} items")
            elif key_type == 'string':
                print(f"  📄 {key} (string)")
            else:
                print(f"  ❓ {key} ({key_type})")
                
    except Exception as e:
        print(f"❌ Error connecting to Redis: {e}")

if __name__ == "__main__":
    check_redis_queues() 