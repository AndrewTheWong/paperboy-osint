#!/usr/bin/env python3
"""
Daily Digest Runner

This script generates a daily digest of completed cluster summaries,
grouped by region and topic, and outputs it to console or stores it.
"""

import os
from datetime import datetime
from dotenv import load_dotenv
from app.pipelines.daily_digest import generate_daily_digest
from app.utils.formatter import format_digest


def main():
    """Generate and output daily digest"""
    # Load environment variables
    load_dotenv()
    
    print("=" * 60)
    print("DAILY DIGEST GENERATOR")
    print("=" * 60)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Generate the daily digest
        print("Generating daily digest...")
        digest = generate_daily_digest()
        
        if not digest:
            print("No clusters found for today's digest.")
            return
        
        # Output to console
        print("\n" + "=" * 60)
        print("DAILY DIGEST OUTPUT")
        print("=" * 60)
        print(digest)
        
        # Optionally save to file
        today = datetime.now().strftime('%Y-%m-%d')
        with open(f'daily_digest_{today}.md', 'w', encoding='utf-8') as f:
            f.write(digest)
        print(f'\nDigest saved to daily_digest_{today}.md')
        
        # Optionally store to Supabase Storage (if implemented)
        store_to_supabase = input("Store to Supabase Storage? (y/n): ").lower().strip()
        if store_to_supabase == 'y':
            print("Supabase Storage integration not yet implemented.")
            # TODO: Implement Supabase Storage upload
            # storage_path = f"digests/daily-{datetime.now().strftime('%Y-%m-%d')}.txt"
            # upload_to_supabase_storage(storage_path, digest)
        
        print("\nDaily digest generation completed successfully!")
        
    except Exception as e:
        print(f"Error generating daily digest: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 