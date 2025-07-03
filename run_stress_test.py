#!/usr/bin/env python3
"""
Non-interactive stress test runner
"""

import subprocess
import sys
import time

def run_stress_test():
    """Run stress test with medium configuration"""
    print("🚀 Running Stress Test (Medium Configuration)")
    print("=" * 50)
    
    # Start the stress test process
    process = subprocess.Popen(
        [sys.executable, "stress_test_real_articles.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Send choice "2" for medium test
    stdout, stderr = process.communicate(input="2\n")
    
    print("STDOUT:")
    print(stdout)
    
    if stderr:
        print("STDERR:")
        print(stderr)
    
    print(f"Exit code: {process.returncode}")

if __name__ == "__main__":
    run_stress_test() 