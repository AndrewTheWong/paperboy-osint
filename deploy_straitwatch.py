#!/usr/bin/env python3
"""
StraitWatch Deployment Script

This script sets up and deploys the complete StraitWatch system.
"""

import os
import sys
import subprocess
import asyncio
import json
from pathlib import Path
from datetime import datetime

def setup_environment():
    """Setup environment variables and configuration"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    dirs_to_create = [
        "data/time_series",
        "models/ner",
        "models/events", 
        "models/forecasting",
        "reports",
        "logs"
    ]
    
    for dir_path in dirs_to_create:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Created directory: {dir_path}")
    
    # Check for required environment variables
    required_env_vars = ["SUPABASE_URL", "SUPABASE_KEY"]
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {missing_vars}")
        print("Please set these in your .env file or environment")
        return False
    
    print("✓ Environment setup complete")
    return True

def install_dependencies():
    """Install required Python packages"""
    print("📦 Installing dependencies...")
    
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                      check=True, capture_output=True)
        print("✓ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_database():
    """Setup database schema"""
    print("🗄️  Setting up database...")
    
    try:
        from utils.supabase_client import get_supabase
        supabase = get_supabase()
        
        # Test connection
        result = supabase.table("articles").select("count").limit(1).execute()
        print("✓ Database connection successful")
        
        # Create tables if they don't exist (schema should be applied manually)
        print("✓ Database schema ready")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        print("Please ensure your Supabase database is running and accessible")
        return False

def create_config_files():
    """Create default configuration files"""
    print("⚙️  Creating configuration files...")
    
    # Sources configuration
    sources_config = {
        "rss_feeds": [
            "https://feeds.reuters.com/reuters/world",
            "https://rss.cnn.com/rss/edition.world.rss",
            "https://feeds.bbci.co.uk/news/world/rss.xml"
        ],
        "taiwan_sources": [
            "https://focustaiwan.tw/rss/news.xml",
            "https://www.taipeitimes.com/xml/news.rss"
        ],
        "keywords": [
            "taiwan strait", "china taiwan", "taiwan military",
            "south china sea", "indo-pacific", "strait of taiwan"
        ]
    }
    
    # Create config directory
    Path("config").mkdir(exist_ok=True)
    
    with open("config/sources_config.json", "w") as f:
        json.dump(sources_config, f, indent=2)
    
    print("✓ Configuration files created")

async def test_agents():
    """Test that agents can be imported and initialized"""
    print("🧪 Testing agents...")
    
    try:
        from agents.orchestrator import OrchestratorAgent
        
        orchestrator = OrchestratorAgent()
        print(f"✓ Orchestrator initialized with {len(orchestrator.agents)} agents")
        
        # Test health checks
        result = await orchestrator.run()
        print(f"✓ System health check: {result}")
        
        return True
        
    except Exception as e:
        print(f"❌ Agent testing failed: {e}")
        return False

def create_systemd_service():
    """Create systemd service for production deployment"""
    service_content = f"""[Unit]
Description=StraitWatch OSINT System
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'straitwatch')}
WorkingDirectory={os.getcwd()}
Environment=PATH={os.getcwd()}/venv/bin
ExecStart={sys.executable} agents/orchestrator.py --mode schedule
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
    
    with open("straitwatch.service", "w") as f:
        f.write(service_content)
    
    print("✓ Systemd service file created: straitwatch.service")
    print("  To install: sudo cp straitwatch.service /etc/systemd/system/")
    print("  To enable: sudo systemctl enable straitwatch")
    print("  To start: sudo systemctl start straitwatch")

async def main():
    """Main deployment function"""
    print("🛰️  StraitWatch Deployment Starting...")
    print("=" * 50)
    
    # Step 1: Environment setup
    if not setup_environment():
        sys.exit(1)
    
    # Step 2: Install dependencies
    if not install_dependencies():
        sys.exit(1)
    
    # Step 3: Database setup
    if not setup_database():
        sys.exit(1)
    
    # Step 4: Create config files
    create_config_files()
    
    # Step 5: Test agents
    if not await test_agents():
        print("⚠️  Agent testing failed, but continuing deployment")
    
    # Step 6: Create systemd service
    create_systemd_service()
    
    print("=" * 50)
    print("🎉 StraitWatch deployment complete!")
    print()
    print("Next steps:")
    print("1. Review configuration files in config/")
    print("2. Test individual agents: python agents/orchestrator.py --mode run-once")
    print("3. Start the scheduler: python agents/orchestrator.py --mode schedule")
    print("4. Check system status: python agents/orchestrator.py --mode status")
    print()
    print("For production deployment:")
    print("5. Install systemd service (see instructions above)")
    print("6. Monitor logs in logs/")

if __name__ == "__main__":
    asyncio.run(main())