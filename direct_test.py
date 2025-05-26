#!/usr/bin/env python3
"""
Test if python-dotenv is installed and if it can load the environment variables.
"""
import sys
import os

def test_dotenv():
    """Test if python-dotenv is installed"""
    print("Testing python-dotenv installation...")
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv is installed")
        
        # Try to load .env file
        env_file = os.path.join(os.path.dirname(__file__), '.env')
        if os.path.exists(env_file):
            load_dotenv(env_file)
            print(f"✅ .env file loaded from {env_file}")
            
            # Check for Supabase credentials
            supabase_url = os.getenv('SUPABASE_URL')
            supabase_key = os.getenv('SUPABASE_KEY')
            
            if supabase_url and supabase_key:
                print("✅ Supabase credentials found in .env file")
            else:
                print("❌ Supabase credentials not found in .env file")
        else:
            print(f"❌ .env file not found at {env_file}")
            
            # Check for template.env
            template_file = os.path.join(os.path.dirname(__file__), 'template.env')
            if os.path.exists(template_file):
                print(f"ℹ️ template.env file found at {template_file}")
                print("ℹ️ Run `python setup_environment.py` to set up .env file")
            else:
                print(f"❌ template.env file not found at {template_file}")
        
    except ImportError:
        print("❌ python-dotenv is not installed")
        print("ℹ️ Run `pip install python-dotenv` to install it")
        return False
    
    # Try to import other required packages
    required_packages = [
        'requests',
        'pandas',
        'streamlit',
        'numpy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} is installed")
        except ImportError:
            print(f"❌ {package} is not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n❌ Missing packages: {', '.join(missing_packages)}")
        print("ℹ️ Run `python setup_environment.py` to install all required packages")
        return False
    
    print("\n✅ All basic dependencies are installed")
    return True

if __name__ == "__main__":
    success = test_dotenv()
    sys.exit(0 if success else 1)
