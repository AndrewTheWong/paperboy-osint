#!/usr/bin/env python3
"""
setup_environment.py - Set up the development and testing environment for Paperboy

This script:
1. Installs all required dependencies from requirements.txt
2. Creates a template .env file if one doesn't exist
3. Sets up the necessary directories for testing
"""
import os
import sys
import subprocess
import shutil
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('setup_environment')

def install_dependencies():
    """Install the required dependencies from requirements.txt"""
    logger.info("Installing dependencies from requirements.txt...")
    
    try:
        # Use --no-input flag to avoid prompts
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            capture_output=True,
            text=True
        )
        logger.info("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to install dependencies: {e}")
        logger.error(f"Error details: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"❌ Error installing dependencies: {str(e)}")
        return False

def setup_env_file():
    """Create a .env file from template if it doesn't exist"""
    logger.info("Setting up .env file...")
    
    env_file = Path('.env')
    template_file = Path('template.env')
    
    if env_file.exists():
        logger.info(".env file already exists, skipping")
        return True
    
    if not template_file.exists():
        logger.error("❌ template.env file not found")
        return False
    
    try:
        shutil.copy(template_file, env_file)
        logger.info("✅ Created .env file from template")
        logger.info("⚠️ Please edit .env file with your actual credentials")
        return True
    except Exception as e:
        logger.error(f"❌ Error creating .env file: {str(e)}")
        return False

def create_directories():
    """Create necessary directories for testing"""
    logger.info("Creating necessary directories...")
    
    directories = [
        "data",
        "logs",
        "tests",
        "data/model_ready",
        "data/ucdp/csv"
    ]
    
    for directory in directories:
        dir_path = Path(directory)
        if not dir_path.exists():
            try:
                dir_path.mkdir(parents=True)
                logger.info(f"Created directory: {dir_path}")
            except Exception as e:
                logger.error(f"❌ Failed to create directory {dir_path}: {str(e)}")
    
    # Create __init__.py in tests directory if it doesn't exist
    tests_init = Path("tests/__init__.py")
    if not tests_init.exists():
        try:
            with open(tests_init, 'w') as f:
                f.write("# Tests package\n")
            logger.info("Created tests/__init__.py")
        except Exception as e:
            logger.error(f"❌ Failed to create tests/__init__.py: {str(e)}")
    
    logger.info("✅ Directory setup complete")
    return True

def main():
    """Run the setup process"""
    logger.info("Starting Paperboy environment setup")
    
    # Create directories first
    create_directories()
    
    # Set up .env file
    setup_env_file()
    
    # Install dependencies last
    install_success = install_dependencies()
    
    if install_success:
        logger.info("\n✅ Environment setup complete!")
        logger.info("You can now run tests with: python run_all_tests.py")
    else:
        logger.error("\n⚠️ Environment setup completed with errors")
        logger.error("Please check the logs and resolve issues before running tests")
    
    return 0 if install_success else 1

if __name__ == "__main__":
    sys.exit(main()) 