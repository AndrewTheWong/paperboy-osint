import os
import requests
import json
import pandas as pd
import logging
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
ACLED_API_KEY = os.getenv("ACLED_API_KEY")
ACLED_EMAIL = os.getenv("ACLED_EMAIL")

def test_acled_api():
    """
    Test ACLED API access with proper authentication and pagination.
    """
    if not ACLED_API_KEY:
        logger.error("ACLED_API_KEY not found in .env file")
        return False
        
    if not ACLED_EMAIL:
        logger.error("ACLED_EMAIL not found in .env file")
        return False
    
    logger.info(f"Testing ACLED API access with key: {ACLED_API_KEY[:4]}*** and email: {ACLED_EMAIL}")
    
    # First test - direct API status check
    status_url = "https://api.acleddata.com/version"
    try:
        logger.info(f"Testing ACLED API health status...")
        response = requests.get(status_url, timeout=30)
        if response.status_code == 200:
            logger.info(f"ACLED API status: OK ({response.status_code})")
            status_data = response.json()
            logger.info(f"API version info: {status_data}")
        else:
            logger.error(f"ACLED API status check failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error checking ACLED API status: {e}")
        return False
    
    # Second test - authentication 
    logger.info("Testing ACLED authentication...")
    auth_test_url = "https://api.acleddata.com/acled/read"
    params = {
        "key": ACLED_API_KEY,
        "email": ACLED_EMAIL,
        "page": 1,
        "limit": 1
    }
    
    try:
        response = requests.get(auth_test_url, params=params, timeout=30)
        if response.status_code == 200:
            logger.info(f"Authentication successful (status code {response.status_code})")
        else:
            logger.error(f"Authentication failed: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error testing authentication: {e}")
        return False
    
    # Test with longer date range periods and multiple regions
    # Try past 10 years
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365*10)  # 10 years ago
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    regions = ["Africa", "Middle East", "Europe", "Asia"]
    found_data = False
    
    for region in regions:
        logger.info(f"Testing ACLED API for region: {region} from {start_date_str} to {end_date_str}")
        
        params = {
            "key": ACLED_API_KEY,
            "email": ACLED_EMAIL,
            "page": 1,
            "limit": 10,
            "format": "json",
            "event_date": f"{start_date_str}|{end_date_str}",
            "region": region
        }
        
        try:
            response = requests.get(auth_test_url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]:
                    events_count = len(data["data"])
                    logger.info(f"✅ SUCCESS: Found {events_count} events in {region}")
                    
                    # Print sample event
                    if events_count > 0:
                        sample_event = data["data"][0]
                        event_date = sample_event.get("event_date", "Unknown")
                        event_type = sample_event.get("event_type", "Unknown")
                        actor1 = sample_event.get("actor1", "Unknown")
                        actor2 = sample_event.get("actor2", "Unknown")
                        location = sample_event.get("location", "Unknown")
                        
                        logger.info(f"Sample event: {event_date}, {event_type} between {actor1} and {actor2} in {location}")
                        found_data = True
                        
                        # Test pagination - try page 2
                        params["page"] = 2
                        page2_response = requests.get(auth_test_url, params=params, timeout=60)
                        if page2_response.status_code == 200:
                            page2_data = page2_response.json()
                            if "data" in page2_data and page2_data["data"]:
                                logger.info(f"✅ Pagination works: page 2 returned {len(page2_data['data'])} events")
                            else:
                                logger.info("No data on page 2")
                        break
                    
                else:
                    logger.warning(f"No events found in {region}")
            else:
                logger.error(f"Failed to query {region}: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error querying {region}: {e}")
    
    if not found_data:
        logger.error("Failed to find ACLED data in any region")
        
        # As a last resort, try without region filter
        logger.info("Attempting query without region filter...")
        params = {
            "key": ACLED_API_KEY,
            "email": ACLED_EMAIL,
            "page": 1,
            "limit": 10,
            "format": "json",
            "event_date": f"{start_date_str}|{end_date_str}"
        }
        
        try:
            response = requests.get(auth_test_url, params=params, timeout=60)
            if response.status_code == 200:
                data = response.json()
                if "data" in data and data["data"]:
                    events_count = len(data["data"])
                    logger.info(f"✅ SUCCESS: Found {events_count} events without region filter")
                    found_data = True
                else:
                    logger.error("No data found even without region filter")
            else:
                logger.error(f"Failed to query without region: {response.status_code} - {response.text}")
                
        except Exception as e:
            logger.error(f"Error querying without region: {e}")
            
    return found_data

if __name__ == "__main__":
    logger.info("Testing ACLED API access...")
    if test_acled_api():
        logger.info("✅ ACLED API test successful")
    else:
        logger.error("❌ ACLED API test failed")
        logger.error("❌ ACLED API test failed") 