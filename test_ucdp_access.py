import os
import requests
import json
import pandas as pd
import logging
import time

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def test_ucdp_api():
    """
    Test UCDP API access and data processing.
    """
    # Make sure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # API parameters
    resource = "gedevents"
    version = "24.1"
    pagesize = 10  # Small limit for testing
    
    logger.info(f"Testing UCDP API access ({resource}/{version})")
    
    base_url = f"https://ucdpapi.pcr.uu.se/api/{resource}/{version}"
    
    # Test first page
    url = f"{base_url}?page=1&pagesize={pagesize}"
    
    try:
        logger.info(f"Requesting data from {url}")
        response = requests.get(url, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            if "Result" in data and data["Result"]:
                logger.info(f"✅ SUCCESS: Retrieved {len(data['Result'])} records")
                
                # Check expected fields
                event = data["Result"][0]
                logger.info(f"First event fields: {list(event.keys())}")
                
                # Check for critical fields
                expected_fields = ["id", "year", "side_a", "side_b", "type_of_violence", "best", "region"]
                missing_fields = [field for field in expected_fields if field.lower() not in {k.lower(): v for k, v in event.items()}]
                
                if missing_fields:
                    logger.warning(f"Missing fields: {missing_fields}")
                else:
                    logger.info("✅ All essential fields present")
                
                # Test pagination
                logger.info("Testing pagination...")
                url_page2 = f"{base_url}?page=2&pagesize={pagesize}"
                
                try:
                    response_page2 = requests.get(url_page2, timeout=60)
                    
                    if response_page2.status_code == 200:
                        data_page2 = response_page2.json()
                        
                        if "Result" in data_page2 and data_page2["Result"]:
                            logger.info(f"✅ Pagination works: page 2 returned {len(data_page2['Result'])} records")
                            
                            # Test data processing
                            logger.info("Testing data processing logic...")
                            processed_data = []
                            
                            for event in data["Result"]:
                                # Extract relevant fields (normalize keys to lowercase)
                                event_lower = {k.lower(): v for k, v in event.items()}
                                
                                side_a = event_lower.get("side_a", "Unknown")
                                side_b = event_lower.get("side_b", "Unknown")
                                type_of_violence = event_lower.get("type_of_violence", "Unknown")
                                region = event_lower.get("region", "Unknown")
                                country = event_lower.get("country", "Unknown")
                                year = event_lower.get("year", "Unknown")
                                best = event_lower.get("best", 0)  # casualties
                                
                                # Construct synthetic text field
                                text = f"Type {type_of_violence} conflict between {side_a} and {side_b} in {country or region}"
                                
                                processed_data.append({
                                    "year": year,
                                    "text": text,
                                    "side_a": side_a,
                                    "side_b": side_b,
                                    "region": region,
                                    "country": country,
                                    "casualties": best,
                                    "label": 1,  # All UCDP events are conflicts
                                    "source": "ucdp_api"
                                })
                            
                            # Convert to DataFrame
                            df = pd.DataFrame(processed_data)
                            logger.info(f"Processed {len(df)} events")
                            
                            # Save sample to a temporary file
                            sample_file = "data/ucdp_sample.json"
                            with open(sample_file, "w") as f:
                                json.dump(processed_data[:5], f, indent=2)
                            logger.info(f"Saved sample to {sample_file}")
                            
                            return True
                        else:
                            logger.warning("No data in page 2")
                            return True  # Still a success if first page worked
                    else:
                        logger.error(f"Error with pagination: {response_page2.status_code}")
                        return False
                        
                except Exception as e:
                    logger.error(f"Error testing pagination: {e}")
                    return False
            else:
                logger.error("No data in response")
                return False
        else:
            logger.error(f"Failed to fetch data: Status code {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Error testing UCDP API: {e}")
        return False

def test_ucdp_csv():
    """
    Test loading and processing UCDP CSV files.
    """
    # Directory for UCDP CSV files
    csv_dir = "data/ucdp/csv"
    os.makedirs(csv_dir, exist_ok=True)
    
    logger.info(f"Testing UCDP CSV processing from directory: {csv_dir}")
    
    # Check if directory contains any CSV files
    csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}. Skipping test.")
        return None
    
    logger.info(f"Found {len(csv_files)} CSV files")
    
    # Process the first file for testing
    test_file = os.path.join(csv_dir, csv_files[0])
    logger.info(f"Testing with file: {test_file}")
    
    try:
        # Try to read the CSV
        df = pd.read_csv(test_file, encoding='latin-1')
        logger.info(f"✅ Successfully read CSV with {len(df)} rows and {len(df.columns)} columns")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # Test data processing logic
        logger.info("Testing data normalization and text generation...")
        
        # Normalize column names
        df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
        
        # Define text field construction based on available columns
        if all(col in df.columns for col in ['side_a', 'side_b', 'region']):
            df['text'] = df.apply(
                lambda row: f"Conflict between {row['side_a']} and {row['side_b']} in {row['region']}",
                axis=1
            )
            logger.info("✅ Successfully created text field")
        else:
            logger.warning("Missing columns needed for text generation")
        
        # Set label to 1 for all rows (all UCDP data represents conflicts)
        df['label'] = 1
        df['source'] = 'ucdp_csv'
        
        # Sample rows
        logger.info(f"First row text example: {df['text'].iloc[0] if 'text' in df.columns else 'N/A'}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error processing CSV file: {e}")
        return False

if __name__ == "__main__":
    logger.info("Testing UCDP data access...")
    
    # Test UCDP API
    if test_ucdp_api():
        logger.info("✅ UCDP API test successful")
    else:
        logger.error("❌ UCDP API test failed")
    
    # Test UCDP CSV processing
    csv_result = test_ucdp_csv()
    if csv_result is True:
        logger.info("✅ UCDP CSV processing test successful")
    elif csv_result is False:
        logger.error("❌ UCDP CSV processing test failed")
    # None result means no CSV files were found 