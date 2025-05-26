import os
import requests
import zipfile
import io
import pandas as pd
import logging
import datetime
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

def test_gdelt_download():
    """
    Test GDELT 2.0 data access using various URL patterns to determine which ones work.
    Focus specifically on GDELT 2.0 and not GDELT 1.0.
    """
    # Get the current date for testing
    today = datetime.datetime.now()
    
    # Test with a few recent dates (less likely to have missing data)
    test_dates = [
        today - datetime.timedelta(days=1),  # 1 day ago
        today - datetime.timedelta(days=2),  # 2 days ago
        today - datetime.timedelta(days=3),  # 3 days ago
    ]
    
    # For each date, test different time blocks
    all_times = []
    for base_date in test_dates:
        for hour in [0, 6, 12, 18]:
            for minute in [0, 15, 30, 45]:
                test_time = base_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                all_times.append(test_time)
    
    # Test only GDELT 2.0 URL patterns
    logger.info("Testing GDELT 2.0 last update files...")
    # These files contain URLs to the latest 15-minute update files
    lastupdate_urls = [
        "http://data.gdeltproject.org/gdeltv2/lastupdate.txt",
        "http://data.gdeltproject.org/gdeltv2/lastupdate-mentions.txt",
        "http://data.gdeltproject.org/gdeltv2/lastupdate-gkg.txt"
    ]
    
    # Test lastupdate URLs
    success_url = None
    for url in lastupdate_urls:
        if try_url(url):
            success_url = url
            # Parse the content to get the actual data file URLs
            logger.info(f"Parsing content from {url} to extract data URLs...")
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    for line in response.text.strip().split('\n'):
                        parts = line.split()
                        if len(parts) >= 3:
                            data_url = parts[2]
                            logger.info(f"Extracted data URL: {data_url}")
                            try_url(data_url)
                            
                            # Try to download and read the actual CSV data
                            try:
                                logger.info(f"Attempting to download and parse data from {data_url}...")
                                data_response = requests.get(data_url, timeout=30)
                                if data_response.status_code == 200:
                                    try:
                                        with zipfile.ZipFile(io.BytesIO(data_response.content)) as zf:
                                            csv_file = None
                                            for filename in zf.namelist():
                                                if filename.endswith('.CSV') or filename.endswith('.csv'):
                                                    csv_file = filename
                                                    break
                                            
                                            if csv_file:
                                                logger.info(f"Reading CSV file: {csv_file}")
                                                with zf.open(csv_file) as f:
                                                    # Read data using pandas
                                                    df = pd.read_csv(
                                                        f, sep="\t", header=None, 
                                                        on_bad_lines='skip', nrows=5
                                                    )
                                                    logger.info(f"Successfully parsed CSV with {len(df.columns)} columns")
                                                    logger.info(f"First few columns: {df.iloc[0, :5].tolist()}")
                                                    return success_url, df.shape[1]
                                    except zipfile.BadZipFile:
                                        logger.error(f"Bad zip file: {data_url}")
                                    except Exception as e:
                                        logger.error(f"Error parsing CSV: {e}")
                            except Exception as e:
                                logger.error(f"Error downloading data file: {e}")
            except Exception as e:
                logger.error(f"Error parsing lastupdate content: {e}")
    
    # If lastupdate parsing fails, try specific filename patterns directly
    if not success_url:
        logger.info("Testing specific GDELT 2.0 update files for various times...")
        for test_time in all_times[:10]:  # Test only the first 10 times to keep runtime reasonable
            # Format for GDELT 2.0 events
            formatted_time = test_time.strftime('%Y%m%d%H%M%S')
            url = f"http://data.gdeltproject.org/gdeltv2/{formatted_time}.export.CSV.zip"
            
            if try_url(url):
                success_url = url
                # Try to download and verify column count
                try:
                    logger.info(f"Attempting to download and parse data from {url}...")
                    response = requests.get(url, timeout=30)
                    if response.status_code == 200:
                        try:
                            with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
                                csv_file = None
                                for filename in zf.namelist():
                                    if filename.endswith('.CSV') or filename.endswith('.csv'):
                                        csv_file = filename
                                        break
                                
                                if csv_file:
                                    logger.info(f"Reading CSV file: {csv_file}")
                                    with zf.open(csv_file) as f:
                                        # Read data using pandas
                                        df = pd.read_csv(
                                            f, sep="\t", header=None, 
                                            on_bad_lines='skip', nrows=5
                                        )
                                        logger.info(f"Successfully parsed CSV with {len(df.columns)} columns")
                                        logger.info(f"First few columns: {df.iloc[0, :5].tolist()}")
                                        return success_url, df.shape[1]
                        except zipfile.BadZipFile:
                            logger.error(f"Bad zip file: {url}")
                        except Exception as e:
                            logger.error(f"Error parsing CSV: {e}")
                except Exception as e:
                    logger.error(f"Error downloading data file: {e}")
    
    return None, 0

def try_url(url):
    """Try downloading from a URL and report if it works."""
    try:
        logger.info(f"Trying URL: {url}")
        response = requests.head(url, timeout=10)
        
        if response.status_code == 200:
            logger.info(f"✅ SUCCESS: {url} (Status code: {response.status_code})")
            # For successful URLs, try to get some data to verify content-type
            if "lastupdate" in url:
                text_response = requests.get(url, timeout=10)
                logger.info(f"Content: {text_response.text[:200].strip()}")
            return True
        else:
            logger.warning(f"❌ FAILED: {url} (Status code: {response.status_code})")
            return False
    except Exception as e:
        logger.error(f"❌ ERROR: {url} - {str(e)}")
        return False

if __name__ == "__main__":
    logger.info("Testing GDELT 2.0 URL patterns...")
    success_url, column_count = test_gdelt_download()
    
    if success_url:
        logger.info(f"✅ SUCCESSFUL URL PATTERN: {success_url}")
        logger.info(f"✅ CSV COLUMN COUNT: {column_count}")
    else:
        logger.error("❌ No working GDELT 2.0 URL patterns found") 