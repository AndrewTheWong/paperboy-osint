import os
import requests
import zipfile
import io
import pandas as pd
import numpy as np
import json
import datetime
import argparse
import logging
import glob
from typing import Tuple, List, Dict, Optional
from dotenv import load_dotenv
from pathlib import Path
import time
import pickle

# Define tag vocabulary
TAG_VOCAB = [
    "military movement", "armed clash", "ceasefire", "civil war",
    "diplomatic meeting", "protest", "coup", "airstrike",
    "terror attack", "cross-border raid", "shelling", "civilian deaths",
    "ethnic violence", "insurgency", "mass killing"
]

def tag_encoder(text: str) -> list[int]:
    text = text.lower()
    return [1 if tag in text else 0 for tag in TAG_VOCAB]

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/data_fetch.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create necessary directories
os.makedirs("data", exist_ok=True)
os.makedirs("logs", exist_ok=True)
os.makedirs("data/ucdp/csv", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Load environment variables
load_dotenv()
ACLED_API_KEY = os.getenv("ACLED_API_KEY")
ACLED_EMAIL = os.getenv("ACLED_EMAIL")

def fetch_gdelt(sample_limit=50000, force=False, days_back=30):
    """
    Fetch GDELT 2.0 data using the lastupdate.txt approach to get the most recent data.
    Only uses GDELT 2.0 data (from 2014 onwards).
    
    Args:
        sample_limit: Target number of samples to collect
        force: Whether to force download even if data already exists
        days_back: Number of days to try fetching data for
    """
    output_file = "data/gdelt_events.csv"
    
    # Check if data already exists and force is not set
    if os.path.exists(output_file) and not force:
        logger.info(f"GDELT data already exists at {output_file}. Use --force to re-download.")
        try:
            return pd.read_csv(output_file)
        except pd.errors.EmptyDataError:
            logger.warning(f"Existing GDELT data at {output_file} is empty. Re-downloading.")
    
    logger.info(f"Fetching GDELT 2.0 data (target: {sample_limit} samples)...")
    
    all_gdelt_data = []
    total_samples = 0
    
    # Initialize collections to store data
    event_ids = set()  # Track event IDs to avoid duplicates
    
    # Try to fetch for multiple days to get enough data
    current_date = datetime.datetime.now()
    
    for day_offset in range(days_back):
        if total_samples >= sample_limit:
            break
            
        # Start with the lastupdate.txt file to get recent data
        lastupdate_url = "http://data.gdeltproject.org/gdeltv2/lastupdate.txt"
        logger.info(f"Checking for latest GDELT 2.0 updates from {lastupdate_url}")
        
        try:
            response = requests.get(lastupdate_url, timeout=30)
            
            if response.status_code == 200:
                lines = response.text.strip().split('\n')
                
                for line in lines:
                    parts = line.split()
                    if len(parts) >= 3:
                        data_url = parts[2]
                        
                        # Make sure it's a GDELT 2.0 export file
                        if "export.CSV.zip" in data_url:
                            logger.info(f"Downloading GDELT 2.0 data from {data_url}")
                            
                            try:
                                data_response = requests.get(data_url, timeout=60)
                                
                                if data_response.status_code == 200:
                                    # Extract and process the CSV file
                                    try:
                                        with zipfile.ZipFile(io.BytesIO(data_response.content)) as zip_file:
                                            for file_name in zip_file.namelist():
                                                if file_name.endswith('.CSV') or file_name.endswith('.csv'):
                                                    with zip_file.open(file_name) as f:
                                                        # GDELT 2.0 has 61 columns
                                                        cols = [
                                                            "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
                                                            "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
                                                            "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
                                                            "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
                                                            "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
                                                            "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
                                                            "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
                                                            "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
                                                            "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
                                                            "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
                                                            "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
                                                            "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
                                                            "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
                                                            "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type",
                                                            "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
                                                            "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
                                                            "DATEADDED", "SOURCEURL"
                                                        ]
                                                        
                                                        try:
                                                            # Read GDELT 2.0 data with 61 columns
                                                            df = pd.read_csv(
                                                                f, 
                                                                sep="\t", 
                                                                header=None,
                                                                names=cols,
                                                                usecols=["GLOBALEVENTID", "SQLDATE", "Actor1CountryCode", "Actor2CountryCode", "EventCode", "GoldsteinScale", "SOURCEURL"],
                                                                encoding="latin-1",
                                                                on_bad_lines='skip',
                                                                low_memory=False
                                                            )
                                                            
                                                            # Remove duplicates
                                                            if "GLOBALEVENTID" in df.columns:
                                                                df = df[~df["GLOBALEVENTID"].isin(event_ids)]
                                                                event_ids.update(df["GLOBALEVENTID"].tolist())
                                                            
                                                            # Skip if no data after filtering duplicates
                                                            if df.empty:
                                                                logger.info(f"  No new events in {file_name} after removing duplicates")
                                                                continue
                                                            
                                                            # Process data
                                                            df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
                                                            df["label"] = df["EventCode"].astype(str).str.startswith("19").astype(int)
                                                            df["actor1"] = df["Actor1CountryCode"].fillna("Unknown")
                                                            df["actor2"] = df["Actor2CountryCode"].fillna("Unknown")
                                                            
                                                            # Extract text for each event
                                                            df["text"] = "GDELT event between " + df["actor1"] + " and " + \
                                                                        df["actor2"] + " with code " + df["EventCode"].astype(str)
                                                            
                                                            # Add data source field
                                                            df["source"] = "gdelt"
                                                            
                                                            # Keep only needed columns
                                                            df = df[["date", "text", "actor1", "actor2", "label", "source", "GoldsteinScale"]]
                                                            
                                                            # Get balanced samples
                                                            conflict_events = df[df["label"] == 1]
                                                            non_conflict_events = df[df["label"] == 0]
                                                            
                                                            # Sample to avoid overloading with a single file
                                                            max_conflicts = min(2000, len(conflict_events))
                                                            max_non_conflicts = min(3000, len(non_conflict_events))
                                                            
                                                            conflict_sample = conflict_events.sample(max_conflicts) if max_conflicts > 0 else conflict_events
                                                            non_conflict_sample = non_conflict_events.sample(max_non_conflicts) if max_non_conflicts > 0 else non_conflict_events
                                                            file_sample = pd.concat([conflict_sample, non_conflict_sample])
                                                            
                                                            all_gdelt_data.append(file_sample)
                                                            total_samples += len(file_sample)
                                                            logger.info(f"  Added {len(file_sample)} events from {file_name} (total: {total_samples})")
                                                            
                                                            if total_samples >= sample_limit:
                                                                break
                                                                
                                                        except Exception as e:
                                                            logger.error(f"  Error processing file {file_name}: {e}")
                                            
                                    except zipfile.BadZipFile:
                                        logger.error(f"  Invalid zip file from {data_url}")
                                        
                                else:
                                    logger.warning(f"  Could not download data from {data_url}: Status code {data_response.status_code}")
                                    
                            except Exception as e:
                                logger.error(f"  Error downloading data from {data_url}: {e}")
            else:
                logger.error(f"Could not fetch lastupdate.txt: Status code {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error fetching lastupdate.txt: {e}")
        
        # Move to the previous day if we need more data
        if total_samples < sample_limit:
            current_date = current_date - datetime.timedelta(days=1)
            logger.info(f"Moving to data from {current_date.strftime('%Y-%m-%d')} to get more samples")
            
            # For historical data, try the specific time-based URL pattern
            for hour in [0, 6, 12, 18]:
                if total_samples >= sample_limit:
                    break
                    
                for minute in [0, 15, 30, 45]:
                    if total_samples >= sample_limit:
                        break
                        
                    test_time = current_date.replace(hour=hour, minute=minute, second=0, microsecond=0)
                    formatted_time = test_time.strftime('%Y%m%d%H%M%S')
                    url = f"http://data.gdeltproject.org/gdeltv2/{formatted_time}.export.CSV.zip"
                    
                    logger.info(f"Trying historical GDELT 2.0 data from {url}")
                    
                    try:
                        response = requests.head(url, timeout=10)
                        
                        if response.status_code == 200:
                            # Download the file
                            try:
                                data_response = requests.get(url, timeout=60)
                                
                                if data_response.status_code == 200:
                                    # Extract and process the CSV file
                                    try:
                                        with zipfile.ZipFile(io.BytesIO(data_response.content)) as zip_file:
                                            for file_name in zip_file.namelist():
                                                if file_name.endswith('.CSV') or file_name.endswith('.csv'):
                                                    with zip_file.open(file_name) as f:
                                                        # GDELT 2.0 has 61 columns
                                                        cols = [
                                                            "GLOBALEVENTID", "SQLDATE", "MonthYear", "Year", "FractionDate",
                                                            "Actor1Code", "Actor1Name", "Actor1CountryCode", "Actor1KnownGroupCode",
                                                            "Actor1EthnicCode", "Actor1Religion1Code", "Actor1Religion2Code",
                                                            "Actor1Type1Code", "Actor1Type2Code", "Actor1Type3Code",
                                                            "Actor2Code", "Actor2Name", "Actor2CountryCode", "Actor2KnownGroupCode",
                                                            "Actor2EthnicCode", "Actor2Religion1Code", "Actor2Religion2Code",
                                                            "Actor2Type1Code", "Actor2Type2Code", "Actor2Type3Code",
                                                            "IsRootEvent", "EventCode", "EventBaseCode", "EventRootCode",
                                                            "QuadClass", "GoldsteinScale", "NumMentions", "NumSources", "NumArticles",
                                                            "AvgTone", "Actor1Geo_Type", "Actor1Geo_FullName", "Actor1Geo_CountryCode",
                                                            "Actor1Geo_ADM1Code", "Actor1Geo_ADM2Code", "Actor1Geo_Lat", "Actor1Geo_Long",
                                                            "Actor1Geo_FeatureID", "Actor2Geo_Type", "Actor2Geo_FullName",
                                                            "Actor2Geo_CountryCode", "Actor2Geo_ADM1Code", "Actor2Geo_ADM2Code",
                                                            "Actor2Geo_Lat", "Actor2Geo_Long", "Actor2Geo_FeatureID", "ActionGeo_Type",
                                                            "ActionGeo_FullName", "ActionGeo_CountryCode", "ActionGeo_ADM1Code",
                                                            "ActionGeo_ADM2Code", "ActionGeo_Lat", "ActionGeo_Long", "ActionGeo_FeatureID",
                                                            "DATEADDED", "SOURCEURL"
                                                        ]
                                                        
                                                        try:
                                                            # Read GDELT 2.0 data
                                                            df = pd.read_csv(
                                                                f, 
                                                                sep="\t", 
                                                                header=None,
                                                                names=cols,
                                                                usecols=["GLOBALEVENTID", "SQLDATE", "Actor1CountryCode", "Actor2CountryCode", "EventCode", "GoldsteinScale", "SOURCEURL"],
                                                                encoding="latin-1",
                                                                on_bad_lines='skip',
                                                                low_memory=False
                                                            )
                                                            
                                                            # Remove duplicates
                                                            if "GLOBALEVENTID" in df.columns:
                                                                df = df[~df["GLOBALEVENTID"].isin(event_ids)]
                                                                event_ids.update(df["GLOBALEVENTID"].tolist())
                                                            
                                                            # Skip if no data after filtering duplicates
                                                            if df.empty:
                                                                logger.info(f"  No new events in {file_name} after removing duplicates")
                                                                continue
                                                            
                                                            # Process data
                                                            df["date"] = pd.to_datetime(df["SQLDATE"], format="%Y%m%d", errors="coerce")
                                                            df["label"] = df["EventCode"].astype(str).str.startswith("19").astype(int)
                                                            df["actor1"] = df["Actor1CountryCode"].fillna("Unknown")
                                                            df["actor2"] = df["Actor2CountryCode"].fillna("Unknown")
                                                            
                                                            # Extract text for each event
                                                            df["text"] = "GDELT event between " + df["actor1"] + " and " + \
                                                                        df["actor2"] + " with code " + df["EventCode"].astype(str)
                                                            
                                                            # Add data source field
                                                            df["source"] = "gdelt"
                                                            
                                                            # Keep only needed columns
                                                            df = df[["date", "text", "actor1", "actor2", "label", "source", "GoldsteinScale"]]
                                                            
                                                            # Get balanced samples
                                                            conflict_events = df[df["label"] == 1]
                                                            non_conflict_events = df[df["label"] == 0]
                                                            
                                                            # Sample to avoid overloading with a single file
                                                            max_conflicts = min(2000, len(conflict_events))
                                                            max_non_conflicts = min(3000, len(non_conflict_events))
                                                            
                                                            conflict_sample = conflict_events.sample(max_conflicts) if max_conflicts > 0 else conflict_events
                                                            non_conflict_sample = non_conflict_events.sample(max_non_conflicts) if max_non_conflicts > 0 else non_conflict_events
                                                            file_sample = pd.concat([conflict_sample, non_conflict_sample])
                                                            
                                                            all_gdelt_data.append(file_sample)
                                                            total_samples += len(file_sample)
                                                            logger.info(f"  Added {len(file_sample)} events from {file_name} (total: {total_samples})")
                                                            
                                                            if total_samples >= sample_limit:
                                                                break
                                                                
                                                        except Exception as e:
                                                            logger.error(f"  Error processing file {file_name}: {e}")
                                            
                                    except zipfile.BadZipFile:
                                        logger.error(f"  Invalid zip file from {url}")
                                        
                                else:
                                    logger.warning(f"  Could not download data from {url}: Status code {data_response.status_code}")
                                    
                            except Exception as e:
                                logger.error(f"  Error downloading data from {url}: {e}")
                        else:
                            logger.warning(f"  URL not available: {url}")
                            
                    except Exception as e:
                        logger.error(f"  Error checking URL {url}: {e}")
    
    # Combine all data
    if not all_gdelt_data:
        logger.error("No GDELT data was successfully retrieved. Creating empty DataFrame.")
        # Return empty DataFrame with required columns instead of None
        return pd.DataFrame(columns=["date", "text", "actor1", "actor2", "label", "source", "GoldsteinScale"])
        
    gdelt_df = pd.concat(all_gdelt_data, ignore_index=True)
    
    # Balance the dataset one more time if needed
    conflict_events = gdelt_df[gdelt_df["label"] == 1]
    non_conflict_events = gdelt_df[gdelt_df["label"] == 0]
    logger.info(f"Total conflict events: {len(conflict_events)}, non-conflict events: {len(non_conflict_events)}")
    
    # Make sure we have enough data
    if total_samples < 30000:
        logger.warning(f"Only gathered {total_samples} samples, which is less than the target of 30,000")
    
    # Ensure we have at least 40% conflict events for good training
    min_conflict_ratio = 0.4
    current_conflict_ratio = len(conflict_events) / len(gdelt_df) if len(gdelt_df) > 0 else 0
    
    # Sample non-conflict events if needed to achieve the desired ratio
    if current_conflict_ratio < min_conflict_ratio and len(non_conflict_events) > 0:
        # Calculate the desired number of non-conflict events to keep
        target_non_conflict = int(len(conflict_events) * (1 - min_conflict_ratio) / min_conflict_ratio)
        
        # Sample non-conflict events
        if target_non_conflict < len(non_conflict_events):
            non_conflict_sample = non_conflict_events.sample(target_non_conflict)
            gdelt_df = pd.concat([conflict_events, non_conflict_sample], ignore_index=True)
            logger.info(f"Balanced dataset: {len(conflict_events)} conflict, {len(non_conflict_sample)} non-conflict events")
    
    # Save to CSV
    gdelt_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(gdelt_df)} GDELT events to {output_file}")
    
    return gdelt_df

def fetch_acled(sample_limit=30000, force=False, days_back=3650, region=None):
    """
    Fetch global ACLED conflict data via API, process, and save to JSON and CSV.
    
    Args:
        sample_limit: Target number of samples to collect
        force: Whether to force download even if data already exists
        days_back: Number of days of historical data to fetch (default 10 years)
        region: Optional specific region to fetch ("Africa", "Middle East", etc.)
    """
    output_file_json = "data/acled_events.json"
    output_file_csv = "data/acled_events_clean.csv"
    
    # Check if data already exists and force is not set
    if os.path.exists(output_file_json) and os.path.exists(output_file_csv) and not force:
        logger.info(f"ACLED data already exists at {output_file_csv}. Use --force to re-download.")
        try:
            return pd.read_csv(output_file_csv)
        except (json.JSONDecodeError, pd.errors.EmptyDataError) as e:
            logger.warning(f"Existing ACLED data is invalid: {e}. Re-downloading.")
    
    logger.info(f"Fetching ACLED data for the past {days_back} days (target: {sample_limit} samples)...")
    
    if not ACLED_API_KEY:
        logger.error("Error: ACLED_API_KEY not found in .env file")
        return pd.DataFrame()  # Return empty DataFrame instead of None
        
    if not ACLED_EMAIL:
        logger.error("Error: ACLED_EMAIL not found in .env file")
        return pd.DataFrame()  # Return empty DataFrame instead of None
    
    # Calculate date range
    end_date = datetime.datetime.now()
    start_date = end_date - datetime.timedelta(days=days_back)
    
    # Format dates for ACLED API
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    logger.info(f"Fetching ACLED data from {start_date_str} to {end_date_str}")
    logger.info(f"Using ACLED API key: {ACLED_API_KEY[:4]}*** and email: {ACLED_EMAIL}")
    
    # Regions to query if specified
    regions = [region] if region else []
    
    base_url = "https://api.acleddata.com/acled/read"
    all_events = []
    duplicates_removed = 0
    event_ids = set()  # Track event IDs to avoid duplicates
    
    # Try with larger time ranges if smaller ones fail
    if days_back < 90:
        logger.info(f"Using a small time window ({days_back} days). If no data is returned, consider using a larger time window.")
    
    # First try querying without region filter if no specific region is provided
    if not region:
        logger.info("Querying ACLED data without region filter...")
        page = 1
        max_retries = 5
        page_size = 500  # ACLED API limit per request
        more_pages = True
        
        while more_pages:
            if len(all_events) >= sample_limit:
                logger.info(f"Reached sample limit of {sample_limit}")
                break
                
            params = {
                "key": ACLED_API_KEY,
                "email": ACLED_EMAIL,
                "page": page,
                "limit": page_size,
                "format": "json",
                "event_date": f"{start_date_str}|{end_date_str}"  # Filter by date range
            }
            
            # Log full URL for debugging (without API key)
            debug_params = params.copy()
            debug_params["key"] = debug_params["key"][:4] + "***"
            query_string = "&".join([f"{k}={v}" for k, v in debug_params.items()])
            logger.info(f"Request URL: {base_url}?{query_string}")
            
            success = False
            for retry in range(max_retries):
                try:
                    logger.info(f"Fetching page {page} without region filter (attempt {retry+1}/{max_retries})")
                    response = requests.get(base_url, params=params, timeout=60)
                    
                    # Check for authentication errors
                    if response.status_code == 401:
                        logger.error(f"Authentication failed: {response.text}")
                        return pd.DataFrame()
                    
                    # Check for rate limit errors
                    if response.status_code == 429:
                        logger.error(f"Rate limit exceeded: {response.text}")
                        # Wait longer before retrying
                        wait_time = 30 * (retry + 1)  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    
                    # Check for bad parameter errors
                    if response.status_code == 400:
                        logger.error(f"Bad parameter: {response.text}")
                        break
                    
                    if response.status_code == 200:
                        # Try to parse the JSON response
                        try:
                            data = response.json()
                            
                            # Check if response contains data field
                            if "data" not in data:
                                logger.warning(f"Response missing 'data' field: {data}")
                                logger.warning("No 'data' field in response. API may have changed format.")
                                if retry == max_retries - 1:
                                    break
                                time.sleep(5)
                                continue
                            
                            # Check for empty data array
                            if not data["data"]:
                                logger.info(f"  No events found for this page/time period")
                                more_pages = False
                                success = True
                                break
                            
                            if "count" in data and data["count"] > 0 and data["data"]:
                                page_events = data["data"]
                                logger.info(f"  Retrieved {len(page_events)} events (total records: {data['count']})")
                                
                                # Filter out duplicates
                                new_events = []
                                for event in page_events:
                                    if "event_id_cnty" in event:
                                        event_id = event["event_id_cnty"]
                                        if event_id not in event_ids:
                                            event_ids.add(event_id)
                                            new_events.append(event)
                                        else:
                                            duplicates_removed += 1
                                    else:
                                        # If no ID, use another field as a makeshift ID
                                        event_id = f"{event.get('event_date', '')}_{event.get('event_type', '')}_{event.get('actor1', '')}_{event.get('location', '')}"
                                        if event_id not in event_ids:
                                            event_ids.add(event_id)
                                            new_events.append(event)
                                        else:
                                            duplicates_removed += 1
                                
                                if new_events:
                                    all_events.extend(new_events)
                                    logger.info(f"  Added {len(new_events)} unique events (total: {len(all_events)}, duplicates removed: {duplicates_removed})")
                                else:
                                    logger.info(f"  No new unique events found on this page")
                                
                                # Check if we need to fetch more pages
                                if "count" in data and page * page_size < data["count"]:
                                    page += 1
                                else:
                                    logger.info(f"  Reached the end of results")
                                    more_pages = False
                                    
                                success = True
                                break
                            else:
                                logger.info(f"  No more data available")
                                more_pages = False
                                success = True
                                break
                        except json.JSONDecodeError as e:
                            logger.error(f"  Failed to parse JSON response: {e}")
                            logger.error(f"  Response content: {response.text[:500]}")
                            # Try again with next retry
                    else:
                        logger.error(f"  Error fetching ACLED data: Status code {response.status_code}")
                        logger.error(f"  Response: {response.text[:500]}")
                        # Try again with next retry
                        
                except Exception as e:
                    logger.error(f"  Error fetching ACLED data: {e}")
                    # Try again with next retry
                
                # Wait before retrying
                time.sleep(retry * 5 + 2)
            
            if not success:
                logger.error(f"Failed to fetch data after {max_retries} attempts")
                more_pages = False
    
    # If we have regions to query, or if we didn't get any data without region filter
    if (regions or not all_events) and region is not None:
        # Default regions if none were specified and we didn't get data without filter
        if not regions:
            regions = ["Middle East", "Africa", "Asia", "Europe", "South America", 
                       "North America", "Central America", "Oceania"]
    
    # Fetch data for each region
    for region_name in regions:
        if len(all_events) >= sample_limit:
            logger.info(f"Reached sample limit of {sample_limit}")
            break
            
        logger.info(f"Fetching ACLED data for {region_name}...")
        page = 1
        max_retries = 5  # Increased retries from 3 to 5
        page_size = 500  # ACLED API limit per request
        more_pages = True
        
        while more_pages:
            if len(all_events) >= sample_limit:
                logger.info(f"Reached sample limit of {sample_limit}")
                break
                
            params = {
                "key": ACLED_API_KEY,
                "email": ACLED_EMAIL,
                "page": page,
                "limit": page_size,
                "format": "json",
                "event_date": f"{start_date_str}|{end_date_str}",  # Filter by date range
                "region": region_name
            }
            
            # Log full URL for debugging (without API key)
            debug_params = params.copy()
            debug_params["key"] = debug_params["key"][:4] + "***"
            query_string = "&".join([f"{k}={v}" for k, v in debug_params.items()])
            logger.info(f"Request URL: {base_url}?{query_string}")
            
            success = False
            for retry in range(max_retries):
                try:
                    logger.info(f"Fetching page {page} for {region_name} (attempt {retry+1}/{max_retries})")
                    response = requests.get(base_url, params=params, timeout=60)
                    
                    # Check for authentication errors
                    if response.status_code == 401:
                        logger.error(f"Authentication failed: {response.text}")
                        return pd.DataFrame()
                    
                    # Check for rate limit errors
                    if response.status_code == 429:
                        logger.error(f"Rate limit exceeded: {response.text}")
                        # Wait longer before retrying
                        wait_time = 30 * (retry + 1)  # Exponential backoff
                        logger.info(f"Waiting {wait_time} seconds before retry...")
                        time.sleep(wait_time)
                        continue
                    
                    # Check for bad parameter errors
                    if response.status_code == 400:
                        logger.error(f"Bad parameter: {response.text}")
                        break
                    
                    if response.status_code == 200:
                        # Try to parse the JSON response
                        try:
                            data = response.json()
                            
                            # Check if response contains data field
                            if "data" not in data:
                                logger.warning(f"Response missing 'data' field: {data}")
                                logger.warning("No 'data' field in response. API may have changed format.")
                                if retry == max_retries - 1:
                                    break
                                time.sleep(5)
                                continue
                            
                            # Check for empty data array
                            if not data["data"]:
                                logger.info(f"  No events found in {region_name} for this page/time period")
                                more_pages = False
                                success = True
                                break
                            
                            if "count" in data and data["count"] > 0 and data["data"]:
                                page_events = data["data"]
                                logger.info(f"  Retrieved {len(page_events)} events (total records: {data['count']})")
                                
                                # Filter out duplicates
                                new_events = []
                                for event in page_events:
                                    if "event_id_cnty" in event:
                                        event_id = event["event_id_cnty"]
                                        if event_id not in event_ids:
                                            event_ids.add(event_id)
                                            new_events.append(event)
                                        else:
                                            duplicates_removed += 1
                                    else:
                                        # If no ID, use another field as a makeshift ID
                                        event_id = f"{event.get('event_date', '')}_{event.get('event_type', '')}_{event.get('actor1', '')}_{event.get('location', '')}"
                                        if event_id not in event_ids:
                                            event_ids.add(event_id)
                                            new_events.append(event)
                                        else:
                                            duplicates_removed += 1
                                
                                if new_events:
                                    all_events.extend(new_events)
                                    logger.info(f"  Added {len(new_events)} unique events (total: {len(all_events)}, duplicates removed: {duplicates_removed})")
                                else:
                                    logger.info(f"  No new unique events found on this page")
                                
                                # Check if we need to fetch more pages
                                if "count" in data and page * page_size < data["count"]:
                                    page += 1
                                else:
                                    logger.info(f"  Reached the end of results for {region_name}")
                                    more_pages = False
                                    
                                success = True
                                break
                            else:
                                logger.info(f"  No more data available for {region_name}")
                                more_pages = False
                                success = True
                                break
                        except json.JSONDecodeError as e:
                            logger.error(f"  Failed to parse JSON response: {e}")
                            logger.error(f"  Response content: {response.text[:500]}")
                            # Try again with next retry
                    else:
                        logger.error(f"  Error fetching ACLED data: Status code {response.status_code}")
                        logger.error(f"  Response: {response.text[:500]}")
                        # Try again with next retry
                        
                except Exception as e:
                    logger.error(f"  Error fetching ACLED data: {e}")
                    # Try again with next retry
                
                # Wait before retrying
                wait_time = retry * 5 + 5  # Exponential backoff
                logger.info(f"  Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
            
            if not success:
                logger.error(f"  Failed to fetch data for {region_name} after {max_retries} attempts, moving to next region")
                more_pages = False
    
    # Check if we got any data
    if not all_events:
        logger.error("No ACLED data was successfully retrieved")
        logger.info("This could be due to:")
        logger.info("1. No events within the specified date range")
        logger.info("2. API key or email authentication issues")
        logger.info("3. Rate limits or temporary API unavailability")
        logger.info("Try increasing the days_back parameter or check API credentials")
        
        # Save empty JSON to avoid repeated failed requests
        with open(output_file_json, "w") as f:
            json.dump([], f)
        return pd.DataFrame()
    
    # Save raw JSON
    with open(output_file_json, "w") as f:
        json.dump(all_events, f)
    logger.info(f"Saved {len(all_events)} raw ACLED events to {output_file_json}")
    
    # Process the data
    processed_events = []
    
    for event in all_events:
        # Extract relevant fields
        event_type = event.get("event_type", "")
        actor1 = event.get("actor1", "")
        actor2 = event.get("actor2", "")
        fatalities = event.get("fatalities", 0)
        notes = event.get("notes", "")
        event_date = event.get("event_date", "")
        location = event.get("location", "")
        country = event.get("country", "")
        
        # Generate text field
        text = f"{event_type} event between {actor1} and {actor2} in {location}, {country}: {notes}"
        
        # Label: 1 if event_type is in conflict categories
        conflict_types = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
        label = 1 if event_type in conflict_types else 0
        
        processed_events.append({
            "date": event_date,
            "text": text,
            "actor1": actor1,
            "actor2": actor2,
            "label": label,
            "fatalities": fatalities,
            "source": "acled",
            "event_type": event_type,
            "location": location
        })
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_events)
    
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    
    # Save to CSV
    df.to_csv(output_file_csv, index=False)
    logger.info(f"Saved {len(df)} processed ACLED events to {output_file_csv}")
    
    # Report statistics
    conflict_events = df[df["label"] == 1]
    non_conflict_events = df[df["label"] == 0]
    logger.info(f"ACLED data statistics:")
    logger.info(f"  Total events: {len(df)}")
    logger.info(f"  Conflict events: {len(conflict_events)} ({len(conflict_events)/len(df)*100:.1f}%)")
    logger.info(f"  Non-conflict events: {len(non_conflict_events)} ({len(non_conflict_events)/len(df)*100:.1f}%)")
    logger.info(f"  Events with text: {df['text'].notna().sum()} ({df['text'].notna().sum()/len(df)*100:.1f}%)")
    logger.info(f"  Unique event types: {df['event_type'].nunique()}")
    logger.info(f"  Top event types: {df['event_type'].value_counts().head(3).to_dict()}")
    
    return df

def load_ucdp_csv():
    """
    Load and process UCDP CSV files from the data/ucdp/csv directory.
    
    The function automatically detects and handles different CSV formats,
    including those with 57-61 columns.
    
    Returns:
        DataFrame with processed UCDP CSV data
    """
    csv_dir = "data/ucdp/csv"
    output_file = "data/ucdp_csv_clean.csv"
    
    # Check if the output file already exists
    if os.path.exists(output_file):
        logger.info(f"UCDP CSV data already processed at {output_file}")
        try:
            return pd.read_csv(output_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Error reading processed UCDP CSV data: {e}. Reprocessing.")
    
    # Find all CSV files in the directory
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return pd.DataFrame()
    
    logger.info(f"Found {len(csv_files)} UCDP CSV files to process")
    
    all_dataframes = []
    total_rows = 0
    
    for csv_file in csv_files:
        file_name = os.path.basename(csv_file)
        logger.info(f"Processing {file_name}...")
        
        try:
            # Try to read the CSV with default settings
            df = pd.read_csv(csv_file, encoding='latin-1', low_memory=False)
            
            # Normalize column names to lowercase, strip, and replace spaces with underscores
            df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]
            
            # Check for essential columns
            essential_columns = ['side_a', 'side_b', 'year']
            if not all(col in df.columns for col in essential_columns):
                essential_found = [col for col in essential_columns if col in df.columns]
                essential_missing = [col for col in essential_columns if col not in df.columns]
                logger.warning(f"File {file_name} is missing essential columns: {essential_missing}")
                logger.warning(f"Found columns: {essential_found}")
                
                # Try fuzzy matching for columns
                col_mapping = {}
                for missing_col in essential_missing:
                    # Try variations of column names
                    variations = [
                        missing_col.upper(),
                        missing_col.replace('_', ' '),
                        missing_col.replace('_', ''),
                        f"{missing_col}_name",
                        f"{missing_col}_id",
                    ]
                    
                    for var in variations:
                        matches = [col for col in df.columns if var in col]
                        if matches:
                            col_mapping[missing_col] = matches[0]
                            logger.info(f"Mapped missing column {missing_col} to {matches[0]}")
                            break
                
                # Rename columns based on mapping
                if col_mapping:
                    df = df.rename(columns=col_mapping)
                else:
                    logger.error(f"Could not find suitable columns in {file_name}, skipping")
                    continue
            
            # Check for required fields and fill if missing
            required_fields = ["year", "side_a", "side_b", "region", "country", "type_of_violence"]
            for field in required_fields:
                if field not in df.columns:
                    logger.warning(f"Adding missing column {field} with 'Unknown' values")
                    df[field] = "Unknown"
            
            # Generate a text field for each event
            location_field = "country" if "country" in df.columns else "region"
            
            df["text"] = df.apply(
                lambda row: f"Conflict between {row['side_a']} and {row['side_b']} in {row[location_field]} ({row['year']})" + 
                           (f", type: {row['type_of_violence']}" if 'type_of_violence' in df.columns else ""),
                axis=1
            )
            
            # Set label to 1 for all rows (all UCDP data represents conflicts)
            df["label"] = 1
            
            # Add source field and date
            df["source"] = f"ucdp_csv_{file_name}"
            df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01", errors='coerce')
            
            # Select columns for output
            columns_to_keep = ['date', 'text', 'label', 'source', 'side_a', 'side_b']
            # Add any additional useful columns if they exist
            for col in ['type_of_violence', 'deaths_best', 'best', 'high', 'low', 'year']:
                if col in df.columns:
                    columns_to_keep.append(col)
            
            # Keep only columns that exist in the DataFrame
            available_cols = [col for col in columns_to_keep if col in df.columns]
            df = df[available_cols]
            
            # Add to collection
            all_dataframes.append(df)
            total_rows += len(df)
            logger.info(f"Added {len(df)} rows from {file_name} (total: {total_rows})")
            
        except Exception as e:
            logger.error(f"Error processing file {file_name}: {e}")
    
    # Combine all data
    if not all_dataframes:
        logger.warning("No UCDP CSV data was successfully processed")
        return pd.DataFrame()
    
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    # Remove duplicates if any
    pre_dedup_count = len(combined_df)
    combined_df = combined_df.drop_duplicates(subset=['text', 'date'], keep='first')
    post_dedup_count = len(combined_df)
    
    if pre_dedup_count > post_dedup_count:
        logger.info(f"Removed {pre_dedup_count - post_dedup_count} duplicate entries")
    
    # Save to CSV
    combined_df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(combined_df)} processed UCDP CSV events to {output_file}")
    
    # Report statistics
    logger.info(f"UCDP CSV data statistics:")
    logger.info(f"  Total events: {len(combined_df)}")
    logger.info(f"  Events with text: {combined_df['text'].notna().sum()} ({combined_df['text'].notna().sum()/len(combined_df)*100:.1f}%)")
    if "type_of_violence" in combined_df.columns:
        logger.info(f"  Violence types: {combined_df['type_of_violence'].value_counts().to_dict()}")
    
    return combined_df

def fetch_ucdp_api(resource="gedevents", version="24.1", pagesize=1000, force=False, sample_limit=100000):
    """
    Fetch data from UCDP API, process, and save to CSV.
    
    Args:
        resource: API resource (e.g., "gedevents" for Georeferenced Event Dataset)
        version: API version (e.g., "24.1" is the latest as of 2025)
        pagesize: Number of records per page
        force: Whether to force download even if data already exists
        sample_limit: Maximum number of records to fetch (default: 100,000)
    """
    output_file = "data/ucdp_api_clean.csv"
    raw_file = "data/ucdp_api_raw.json"
    
    # Check if data already exists and force is not set
    if os.path.exists(output_file) and not force:
        logger.info(f"UCDP API data already exists at {output_file}. Use --force to re-download.")
        try:
            return pd.read_csv(output_file)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Error reading existing UCDP API data: {e}. Re-downloading.")
    
    logger.info(f"Fetching data from UCDP API ({resource}/{version}), limit: {sample_limit} records...")
    
    base_url = f"https://ucdpapi.pcr.uu.se/api/{resource}/{version}"
    
    all_data = []
    current_page = 1
    more_pages = True
    max_retries = 3
    
    while more_pages and len(all_data) < sample_limit:
        url = f"{base_url}?page={current_page}&pagesize={pagesize}"
        success = False
        
        for retry in range(max_retries):
            try:
                logger.info(f"Fetching page {current_page} from {url} (attempt {retry+1}/{max_retries})")
                
                response = requests.get(url, timeout=180)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Check if there's data in the current page
                    if "Result" in data and data["Result"]:
                        # Only add up to the sample limit
                        remaining = sample_limit - len(all_data)
                        records_to_add = data["Result"][:remaining] if remaining < len(data["Result"]) else data["Result"]
                        all_data.extend(records_to_add)
                        logger.info(f"  Fetched {len(records_to_add)} records (total: {len(all_data)}/{sample_limit})")
                        
                        # Stop if we've reached the sample limit
                        if len(all_data) >= sample_limit:
                            logger.info(f"  Reached sample limit of {sample_limit} records")
                            more_pages = False
                            break
                        
                        # Check if there are more pages
                        if len(data["Result"]) < pagesize:
                            more_pages = False
                            logger.info("  Reached last page of results")
                        else:
                            current_page += 1
                        
                        success = True
                        break
                    else:
                        more_pages = False
                        logger.info("  No more results or empty response")
                        success = True
                        break
                elif response.status_code == 429:  # Rate limit
                    logger.warning(f"  Rate limit reached (status code {response.status_code}). Waiting before retry...")
                    time.sleep(30)  # Wait longer for rate limits
                elif response.status_code == 404:
                    logger.error(f"  API endpoint not found: {url}")
                    more_pages = False
                    break
                else:
                    logger.error(f"  Failed to fetch data: Status code {response.status_code}")
                    # Try again with next retry
                    
            except Exception as e:
                logger.error(f"  Error fetching data from UCDP API: {e}")
                # Try again with next retry
            
            # Wait before retrying
            time.sleep(retry * 5 + 2)  # Exponential backoff
        
        if not success:
            logger.error(f"  Failed to fetch data after {max_retries} attempts, aborting")
            more_pages = False
    
    # Save raw JSON data
    if all_data:
        with open(raw_file, "w") as f:
            json.dump(all_data, f)
        logger.info(f"Saved raw UCDP API data to {raw_file}")
    else:
        logger.warning("No data retrieved from UCDP API")
        return pd.DataFrame()
    
    # Process the data
    try:
        # Convert to DataFrame
        df = pd.json_normalize(all_data)
        
        # Normalize column names
        df.columns = [col.lower().replace('.', '_') for col in df.columns]
        
        # Check for essential fields
        expected_fields = ["id", "year", "side_a", "side_b", "type_of_violence", "best", "region", "country"]
        available_fields = [field for field in expected_fields if field in df.columns]
        missing_fields = [field for field in expected_fields if field not in df.columns]
        
        if missing_fields:
            logger.warning(f"Missing some expected fields in API response: {missing_fields}")
        
        if not available_fields:
            logger.error("No essential fields found in API response")
            return pd.DataFrame()
        
        # Select available fields and fill missing ones
        for field in expected_fields:
            if field not in df.columns:
                df[field] = "Unknown"
        
        # Synthesize text field
        df["text"] = df.apply(
            lambda row: f"Type {row['type_of_violence']} conflict between {row['side_a']} and {row['side_b']} in {row.get('country', row['region'])}",
            axis=1
        )
        
        # Set label to 1 for all rows (UCDP data is all conflict)
        df["label"] = 1
        
        # Add source field
        df["source"] = "ucdp_api"
        
        # Add date field (using year)
        df["date"] = pd.to_datetime(df["year"].astype(str) + "-01-01", errors='coerce')
        
        # Keep only needed columns
        output_columns = ["date", "text", "label", "source", "side_a", "side_b", "type_of_violence", "best"]
        available_output_columns = [col for col in output_columns if col in df.columns]
        df = df[available_output_columns]
        
        # Save to CSV
        df.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df)} UCDP API events to {output_file}")
        
        # Report statistics
        logger.info(f"UCDP API data statistics:")
        logger.info(f"  Total events: {len(df)}")
        logger.info(f"  Events with text: {df['text'].notna().sum()} ({df['text'].notna().sum()/len(df)*100:.1f}%)")
        if "type_of_violence" in df.columns:
            logger.info(f"  Violence types: {df['type_of_violence'].value_counts().to_dict()}")
        if "best" in df.columns:
            logger.info(f"  Total reported casualties: {df['best'].sum()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error processing UCDP API data: {e}")
        return pd.DataFrame()

def preprocess_gdelt(gdelt_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess GDELT data with text synthesis and tag encoding.
    
    Args:
        gdelt_df: Raw GDELT DataFrame
    
    Returns:
        Processed DataFrame with text, label, and confidence
    """
    if gdelt_df.empty:
        return pd.DataFrame(columns=["text", "label", "confidence", "source"])
    
    # Create copies of needed columns to avoid SettingWithCopyWarning
    processed_df = gdelt_df.copy()
    
    # Create label from EventCode (19* codes are conflict)
    processed_df["label"] = processed_df["EventCode"].astype(str).str.startswith("19").astype(int)
    
    # Convert Goldstein scale to confidence (Goldstein is -10 to +10, convert to 0-1)
    processed_df["confidence"] = ((processed_df["GoldsteinScale"] + 10) / 20).clip(0, 1)
    
    # Synthesize text description
    processed_df["text"] = (
        "Conflict between " + processed_df["actor1"].fillna("UNKNOWN") +
        " and " + processed_df["actor2"].fillna("UNKNOWN") +
        " with EventCode " + processed_df["EventCode"].astype(str)
    )
    
    # Mark source
    processed_df["source"] = "gdelt"
    
    # Keep only needed columns
    return processed_df[["text", "label", "confidence", "source"]]

def preprocess_acled(acled_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess ACLED data with text synthesis and tag encoding.
    
    Args:
        acled_df: Raw ACLED DataFrame
    
    Returns:
        Processed DataFrame with text, label, and confidence
    """
    if acled_df.empty:
        return pd.DataFrame(columns=["text", "label", "confidence", "source"])
    
    # Create copies of needed columns to avoid SettingWithCopyWarning
    processed_df = acled_df.copy()
    
    # Label based on event type (known conflict types)
    conflict_types = ["Battles", "Explosions/Remote violence", "Violence against civilians"]
    if "event_type" in processed_df.columns:
        processed_df["label"] = processed_df["event_type"].isin(conflict_types).astype(int)
    else:
        # Default all ACLED events to conflict if event_type not available
        processed_df["label"] = 1
    
    # Confidence based on fatalities (if available)
    if "fatalities" in processed_df.columns:
        processed_df["confidence"] = processed_df["fatalities"].fillna(0).astype(float) / 10
        processed_df["confidence"] = processed_df["confidence"].clip(0, 1)
    else:
        processed_df["confidence"] = 0.5  # Default confidence
    
    # Synthesize text description - fixed to handle Series properly
    processed_df["text"] = ""
    
    # Process each row individually to build text
    for idx, row in processed_df.iterrows():
        text_parts = []
        
        if "event_type" in processed_df.columns:
            text_parts.append(f"Event: {row['event_type'] if pd.notnull(row['event_type']) else ''}")
        
        if "actor1" in processed_df.columns and "actor2" in processed_df.columns:
            actor1 = row["actor1"] if pd.notnull(row["actor1"]) else "UNKNOWN"
            actor2 = row["actor2"] if pd.notnull(row["actor2"]) else "UNKNOWN"
            text_parts.append(f"Actors: {actor1} vs {actor2}")
        
        if "notes" in processed_df.columns:
            notes = row["notes"] if pd.notnull(row["notes"]) else ""
            text_parts.append(f"Notes: {notes}")
        
        processed_df.at[idx, "text"] = " | ".join(text_parts) if text_parts else "ACLED event with insufficient details"
    
    # Mark source
    processed_df["source"] = "acled"
    
    # Keep only needed columns
    return processed_df[["text", "label", "confidence", "source"]]

def preprocess_ucdp(ucdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess UCDP data with text synthesis and tag encoding.
    
    Args:
        ucdp_df: Raw UCDP DataFrame
    
    Returns:
        Processed DataFrame with text, label, and confidence
    """
    if ucdp_df.empty:
        return pd.DataFrame(columns=["text", "label", "confidence", "source"])
    
    # Create copies of needed columns to avoid SettingWithCopyWarning
    processed_df = ucdp_df.copy()
    
    # All UCDP events are conflicts
    processed_df["label"] = 1
    
    # Confidence based on casualties (if available)
    if "best" in processed_df.columns:
        processed_df["confidence"] = processed_df["best"].fillna(0).astype(float) / 100
        processed_df["confidence"] = processed_df["confidence"].clip(0, 1)
    else:
        processed_df["confidence"] = 0.8  # Higher default confidence for UCDP
    
    # Initialize text column
    processed_df["text"] = ""
    
    # Process each row individually to build text
    for idx, row in processed_df.iterrows():
        text_parts = []
        
        if "type_of_violence" in processed_df.columns:
            text_parts.append(f"Violence type {row['type_of_violence']}")
        
        if "side_a" in processed_df.columns and "side_b" in processed_df.columns:
            side_a = row["side_a"] if pd.notnull(row["side_a"]) else "UNKNOWN"
            side_b = row["side_b"] if pd.notnull(row["side_b"]) else "UNKNOWN"
            text_parts.append(f"between {side_a} and {side_b}")
        
        if "country" in processed_df.columns:
            country = row["country"] if pd.notnull(row["country"]) else ""
            text_parts.append(f"in {country}")
        elif "region" in processed_df.columns:
            region = row["region"] if pd.notnull(row["region"]) else ""
            text_parts.append(f"in {region}")
        
        processed_df.at[idx, "text"] = " ".join(text_parts) if text_parts else "UCDP conflict event with insufficient details"
    
    # Mark source
    processed_df["source"] = "ucdp"
    
    # Keep only needed columns
    return processed_df[["text", "label", "confidence", "source"]]

def create_unified_dataset(gdelt_df: pd.DataFrame, acled_df: pd.DataFrame, ucdp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a unified dataset from the three sources with tag encoding.
    
    Args:
        gdelt_df: Processed GDELT DataFrame
        acled_df: Processed ACLED DataFrame
        ucdp_df: Processed UCDP DataFrame
        
    Returns:
        Combined DataFrame with tags and additional metrics
    """
    # Combine all datasets
    combined_df = pd.concat([gdelt_df, acled_df, ucdp_df], ignore_index=True)
    
    # Drop rows with missing text
    combined_df = combined_df.dropna(subset=["text"])
    
    # Encode tags
    logger.info("Encoding tags for all events...")
    combined_df["tags"] = combined_df["text"].apply(tag_encoder)
    
    # Calculate tag coverage metrics
    combined_df["tag_count"] = combined_df["tags"].apply(sum)
    combined_df["no_tags"] = combined_df["tag_count"] == 0
    
    # Log tag coverage statistics
    total = len(combined_df)
    no_tagged = combined_df["no_tags"].sum()
    logger.info(f"[STATS] Total events: {total}")
    logger.info(f"[WARNING] Events with 0 matched tags: {no_tagged} ({100 * no_tagged / total:.1f}%)")
    
    # Calculate and log most common tags
    tag_matrix = np.array(combined_df["tags"].tolist())
    tag_totals = tag_matrix.sum(axis=0)
    
    logger.info("Most common tags:")
    for tag, count in sorted(zip(TAG_VOCAB, tag_totals), key=lambda x: -x[1])[:10]:
        logger.info(f"[TAG] {tag}: {int(count)} matches")
    
    return combined_df

def prepare_model_data(combined_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for XGBoost model training.
    
    Args:
        combined_df: Combined DataFrame with tags and confidence
        
    Returns:
        X (features), y (labels), texts (original text)
    """
    # Stack tag vectors with confidence for feature matrix
    X = np.hstack([
        np.array(combined_df["tags"].tolist()), 
        combined_df["confidence"].values.reshape(-1, 1)
    ])
    
    # Extract labels
    y = combined_df["label"].values
    
    # Save original text for analysis
    texts = combined_df["text"].tolist()
    
    return X, y, texts

def main():
    parser = argparse.ArgumentParser(description='Fetch and clean conflict data from multiple sources')
    parser.add_argument('--force', action='store_true', help='Force download even if data already exists')
    parser.add_argument('--gdelt-samples', type=int, default=50000, help='Target number of GDELT samples')
    parser.add_argument('--gdelt-days', type=int, default=30, help='Number of days to fetch GDELT data for')
    parser.add_argument('--acled-samples', type=int, default=30000, help='Target number of ACLED samples')
    parser.add_argument('--acled-days', type=int, default=3650, help='Number of days to fetch ACLED data for')
    parser.add_argument('--acled-region', type=str, help='Specific region to fetch ACLED data for (e.g., "Africa", "Middle East")')
    parser.add_argument('--ucdp-version', type=str, default="24.1", help='UCDP API version to use')
    parser.add_argument('--ucdp-resource', type=str, default="gedevents", help='UCDP API resource to use')
    parser.add_argument('--ucdp-limit', type=int, default=100000, help='Maximum number of UCDP records to fetch')
    parser.add_argument('--skip-gdelt', action='store_true', help='Skip GDELT data fetching')
    parser.add_argument('--skip-acled', action='store_true', help='Skip ACLED data fetching')
    parser.add_argument('--skip-ucdp-api', action='store_true', help='Skip UCDP API data fetching')
    parser.add_argument('--skip-ucdp-csv', action='store_true', help='Skip UCDP CSV data loading')
    parser.add_argument('--min-total-samples', type=int, default=30000, help='Minimum total samples required for model training')
    args = parser.parse_args()

    # Raw DataFrames
    gdelt_raw_df = pd.DataFrame()
    acled_raw_df = pd.DataFrame()
    ucdp_raw_df = pd.DataFrame()
    
    # Fetch GDELT data
    if not args.skip_gdelt:
        gdelt_raw_df = fetch_gdelt(
            sample_limit=args.gdelt_samples, 
            force=args.force, 
            days_back=args.gdelt_days
        )
    else:
        logger.info("Skipping GDELT data fetching")
    
    # Fetch ACLED data if credentials are available
    if not args.skip_acled:
        if ACLED_API_KEY and ACLED_EMAIL:
            acled_raw_df = fetch_acled(
                sample_limit=args.acled_samples, 
                force=args.force, 
                days_back=args.acled_days,
                region=args.acled_region
            )
        else:
            logger.warning("ACLED API credentials not found in environment variables. Skipping ACLED data.")
    else:
        logger.info("Skipping ACLED data fetching")
    
    # Load UCDP CSV data
    ucdp_csv_df = pd.DataFrame()
    if not args.skip_ucdp_csv:
        ucdp_csv_df = load_ucdp_csv()
    else:
        logger.info("Skipping UCDP CSV data loading")
    
    # Fetch UCDP API data
    ucdp_api_df = pd.DataFrame()
    if not args.skip_ucdp_api:
        ucdp_api_df = fetch_ucdp_api(
            resource=args.ucdp_resource,
            version=args.ucdp_version,
            force=args.force,
            sample_limit=args.ucdp_limit
        )
    else:
        logger.info("Skipping UCDP API data fetching")
    
    # Combine UCDP data sources
    if not ucdp_csv_df.empty and not ucdp_api_df.empty:
        ucdp_raw_df = pd.concat([ucdp_csv_df, ucdp_api_df], ignore_index=True)
    elif not ucdp_csv_df.empty:
        ucdp_raw_df = ucdp_csv_df
    elif not ucdp_api_df.empty:
        ucdp_raw_df = ucdp_api_df
    
    # Log raw data counts
    logger.info("=== Raw Data Counts ===")
    logger.info(f"GDELT: {len(gdelt_raw_df)} events")
    logger.info(f"ACLED: {len(acled_raw_df)} events")
    logger.info(f"UCDP: {len(ucdp_raw_df)} events")
    
    # Preprocess each dataset
    logger.info("Preprocessing datasets...")
    gdelt_processed = preprocess_gdelt(gdelt_raw_df)
    acled_processed = preprocess_acled(acled_raw_df)
    ucdp_processed = preprocess_ucdp(ucdp_raw_df)
    
    # Create unified dataset with tag encoding
    logger.info("Creating unified dataset...")
    combined_df = create_unified_dataset(gdelt_processed, acled_processed, ucdp_processed)
    
    # Check if we have enough data
    total_samples = len(combined_df)
    if total_samples < args.min_total_samples:
        logger.warning(f"WARNING: Only {total_samples} samples collected, which is below the recommended minimum of {args.min_total_samples}")
        logger.warning("Model training may not be effective. Consider fetching more data or using a smaller minimum sample threshold.")
    else:
        logger.info(f"SUCCESS: Collected {total_samples} samples, which exceeds the minimum requirement of {args.min_total_samples}")
    
    # Save the unified dataset
    logger.info("Saving unified dataset...")
    # Save as CSV
    combined_df.to_csv("data/all_conflict_data.csv", index=False)
    # Save as JSON (lines format for efficiency with large datasets)
    combined_df.to_json("data/all_conflict_data.json", orient="records", lines=True)
    
    # Prepare model-ready data
    X, y, texts = prepare_model_data(combined_df)
    
    # Save model-ready data as numpy arrays
    logger.info("Saving model-ready data...")
    os.makedirs("data/model_ready", exist_ok=True)
    np.save("data/model_ready/X_features.npy", X)
    np.save("data/model_ready/y_labels.npy", y)
    
    # Save texts as pickle for reference
    with open("data/model_ready/texts.pkl", "wb") as f:
        pickle.dump(texts, f)
    
    # Log final dataset statistics
    logger.info("\n=== Final Dataset Statistics ===")
    logger.info(f"Total samples: {len(combined_df)}")
    logger.info(f"Conflict events: {combined_df['label'].sum()} ({combined_df['label'].mean()*100:.1f}%)")
    logger.info(f"Non-conflict events: {len(combined_df) - combined_df['label'].sum()} ({(1-combined_df['label'].mean())*100:.1f}%)")
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Number of features: {X.shape[1]} ({len(TAG_VOCAB)} tags + 1 confidence score)")
    
    logger.info("Data processing pipeline completed successfully")
    
    return combined_df

if __name__ == "__main__":
    main() 