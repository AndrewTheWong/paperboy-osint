"""
Script: query_gdelt_bigquery.py
Purpose: Pull 10 years of GDELT data filtered by country from BigQuery.
"""

from google.cloud import bigquery
import pandas as pd
import os

# Auth (set GOOGLE_APPLICATION_CREDENTIALS env var to service account JSON)
client = bigquery.Client()

QUERY = """
SELECT
  DATE(PARSE_DATE('%Y%m%d', SQLDATE)) AS event_date,
  ActionGeo_CountryCode AS country,
  GoldsteinScale AS goldstein_score
FROM
  `gdelt-bq.gdeltv2.events`
WHERE
  SQLDATE BETWEEN 20150101 AND 20241231
  AND ActionGeo_CountryCode IN ('TW', 'CH')  -- Taiwan, China
"""

print("⏳ Running BigQuery...")
df = client.query(QUERY).to_dataframe()
print(f"✅ Retrieved {len(df)} rows.")

df = df.dropna()
df.to_csv("data/gdelt_bigquery_taiwan_china.csv", index=False)
print("✅ Saved to data/gdelt_bigquery_taiwan_china.csv") 