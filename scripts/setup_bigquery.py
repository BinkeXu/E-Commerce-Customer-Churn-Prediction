
import os
from google.cloud import bigquery
import pandas as pd
from google.oauth2 import service_account

# Configuration
KEY_PATH = "secrets/google-key.json"
PROJECT_ID = "airy-web-484800-u5"
DATASET_RAW = "raw_ecommerce"
DATASET_DBT = "ecommerce_churn"
CSV_PATH = "OnlineRetail.csv"
LOCATION = "US"

def setup_bigquery():
    print("=== Setting up BigQuery Infrastructure ===")
    
    # Authenticate
    if not os.path.exists(KEY_PATH):
        print(f"Error: Key file not found at {KEY_PATH}")
        return False
        
    credentials = service_account.Credentials.from_service_account_file(KEY_PATH)
    client = bigquery.Client(credentials=credentials, project=PROJECT_ID)
    
    # 1. Create Datasets
    for dataset_id in [DATASET_RAW, DATASET_DBT]:
        dataset_ref = f"{PROJECT_ID}.{dataset_id}"
        try:
            client.get_dataset(dataset_ref)
            print(f"Dataset {dataset_id} already exists.")
        except Exception:
            print(f"Creating dataset {dataset_id}...")
            dataset = bigquery.Dataset(dataset_ref)
            dataset.location = LOCATION
            client.create_dataset(dataset)
            print(f"Dataset {dataset_id} created.")

    # 2. Upload CSV
    table_ref = f"{PROJECT_ID}.{DATASET_RAW}.OnlineRetail"
    try:
        client.get_table(table_ref)
        print(f"Table {table_ref} already exists. Skipping upload.")
    except Exception:
        if not os.path.exists(CSV_PATH):
            print(f"Error: CSV file not found at {CSV_PATH}")
            return False
            
        print(f"Uploading {CSV_PATH} to {table_ref}...")
        
        # Load CSV
        # Using autodetect for schema for simplicity, or we can define it. 
        # Given potential encoding issues seen in loader.py, we try reading with pandas first to be safe
        try:
            # Try iso-8859-1 which is common for this dataset
            df = pd.read_csv(CSV_PATH, encoding='iso-8859-1')
        except Exception as e:
            print(f"Error reading CSV: {e}")
            return False
            
        # Upload
        # Simple upload using pandas-gbq or client.load_table_from_dataframe
        job_config = bigquery.LoadJobConfig(
            write_disposition="WRITE_TRUNCATE",
        )
        
        job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
        job.result()  # Wait for completion
        print(f"Uploaded {len(df)} rows to {table_ref}.")

    print("=== Infrastructure Setup Complete ===")
    return True

if __name__ == "__main__":
    setup_bigquery()
