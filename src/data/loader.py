import pandas as pd
from google.cloud import bigquery
import os
from typing import Optional

class DataLoader:
    """Load data from BigQuery."""
    
    def __init__(self):
        # Credentials should be set via GOOGLE_APPLICATION_CREDENTIALS env var
        # or it will default to default credentials
        self.client = bigquery.Client()
        
    def load_features_from_bq(self, table_id: str = "ecommerce_churn.fct_churn_features") -> Optional[pd.DataFrame]:
        """Fetch the pre-calculated features from BigQuery."""
        query = f"""
            SELECT *
            FROM `{table_id}`
        """
        try:
            print(f"Fetching data from {table_id}...")
            df = self.client.query(query).to_dataframe()
            print(f"Data loaded from BigQuery: {df.shape}")
            return df
        except Exception as e:
            print(f"Error loading data from BigQuery: {e}")
            return None
