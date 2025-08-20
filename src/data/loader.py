import pandas as pd
import numpy as np
from typing import Tuple, Optional
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Load and preprocess the OnlineRetail dataset."""
    
    def __init__(self, file_path: str = "OnlineRetail.csv"):
        self.file_path = file_path
        self.data = None
        
    def load_data(self) -> pd.DataFrame:
        """Load the CSV data and perform initial cleaning."""
        try:
            # Try different encodings for the CSV file
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    self.data = pd.read_csv(self.file_path, encoding=encoding)
                    print(f"Data loaded successfully with {encoding} encoding: {self.data.shape}")
                    return self.data
                except UnicodeDecodeError:
                    continue
                   
            # If all encodings fail, try with error handling
            self.data = pd.read_csv(self.file_path, encoding='utf-8', errors='ignore')
            print(f"Data loaded with error handling: {self.data.shape}")
            return self.data
            
        except FileNotFoundError:
            print(f"File not found: {self.file_path}")
            return None
    
    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset by removing invalid entries."""
        if self.data is None:
            self.load_data()
            
        # Remove rows with missing CustomerID
        self.data = self.data.dropna(subset=['CustomerID'])
        
        # Convert CustomerID to integer
        self.data['CustomerID'] = self.data['CustomerID'].astype(int)
        
        # Convert InvoiceDate to datetime
        self.data['InvoiceDate'] = pd.to_datetime(self.data['InvoiceDate'])
        
        # Remove rows with negative quantities or prices
        self.data = self.data[
            (self.data['Quantity'] > 0) & 
            (self.data['UnitPrice'] > 0)
        ]
        
        # Calculate total amount
        self.data['TotalAmount'] = self.data['Quantity'] * self.data['UnitPrice']
        
        # Remove cancelled invoices (those starting with 'C')
        self.data = self.data[~self.data['InvoiceNo'].str.startswith('C', na=False)]
        
        print(f"Data cleaned: {self.data.shape}")
        return self.data
    
    def get_customer_summary(self) -> pd.DataFrame:
        """Get customer-level summary statistics."""
        if self.data is None:
            self.clean_data()
            
        customer_summary = self.data.groupby('CustomerID').agg({
            'InvoiceNo': 'nunique',
            'TotalAmount': 'sum',
            'InvoiceDate': ['min', 'max'],
            'StockCode': 'nunique'
        }).reset_index()
        
        customer_summary.columns = [
            'CustomerID', 'TotalInvoices', 'TotalSpent', 
            'FirstPurchase', 'LastPurchase', 'UniqueProducts'
        ]
        
        return customer_summary
    
    def get_churn_labels(self, days_threshold: int = 90) -> pd.DataFrame:
        """Create churn labels based on 90-day inactivity."""
        customer_summary = self.get_customer_summary()
        
        # Calculate days since last purchase
        max_date = self.data['InvoiceDate'].max()
        customer_summary['DaysSinceLastPurchase'] = (
            max_date - customer_summary['LastPurchase']
        ).dt.days
        
        # Create churn label (1 = churned, 0 = active)
        customer_summary['Churned'] = (
            customer_summary['DaysSinceLastPurchase'] > days_threshold
        ).astype(int)
        
        return customer_summary
    
    def get_clean_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Get both raw data and customer summary with churn labels."""
        raw_data = self.clean_data()
        customer_data = self.get_churn_labels()
        return raw_data, customer_data
