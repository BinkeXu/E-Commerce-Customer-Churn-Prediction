import pandas as pd
import numpy as np
from typing import Tuple
from datetime import datetime, timedelta

class RFMFeatures:
    """Calculate RFM (Recency, Frequency, Monetary) features for customer churn prediction."""
    
    def __init__(self, reference_date: datetime = None):
        self.reference_date = reference_date
        
    def calculate_rfm(self, customer_data: pd.DataFrame, 
                     raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates core RFM metrics for each customer
        Recency: Days since last purchase (lower = better)
        Frequency: Total number of unique purchases (higher = better)
        Monetary: Total amount spent across all transactions (higher = better)
        """

        if self.reference_date is None:
            self.reference_date = raw_data['InvoiceDate'].max()
            
        # Group by customer to get RFM metrics
        rfm = raw_data.groupby('CustomerID').agg({
            'InvoiceDate': lambda x: (self.reference_date - x.max()).days,  # Recency
            'InvoiceNo': 'nunique',  # Frequency
            'TotalAmount': 'sum'      # Monetary
        }).reset_index()
        
        rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
        
        # Merge with customer data
        rfm_features = customer_data.merge(rfm, on='CustomerID', how='left')
        
        return rfm_features
    
    def calculate_time_features(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-based features.
        Customer Lifetime: Days between first and last purchase
        Average Inter-Purchase Time: Average days between consecutive purchases
        Days Since First Purchase: Total time since customer started
        """

        features = customer_data.copy()
        
        # Customer lifetime (days between first and last purchase)
        features['CustomerLifetime'] = (
            features['LastPurchase'] - features['FirstPurchase']
        ).dt.days
        
        # Average inter-purchase time
        features['AvgInterPurchaseTime'] = features['CustomerLifetime'] / (
            features['TotalInvoices'] - 1
        )
        features['AvgInterPurchaseTime'] = features['AvgInterPurchaseTime'].fillna(0)
        
        # Days since first purchase
        features['DaysSinceFirstPurchase'] = (
            self.reference_date - features['FirstPurchase']
        ).dt.days
        
        return features
    
    def calculate_behavioral_features(self, customer_data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate behavioral features.
        Average Order Value: Total spent ÷ number of orders
        Products Per Order: Unique products ÷ number of orders
        Spending Velocity: Total spent ÷ days since first purchase
        """
        
        features = customer_data.copy()
        
        # Average order value
        features['AvgOrderValue'] = features['TotalSpent'] / features['TotalInvoices']
        
        # Products per order
        features['ProductsPerOrder'] = features['UniqueProducts'] / features['TotalInvoices']
        
        # Spending velocity (total spent per day since first purchase)
        features['SpendingVelocity'] = features['TotalSpent'] / (
            features['DaysSinceFirstPurchase'] + 1
        )
        
        return features
    
    def create_all_features(self, customer_data: pd.DataFrame, 
                          raw_data: pd.DataFrame) -> pd.DataFrame:
        """Create all features for the dataset."""
        
        # Calculate RFM features
        features = self.calculate_rfm(customer_data, raw_data)
        
        # Add time-based features
        features = self.calculate_time_features(features)
        
        # Add behavioral features
        features = self.calculate_behavioral_features(features)
        
        # Create RFM scores (1-5 scale)
        features = self._create_rfm_scores(features)
        
        return features
    
    def _create_rfm_scores(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Converts raw RFM values to 1-5 scale scores
        Recency Score: 5 (recent) to 1 (old) - lower recency is better
        Frequency Score: 1 (low) to 5 (high) - higher frequency is better
        Monetary Score: 1 (low) to 5 (high) - higher spending is better
        Combined RFM Score: Sum of all three scores (3-15 range)
        """
        
        # Recency score (lower is better)
        try:
            features['RecencyScore'] = pd.qcut(
                features['Recency'], 
                q=5, 
                labels=[5, 4, 3, 2, 1],
                duplicates='drop'
            ).astype(int)
        except ValueError:
            # Fallback: use rank-based scoring
            features['RecencyScore'] = pd.qcut(
                features['Recency'].rank(method='first'), 
                q=5, 
                labels=[5, 4, 3, 2, 1]
            ).astype(int)
        
        # Frequency score (higher is better)
        try:
            features['FrequencyScore'] = pd.qcut(
                features['Frequency'], 
                q=5, 
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            ).astype(int)
        except ValueError:
            # Fallback: use rank-based scoring
            features['FrequencyScore'] = pd.qcut(
                features['Frequency'].rank(method='first'), 
                q=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
        
        # Monetary score (higher is better)
        try:
            features['MonetaryScore'] = pd.qcut(
                features['Monetary'], 
                q=5, 
                labels=[1, 2, 3, 4, 5],
                duplicates='drop'
            ).astype(int)
        except ValueError:
            # Fallback: use rank-based scoring
            features['MonetaryScore'] = pd.qcut(
                features['Monetary'].rank(method='first'), 
                q=5, 
                labels=[1, 2, 3, 4, 5]
            ).astype(int)
        
        # Combined RFM score
        features['RFMScore'] = (
            features['RecencyScore'] + 
            features['FrequencyScore'] + 
            features['MonetaryScore']
        )
        
        return features
    
    def prepare_features_for_modeling(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features for machine learning modeling.
        Selects 12 numerical features for machine learning
        Removes rows with infinite/NaN values
        Returns clean feature matrix and target variable (churn status)
        """
        
        # Select numerical features
        feature_columns = [
            'Recency', 'Frequency', 'Monetary',
            'CustomerLifetime', 'AvgInterPurchaseTime',
            'DaysSinceFirstPurchase', 'AvgOrderValue',
            'ProductsPerOrder', 'SpendingVelocity',
            'RecencyScore', 'FrequencyScore', 'MonetaryScore', 'RFMScore'
        ]
        
        # Remove rows with infinite or NaN values
        features_clean = features[feature_columns].replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.dropna()
        
        # Get corresponding target variable
        target = features.loc[features_clean.index, 'Churned']
        
        return features_clean, target
