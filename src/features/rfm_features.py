import pandas as pd
import numpy as np
from typing import Tuple, Optional

class RFMFeatures:
    """
    Handle feature preparation for modeling.
    Note: Feature calculation logic has been migrated to dbt/BigQuery.
    This class now primarily handles final formatting/selection.
    """
    
    def __init__(self):
        pass
        
    def prepare_features_for_modeling(self, features: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        """
        Prepare features for machine learning modeling.
        Selects numerical features and target variable.
        """
        if features is None:
            return pd.DataFrame(), None
            
        feature_columns = [
            'Recency', 'Frequency', 'Monetary',
            'CustomerLifetime', 'AvgInterPurchaseTime',
            'DaysSinceFirstPurchase', 'AvgOrderValue',
            'ProductsPerOrder', 'SpendingVelocity',
            'RecencyScore', 'FrequencyScore', 'MonetaryScore', 'RFMScore'
        ]
        
        # Ensure columns exist
        available_cols = [c for c in feature_columns if c in features.columns]
        
        # Replace infinite values and drop NaNs
        features_clean = features[available_cols].replace([np.inf, -np.inf], np.nan)
        features_clean = features_clean.dropna()
        
        # Extract target if present
        if 'Churned' in features.columns:
            target = features.loc[features_clean.index, 'Churned'].astype(int)
        else:
            target = None
            
        return features_clean, target
