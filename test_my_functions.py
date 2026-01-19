import sys
sys.path.append('src')

from data.loader import DataLoader
from features.rfm_features import RFMFeatures
from models.churn_model import ChurnModel
import numpy as np
import pandas as pd

def test_data_loading():
    print("=== Testing Data Loader (BigQuery) ===")
    
    try:
        loader = DataLoader()
        print("DataLoader initialized")
        
        # Load features directly from (mocked or real) BigQuery
        print("Fetching features from BigQuery...")
        features = loader.load_features_from_bq()
        
        if features is not None:
            print(f"Features loaded: {features.shape}")
            print("\n=== Sample Features ===")
            print(features.head().to_string())
            return features
        else:
            print("Failed to load features from BigQuery")
            return None
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_training_flow():
    print("\n=== Testing Training Flow ===")
    
    try:
        # 1. Load Data
        features = test_data_loading()
        if features is None:
            return
            
        # 2. Prepare for modeling
        print("\nPreparing features for modeling...")
        rfm_calculator = RFMFeatures()
        X, y = rfm_calculator.prepare_features_for_modeling(features)
        
        if X is None:
            print("Failed to prepare features")
            return
            
        print(f"Final X shape: {X.shape}")
        if y is not None:
            print(f"Target distribution: {y.value_counts().to_dict()}")
            
        # 3. Train Model
        print("\nInitializing Churn Model...")
        churn_model = ChurnModel()
        
        print("Training and evaluating...")
        results = churn_model.train_and_evaluate(X, y)
        
        print("\n=== Training Results ===")
        print(f"Best Model: {results['best_model_name']}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_training_flow()
