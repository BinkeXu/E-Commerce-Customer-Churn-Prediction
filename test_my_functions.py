import sys
sys.path.append('src')

from data.loader import DataLoader
from features.rfm_features import RFMFeatures
import numpy as np

def Testing_Data_Loader():
    print("=== Testing Data Loader ===")
    
    try:
        # Initialize loader
        loader = DataLoader()
        print("DataLoader initialized successfully")
        
        # Load and clean data
        print("\nLoading and cleaning data...")
        raw_data, customer_data = loader.get_clean_data()
        
        print("\n=== DATA LOADING RESULTS ===")
        print(f"Raw data shape: {raw_data.shape}")
        print(f"Customer data shape: {customer_data.shape}")
        print(f"Total transactions: {len(raw_data):,}")
        print(f"Unique customers: {customer_data['CustomerID'].nunique():,}")
        print(f"Date range: {raw_data['InvoiceDate'].min().strftime('%Y-%m-%d')} to {raw_data['InvoiceDate'].max().strftime('%Y-%m-%d')}")
        print(f"Total revenue: £{raw_data['TotalAmount'].sum():,.2f}")
        print(f"Churn rate: {customer_data['Churned'].mean():.1%}")
        
        print("\n=== SAMPLE CUSTOMER DATA ===")
        print(customer_data.head(10).to_string())
        
        print("\n=== DATA TYPES ===")
        print(raw_data.dtypes)
        
        print("\n=== MISSING VALUES ===")
        print(raw_data.isnull().sum())
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

def test_rfm_features():
    print("=== Testing RFM Features ===")
    
    try:
        # Load data first
        print("Loading data...")
        loader = DataLoader()
        raw_data, customer_data = loader.get_clean_data()
        
        # Initialize RFM features
        print("Initializing RFM features...")
        rfm_calculator = RFMFeatures()
        
        # Calculate all features
        print("Calculating RFM features...")
        features = rfm_calculator.create_all_features(customer_data, raw_data)
        
        print(f"\nFeatures shape: {features.shape}")
        print(f"Feature columns: {list(features.columns)}")
        
        # Show sample features
        print("\n=== Sample Features ===")
        print(features.head(10).to_string())
        
        # Check for any infinite or NaN values
        print("\n=== Data Quality Check ===")
        print(f"Rows with NaN: {features.isnull().sum().sum()}")
        print(f"Rows with infinite values: {np.isinf(features.select_dtypes(include=[np.number])).sum().sum()}")
        
        # Prepare features for modeling
        print("\nPreparing features for modeling...")
        X, y = rfm_calculator.prepare_features_for_modeling(features)
        
        print(f"Final features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        print(f"Target distribution: {y.value_counts().to_dict()}")
        
        # Show feature statistics
        print("\n=== Feature Statistics ===")
        print(X.describe())
        
        return X, y  # Return for use in model testing
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def test_churn_model():
    print("=== Testing Churn Model ===")
    
    try:
        # First get the features
        print("Getting features...")
        X, y = test_rfm_features()
        
        if X is None or y is None:
            print("Failed to get features. Cannot test model.")
            return
        
        # Initialize the churn model
        print("\nInitializing Churn Model...")
        from models.churn_model import ChurnModel
        churn_model = ChurnModel()
        
        # Train and evaluate models
        print("\nTraining and evaluating models...")
        results = churn_model.train_and_evaluate(X, y)
        
        print("\n=== Training Results ===")
        print(f"Best Model: {results['best_model_name']}")
        print(f"Baseline F1: {results['baseline_metrics']['f1_score']:.4f}")
        print(f"LightGBM F1: {results['lgb_metrics']['f1_score']:.4f}")
        
        # Save the best model
        print("\nSaving best model...")
        churn_model.save_model(results['best_model'], "best_model")
        
        print("\n=== Model Testing Complete ===")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # Uncomment the function you want to test
    # Testing_Data_Loader()
    # test_rfm_features()
    test_churn_model()
