import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    roc_auc_score, precision_score, recall_score, f1_score
)
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import joblib
import os
from typing import Tuple, Dict, Any

class ChurnModel:
    """Customer churn prediction model using RFM features."""
    
    def __init__(self, model_dir: str = "final_models"):
        self.model_dir = model_dir
        self.scaler = StandardScaler()
        self.baseline_model = None
        self.lgb_model = None
        self.feature_names = None
        
        # Create models directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
    
    def prepare_data(self, features: pd.DataFrame, target: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing.
        Splits data into training (80%) and testing (20%) sets
        Applies feature scaling using StandardScaler
        Stores feature names for later use
        """
        
        # Store feature names
        self.feature_names = features.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42, stratify=target
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_baseline(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train Logistic Regression baseline model.
        Simple linear model as a baseline
        Uses balanced class weights for imbalanced data
        Performs 5-fold cross-validation
        """
        
        print("Training Logistic Regression baseline model...")
        
        self.baseline_model = LogisticRegression(
            random_state=42, 
            max_iter=1000,
            class_weight='balanced'
        )
        
        # Train the model
        self.baseline_model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.baseline_model, X_train, y_train, 
            cv=5, scoring='f1'
        )
        
        print(f"Baseline model CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, float]:
        """
        Train LightGBM model.
        Advanced gradient boosting model
        Optimized hyperparameters for binary classification
        Also uses 5-fold cross-validation
        """
        
        print("Training LightGBM model...")
        
        # Use LightGBM's scikit-learn API for compatibility
        from lightgbm import LGBMClassifier
        
        self.lgb_model = LGBMClassifier(
            objective='binary',
            metric='binary_logloss',
            boosting_type='gbdt',
            num_leaves=31,
            learning_rate=0.05,
            feature_fraction=0.9,
            bagging_fraction=0.8,
            bagging_freq=5,
            verbose=-1,
            random_state=42,
            n_estimators=1000
        )
        
        # Train the model
        self.lgb_model.fit(X_train, y_train)
        
        # Cross-validation score
        cv_scores = cross_val_score(
            self.lgb_model, X_train, y_train, 
            cv=5, scoring='f1'
        )
        
        print(f"LightGBM model CV F1-score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return {
            'cv_f1_mean': cv_scores.mean(),
            'cv_f1_std': cv_scores.std()
        }
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> Dict[str, float]:
        """
        Evaluate a trained model.
        Calculates key metrics: Precision, Recall, F1-Score, ROC-AUC
        Generates classification reports
        Compares performance between models
        """
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
        }
        
        print(f"\n{model_name} Model Performance:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-Score: {metrics['f1_score']:.4f}")
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        
        # Print classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        return metrics
    
    def train_and_evaluate(self, features: pd.DataFrame, target: pd.Series) -> Dict[str, Any]:
        """
        Complete training and evaluation pipeline.
        Trains both models
        Evaluates on test set
        Automatically selects the best performing model
        Returns comprehensive results
        """
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_data(features, target)
        
        # Train baseline model
        baseline_metrics = self.train_baseline(X_train, y_train)
        baseline_performance = self.evaluate_model(
            self.baseline_model, X_test, y_test, "Logistic Regression"
        )
        
        # Train LightGBM model
        lgb_metrics = self.train_lightgbm(X_train, y_train)
        lgb_performance = self.evaluate_model(
            self.lgb_model, X_test, y_test, "LightGBM"
        )
        
        # Compare models
        print("\n" + "="*50)
        print("MODEL COMPARISON")
        print("="*50)
        print(f"Logistic Regression F1: {baseline_performance['f1_score']:.4f}")
        print(f"LightGBM F1: {lgb_performance['f1_score']:.4f}")
        
        # Select best model
        if lgb_performance['f1_score'] > baseline_performance['f1_score']:
            best_model = self.lgb_model
            best_model_name = "LightGBM"
            print(f"\nBest model: {best_model_name}")
        else:
            best_model = self.baseline_model
            best_model_name = "Logistic Regression"
            print(f"\nBest model: {best_model_name}")
        
        return {
            'baseline_metrics': baseline_performance,
            'lgb_metrics': lgb_performance,
            'best_model': best_model,
            'best_model_name': best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
    
    def save_model(self, model, model_name: str = "best_model"):
        """
        Save the trained model and scaler.
        Saves trained models, scalers, and feature names
        Uses joblib for efficient serialization
        Creates a final_models directory
        """
        
        # Save the model
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        joblib.dump(model, model_path)
        
        # Save the scaler
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
        joblib.dump(self.scaler, scaler_path)
        
        # Save feature names
        features_path = os.path.join(self.model_dir, f"{model_name}_features.json")
        import json
        with open(features_path, 'w') as f:
            json.dump(self.feature_names, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
        print(f"Feature names saved to {features_path}")
    
    def load_model(self, model_name: str = "best_model"):
        """Load a saved model and scaler."""
        
        model_path = os.path.join(self.model_dir, f"{model_name}.joblib")
        scaler_path = os.path.join(self.model_dir, f"{model_name}_scaler.joblib")
        features_path = os.path.join(self.model_dir, f"{model_name}_features.json")
        
        if not all(os.path.exists(p) for p in [model_path, scaler_path, features_path]):
            raise FileNotFoundError("Model files not found")
        
        # Load components
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        with open(features_path, 'r') as f:
            import json
            feature_names = json.load(f)
        
        return model, scaler, feature_names
