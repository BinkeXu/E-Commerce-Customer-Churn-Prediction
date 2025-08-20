from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os
from typing import List, Dict, Any
import uvicorn

# Import our modules
import sys
sys.path.append('src')
from models.churn_model import ChurnModel

app = FastAPI(
    title="E-Commerce Customer Churn Prediction API",
    description="API for predicting customer churn using RFM analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    recency: float
    frequency: float
    monetary: float
    customer_lifetime: float
    avg_inter_purchase_time: float
    days_since_first_purchase: float
    avg_order_value: float
    products_per_order: float
    spending_velocity: float
    recency_score: int
    frequency_score: int
    monetary_score: int
    rfm_score: int

class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float
    churn_prediction: bool
    risk_level: str
    confidence: float

class BatchPrediction(BaseModel):
    predictions: List[ChurnPrediction]
    total_customers: int
    churn_rate: float

# Global variables for model
model = None
scaler = None
feature_names = None

@app.on_event("startup")
async def load_model():
    """Load the trained model on startup."""
    global model, scaler, feature_names
    
    try:
        churn_model = ChurnModel()
        model, scaler, feature_names = churn_model.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please train the model first using the notebooks.")

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "E-Commerce Customer Churn Prediction API",
        "version": "1.0.0",
        "status": "running"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None
    }

@app.post("/predict", response_model=ChurnPrediction)
async def predict_churn(customer_features: CustomerFeatures):
    """Predict churn for a single customer."""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        # Convert features to DataFrame
        features_dict = customer_features.dict()
        features_df = pd.DataFrame([features_dict])
        
        # Ensure correct feature order
        features_df = features_df[feature_names]
        
        # Scale features
        features_scaled = scaler.transform(features_df)
        
        # Make prediction
        churn_prob = model.predict_proba(features_scaled)[0, 1]
        churn_pred = churn_prob > 0.5
        
        # Determine risk level
        if churn_prob < 0.3:
            risk_level = "Low"
        elif churn_prob < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"
        
        # Calculate confidence (distance from decision boundary)
        confidence = abs(churn_prob - 0.5) * 2
        
        return ChurnPrediction(
            customer_id="CUSTOMER_001",
            churn_probability=float(churn_prob),
            churn_prediction=bool(churn_pred),
            risk_level=risk_level,
            confidence=float(confidence)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/predict_batch", response_model=BatchPrediction)
async def predict_churn_batch(customers: List[CustomerFeatures]):
    """Predict churn for multiple customers."""
    
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded. Please train the model first.")
    
    try:
        predictions = []
        
        for i, customer in enumerate(customers):
            # Convert features to DataFrame
            features_dict = customer.dict()
            features_df = pd.DataFrame([features_dict])
            
            # Ensure correct feature order
            features_df = features_df[feature_names]
            
            # Scale features
            features_scaled = scaler.transform(features_df)
            
            # Make prediction
            churn_prob = model.predict_proba(features_scaled)[0, 1]
            churn_pred = churn_prob > 0.5
            
            # Determine risk level
            if churn_prob < 0.3:
                risk_level = "Low"
            elif churn_prob < 0.7:
                risk_level = "Medium"
            else:
                risk_level = "High"
            
            # Calculate confidence
            confidence = abs(churn_prob - 0.5) * 2
            
            predictions.append(ChurnPrediction(
                customer_id=f"CUSTOMER_{i+1:03d}",
                churn_probability=float(churn_prob),
                churn_prediction=bool(churn_pred),
                risk_level=risk_level,
                confidence=float(confidence)
            ))
        
        # Calculate batch statistics
        total_customers = len(predictions)
        churn_rate = sum(1 for p in predictions if p.churn_prediction) / total_customers
        
        return BatchPrediction(
            predictions=predictions,
            total_customers=total_customers,
            churn_rate=float(churn_rate)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")

@app.get("/model_info")
async def get_model_info():
    """Get information about the loaded model."""
    
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_type = type(model).__name__
    
    return {
        "model_type": model_type,
        "feature_names": feature_names,
        "num_features": len(feature_names) if feature_names else 0,
        "scaler_type": type(scaler).__name__ if scaler else None
    }

@app.get("/example_features")
async def get_example_features():
    """Get example customer features for testing the API."""
    
    example_features = CustomerFeatures(
        recency=30.0,
        frequency=5.0,
        monetary=1500.0,
        customer_lifetime=365.0,
        avg_inter_purchase_time=73.0,
        days_since_first_purchase=400.0,
        avg_order_value=300.0,
        products_per_order=8.0,
        spending_velocity=4.1,
        recency_score=4,
        frequency_score=3,
        monetary_score=4,
        rfm_score=11
    )
    
    return {
        "example_features": example_features,
        "description": "Use these features to test the /predict endpoint"
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
