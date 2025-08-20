import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import json
from datetime import datetime, timedelta
import sys
import os

# Add src to path for imports
sys.path.append('src')

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        color: white;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
</style>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🔄 Customer Churn Prediction</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("## 🎯 Navigation")
    page = st.selectbox(
        "Choose a page:",
        ["🏠 Dashboard", "🔮 Single Prediction", "📊 Batch Analysis", "📈 Model Performance", "⚙️ Settings"]
    )
    
    st.markdown("---")
    st.markdown("## 📊 Quick Stats")
    
    # Placeholder for quick stats
    if 'api_connected' not in st.session_state:
        st.session_state.api_connected = False
    
    if st.session_state.api_connected:
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                st.success("✅ API Connected")
                st.metric("Model Status", "Loaded" if response.json()["model_loaded"] else "Not Loaded")
            else:
                st.error("❌ API Error")
        except:
            st.error("❌ API Unreachable")
    else:
        st.info("🔌 Connect to API")

# Main content based on page selection
if page == "🏠 Dashboard":
    st.markdown("## 📊 Project Overview")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Objective</h3>
            <p>Predict customer churn using RFM analysis</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🔍 Features</h3>
            <p>RFM + Behavioral metrics</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Models</h3>
            <p>Logistic Regression + LightGBM</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Data overview
    st.markdown("## 📈 Dataset Overview")
    
    try:
        # Load sample data for visualization
        from src.data.loader import DataLoader
        
        loader = DataLoader()
        raw_data, customer_data = loader.get_clean_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Raw Data Statistics")
            st.write(f"**Total Transactions:** {len(raw_data):,}")
            st.write(f"**Unique Customers:** {customer_data['CustomerID'].nunique():,}")
            st.write(f"**Date Range:** {raw_data['InvoiceDate'].min().strftime('%Y-%m-%d')} to {raw_data['InvoiceDate'].max().strftime('%Y-%m-%d')}")
            st.write(f"**Total Revenue:** £{raw_data['TotalAmount'].sum():,.2f}")
        
        with col2:
            st.markdown("### 🎯 Churn Distribution")
            churn_counts = customer_data['Churned'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=['Active', 'Churned'],
                title="Customer Churn Status",
                color_discrete_sequence=['#00ff88', '#ff6b6b']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # RFM Distribution
        st.markdown("### 📊 RFM Metrics Distribution")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            fig = px.histogram(
                customer_data, 
                x='TotalSpent', 
                nbins=30,
                title="Monetary Distribution",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.histogram(
                customer_data, 
                x='TotalInvoices', 
                nbins=30,
                title="Frequency Distribution",
                color_discrete_sequence=['#764ba2']
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col3:
            fig = px.histogram(
                customer_data, 
                x='DaysSinceLastPurchase', 
                nbins=30,
                title="Recency Distribution",
                color_discrete_sequence=['#f093fb']
            )
            st.plotly_chart(fig, use_container_width=True)
            
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("Please ensure the data file is available and the modules are properly set up.")

elif page == "🔮 Single Prediction":
    st.markdown("## 🔮 Single Customer Churn Prediction")
    
    # API connection status
    if st.button("🔌 Connect to API", key="connect_api"):
        try:
            response = requests.get("http://localhost:8000/health")
            if response.status_code == 200:
                st.session_state.api_connected = True
                st.success("✅ Successfully connected to API!")
            else:
                st.error("❌ API returned an error")
        except:
            st.error("❌ Could not connect to API. Make sure it's running on localhost:8000")
    
    if st.session_state.api_connected:
        st.markdown("### 📝 Enter Customer Features")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Time-based Features")
            recency = st.number_input("Days since last purchase", min_value=0, max_value=1000, value=30)
            frequency = st.number_input("Total number of purchases", min_value=1, max_value=100, value=5)
            customer_lifetime = st.number_input("Customer lifetime (days)", min_value=1, max_value=2000, value=365)
            days_since_first = st.number_input("Days since first purchase", min_value=1, max_value=2000, value=400)
        
        with col2:
            st.markdown("#### 💰 Financial Features")
            monetary = st.number_input("Total amount spent", min_value=0.0, max_value=10000.0, value=1500.0)
            avg_order_value = st.number_input("Average order value", min_value=0.0, max_value=1000.0, value=300.0)
            unique_products = st.number_input("Unique products purchased", min_value=1, max_value=100, value=8)
            spending_velocity = st.number_input("Spending velocity", min_value=0.0, max_value=100.0, value=4.1)
        
        # Calculate derived features
        avg_inter_purchase = customer_lifetime / (frequency - 1) if frequency > 1 else 0
        products_per_order = unique_products / frequency if frequency > 0 else 0
        
        # RFM Scores (simplified calculation)
        recency_score = max(1, min(5, 6 - (recency // 30)))
        frequency_score = max(1, min(5, (frequency // 2) + 1))
        monetary_score = max(1, min(5, (int(monetary // 500)) + 1))
        rfm_score = recency_score + frequency_score + monetary_score
        
        if st.button("🔮 Predict Churn", key="predict_single"):
            try:
                # Prepare features for API
                features = {
                    "recency": float(recency),
                    "frequency": float(frequency),
                    "monetary": float(monetary),
                    "customer_lifetime": float(customer_lifetime),
                    "avg_inter_purchase_time": float(avg_inter_purchase),
                    "days_since_first_purchase": float(days_since_first),
                    "avg_order_value": float(avg_order_value),
                    "products_per_order": float(products_per_order),
                    "spending_velocity": float(spending_velocity),
                    "recency_score": int(recency_score),
                    "frequency_score": int(frequency_score),
                    "monetary_score": int(monetary_score),
                    "rfm_score": int(rfm_score)
                }
                
                # Make prediction
                response = requests.post("http://localhost:8000/predict", json=features)
                
                if response.status_code == 200:
                    prediction = response.json()
                    
                    st.markdown("### 🎯 Prediction Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Churn Probability", f"{prediction['churn_probability']:.1%}")
                    
                    with col2:
                        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        st.metric("Risk Level", f"{risk_color[prediction['risk_level']]} {prediction['risk_level']}")
                    
                    with col3:
                        st.metric("Confidence", f"{prediction['confidence']:.1%}")
                    
                    # Visual representation
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=prediction['churn_probability'] * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Churn Risk Gauge"},
                        delta={'reference': 50},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "lightgreen"},
                                {'range': [30, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 70
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error(f"API Error: {response.text}")
                    
            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
    else:
        st.info("🔌 Please connect to the API first to make predictions.")

elif page == "📊 Batch Analysis":
    st.markdown("## 📊 Batch Customer Analysis")
    
    if not st.session_state.api_connected:
        st.warning("🔌 Please connect to the API first from the Single Prediction page.")
        return
    
    st.markdown("### 📁 Upload Customer Data")
    
    uploaded_file = st.file_uploader(
        "Upload CSV file with customer features", 
        type=['csv'],
        help="CSV should contain columns: recency, frequency, monetary, etc."
    )
    
    if uploaded_file is not None:
        try:
            # Load the data
            df = pd.read_csv(uploaded_file)
            st.write("### 📋 Preview of uploaded data:")
            st.dataframe(df.head())
            
            if st.button("🔮 Analyze All Customers"):
                # Convert to API format
                customers = []
                for _, row in df.iterrows():
                    customer = {
                        "recency": float(row.get('recency', 0)),
                        "frequency": float(row.get('frequency', 1)),
                        "monetary": float(row.get('monetary', 0)),
                        "customer_lifetime": float(row.get('customer_lifetime', 365)),
                        "avg_inter_purchase_time": float(row.get('avg_inter_purchase_time', 0)),
                        "days_since_first_purchase": float(row.get('days_since_first_purchase', 400)),
                        "avg_order_value": float(row.get('avg_order_value', 0)),
                        "products_per_order": float(row.get('products_per_order', 1)),
                        "spending_velocity": float(row.get('spending_velocity', 0)),
                        "recency_score": int(row.get('recency_score', 3)),
                        "frequency_score": int(row.get('frequency_score', 3)),
                        "monetary_score": int(row.get('monetary_score', 3)),
                        "rfm_score": int(row.get('rfm_score', 9))
                    }
                    customers.append(customer)
                
                # Make batch prediction
                response = requests.post("http://localhost:8000/predict_batch", json=customers)
                
                if response.status_code == 200:
                    results = response.json()
                    
                    st.markdown("### 📊 Batch Analysis Results")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Customers", results['total_customers'])
                    
                    with col2:
                        st.metric("Predicted Churn Rate", f"{results['churn_rate']:.1%}")
                    
                    with col3:
                        churn_count = sum(1 for p in results['predictions'] if p['churn_prediction'])
                        st.metric("Churn Count", churn_count)
                    
                    # Create results DataFrame
                    results_df = pd.DataFrame(results['predictions'])
                    
                    # Risk level distribution
                    st.markdown("### 🎯 Risk Level Distribution")
                    risk_counts = results_df['risk_level'].value_counts()
                    fig = px.bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        title="Customer Risk Level Distribution",
                        color_discrete_sequence=['#667eea', '#764ba2', '#f093fb']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Churn probability distribution
                    st.markdown("### 📈 Churn Probability Distribution")
                    fig = px.histogram(
                        results_df,
                        x='churn_probability',
                        nbins=20,
                        title="Distribution of Churn Probabilities",
                        color_discrete_sequence=['#667eea']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results CSV",
                        data=csv,
                        file_name="churn_predictions.csv",
                        mime="text/csv"
                    )
                    
                else:
                    st.error(f"API Error: {response.text}")
                    
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

elif page == "📈 Model Performance":
    st.markdown("## 📈 Model Performance & Insights")
    
    if not st.session_state.api_connected:
        st.warning("🔌 Please connect to the API first to view model performance.")
        return
    
    try:
        # Get model info
        response = requests.get("http://localhost:8000/model_info")
        
        if response.status_code == 200:
            model_info = response.json()
            
            st.markdown("### 🤖 Model Information")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Model Type", model_info['model_type'])
            
            with col2:
                st.metric("Number of Features", model_info['num_features'])
            
            with col3:
                st.metric("Scaler Type", model_info['scaler_type'])
            
            st.markdown("### 🔍 Feature Names")
            st.write(", ".join(model_info['feature_names']))
            
            # Placeholder for model performance metrics
            st.markdown("### 📊 Performance Metrics")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Precision", "0.85", delta="+0.02")
            
            with col2:
                st.metric("Recall", "0.78", delta="-0.01")
            
            with col3:
                st.metric("F1-Score", "0.81", delta="+0.01")
            
            with col4:
                st.metric("ROC-AUC", "0.89", delta="+0.03")
            
            # Feature importance (placeholder)
            st.markdown("### 🎯 Feature Importance")
            
            # Mock feature importance data
            feature_importance = pd.DataFrame({
                'Feature': model_info['feature_names'],
                'Importance': np.random.rand(len(model_info['feature_names']))
            }).sort_values('Importance', ascending=True)
            
            fig = px.barh(
                feature_importance,
                x='Importance',
                y='Feature',
                title="Feature Importance (Mock Data)",
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.error(f"Error getting model info: {response.text}")
            
    except Exception as e:
        st.error(f"Error: {str(e)}")

elif page == "⚙️ Settings":
    st.markdown("## ⚙️ Application Settings")
    
    st.markdown("### 🔌 API Configuration")
    
    api_url = st.text_input(
        "API Base URL",
        value="http://localhost:8000",
        help="Base URL for the FastAPI server"
    )
    
    if st.button("🔄 Test Connection"):
        try:
            response = requests.get(f"{api_url}/health")
            if response.status_code == 200:
                st.success("✅ Connection successful!")
            else:
                st.error("❌ Connection failed")
        except:
            st.error("❌ Could not connect to the API")
    
    st.markdown("### 📊 Data Configuration")
    
    churn_threshold = st.slider(
        "Churn Threshold (days)",
        min_value=30,
        max_value=180,
        value=90,
        help="Number of days without purchase to consider a customer as churned"
    )
    
    st.markdown("### 🎨 Display Settings")
    
    theme = st.selectbox(
        "Theme",
        ["Light", "Dark"],
        help="Choose the display theme"
    )
    
    st.markdown("### 📁 File Paths")
    
    data_path = st.text_input(
        "Data File Path",
        value="data/OnlineRetail.csv",
        help="Path to the OnlineRetail dataset"
    )
    
    model_path = st.text_input(
        "Model Directory",
        value="models",
        help="Directory to store trained models"
    )

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using Streamlit, FastAPI, and Machine Learning
    </div>
    """,
    unsafe_allow_html=True
)
