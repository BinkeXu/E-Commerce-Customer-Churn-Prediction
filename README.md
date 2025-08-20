# E-Commerce Customer Churn Prediction

A machine learning project that predicts customer churn using transactional data and RFM analysis.

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Streamlit demo:**
   ```bash
   streamlit run app/streamlit_app.py
   ```

3. **Start the FastAPI server:**
   ```bash
   uvicorn app.main:app --reload
   ```

4. **Open Jupyter notebooks:**
   ```bash
   jupyter notebook notebooks/
   ```

## 📁 Project Structure

```
├── data/                   # Data files
│   └── OnlineRetail.csv   # Raw dataset
├── notebooks/             # Jupyter notebooks
│   ├── 01_eda.ipynb      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Feature creation
│   └── 03_modeling.ipynb # Model training & evaluation
├── src/                   # Source code
│   ├── data/             # Data processing modules
│   ├── features/         # Feature engineering
│   ├── models/           # ML model training
│   └── api/              # FastAPI application
├── app/                   # Application files
│   ├── main.py           # FastAPI app
│   └── streamlit_app.py  # Streamlit demo
├── models/                # Trained models
├── tests/                 # Unit tests
├── docker/                # Docker configuration
└── requirements.txt       # Python dependencies
```

## 🔧 Features

- **RFM Analysis:** Recency, Frequency, Monetary metrics
- **Advanced Features:** Customer lifetime, inter-purchase time, unique products
- **ML Models:** Logistic Regression baseline + LightGBM/XGBoost
- **API:** FastAPI for real-time predictions
- **Demo:** Streamlit web interface
- **Docker:** Containerized deployment

## 📊 Model Performance

The project focuses on:
- **Precision:** Accuracy of churn predictions
- **Recall:** Coverage of actual churners
- **F1-Score:** Balanced performance metric
- **ROC-AUC:** Model discrimination ability

## 🐳 Docker

```bash
docker build -t churn-prediction .
docker run -p 8000:8000 churn-prediction
```

## 📝 License

MIT License - see LICENSE file for details.
